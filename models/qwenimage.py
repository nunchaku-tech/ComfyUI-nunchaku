"""
This module implements the Nunchaku Qwen-Image model and related components.

.. note::

    Inherits and modifies from https://github.com/comfyanonymous/ComfyUI/blob/v0.3.51/comfy/ldm/qwen_image/model.py
"""

import gc
from typing import Optional, Tuple

import torch
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.modules.attention import optimized_attention_masked
from comfy.ldm.qwen_image.model import (
    GELU,
    FeedForward,
    LastLayer,
    QwenImageTransformer2DModel,
    QwenTimestepProjEmbeddings,
    apply_rotary_emb,
)
from torch import nn

from nunchaku.models.linear import AWQW4A16Linear, SVDQW4A4Linear
from nunchaku.models.utils import CPUOffloadManager
from nunchaku.ops.fused import fused_gelu_mlp

from ..mixins.model import NunchakuModelMixin


class LoRAConfigContainer(nn.Module):
    """
    Lightweight container for LoRA configuration.
    
    This class acts as a transparent proxy to the transformer,
    storing only the LoRA configuration separately for each model copy.
    All method calls and attribute access are forwarded to the transformer.
    
    This design avoids the problems encountered with full wrapper implementations:
    - No need to customize forward() for parameter name conversion
    - No need to handle 5D/4D dimension mismatches
    - No need to implement to_safely() and other ComfyUI methods
    - No type checking failures
    
    Inherits from nn.Module to satisfy PyTorch's module hierarchy requirements.
    
    Attributes
    ----------
    _transformer : NunchakuQwenImageTransformer2DModel
        The shared transformer instance (contains LoRA cache).
    _lora_config_list : list
        Independent LoRA configuration for this container.
    
    Examples
    --------
    >>> transformer = NunchakuQwenImageTransformer2DModel(...)
    >>> container = LoRAConfigContainer(transformer)
    >>> container._lora_config_list.append(("path/to/lora.safetensors", 1.0))
    >>> # All other attributes/methods transparently forwarded to transformer
    >>> output = container(x, timestep, context, ...)  # Calls transformer's forward
    """
    
    def __init__(self, transformer):
        """
        Initialize the container with a transformer instance.
        
        Parameters
        ----------
        transformer : NunchakuQwenImageTransformer2DModel
            The transformer to wrap.
        """
        super().__init__()
        # Use object.__setattr__ to bypass nn.Module's __setattr__ for private attributes
        object.__setattr__(self, '_transformer', transformer)
        object.__setattr__(self, '_lora_config_list', [])
    
    def __getattr__(self, name):
        """
        Forward all attribute access to the transformer.
        
        This makes the container transparent for all operations
        except accessing _transformer and _lora_config_list.
        
        Note: This is called AFTER checking self.__dict__ and self.__class__.__dict__,
        so it won't interfere with nn.Module's internal attributes.
        """
        # Avoid recursion for _transformer
        if name == '_transformer':
            return object.__getattribute__(self, '_transformer')
        return getattr(object.__getattribute__(self, '_transformer'), name)
    
    def __setattr__(self, name, value):
        """
        Store private attributes in container, everything else in transformer.
        
        Private attributes (starting with '_') are stored in the container itself.
        All other attributes are forwarded to the transformer.
        """
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, '_transformer'), name, value)
    
    def forward(self, *args, **kwargs):
        """
        Forward pass - handles LoRA composition then delegates to transformer.
        
        This is the entry point when ComfyUI calls the model.
        We inject the LoRA config into the transformer before calling it.
        """
        # Temporarily inject LoRA config into transformer for this forward pass
        transformer = object.__getattribute__(self, '_transformer')
        lora_config_list = object.__getattribute__(self, '_lora_config_list')
        
        # Save original config (if any)
        original_config = getattr(transformer, '_lora_config_list', None)
        
        # Inject our config
        transformer._lora_config_list = lora_config_list
        
        try:
            # Call transformer's forward
            result = transformer(*args, **kwargs)
        finally:
            # Restore original config
            if original_config is not None:
                transformer._lora_config_list = original_config
            elif hasattr(transformer, '_lora_config_list'):
                delattr(transformer, '_lora_config_list')
        
        return result


class NunchakuGELU(GELU):
    """
    GELU activation with a quantized linear projection.

    Parameters
    ----------
    dim_in : int
        Input feature dimension.
    dim_out : int
        Output feature dimension.
    approximate : str, optional
        Approximation mode for GELU (default: "none").
    bias : bool, optional
        Whether to use bias in the projection (default: True).
    dtype : torch.dtype, optional
        Data type for the projection.
    device : torch.device, optional
        Device for the projection.
    **kwargs
        Additional arguments for the quantized linear layer.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        approximate: str = "none",
        bias: bool = True,
        dtype=None,
        device=None,
        **kwargs,
    ):
        super(GELU, self).__init__()
        self.proj = SVDQW4A4Linear(dim_in, dim_out, bias=bias, torch_dtype=dtype, device=device, **kwargs)
        self.approximate = approximate


class NunchakuFeedForward(FeedForward):
    """
    Feed-forward network with fused quantized layers and optional fused GELU-MLP.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    dim_out : int, optional
        Output feature dimension. If None, set to `dim`.
    mult : int, optional
        Expansion factor for the hidden layer (default: 4).
    dropout : float, optional
        Dropout probability (default: 0.0).
    inner_dim : int, optional
        Hidden layer dimension. If None, computed as `dim * mult`.
    bias : bool, optional
        Whether to use bias in the projections (default: True).
    dtype : torch.dtype, optional
        Data type for the projections.
    device : torch.device, optional
        Device for the projections.
    **kwargs
        Additional arguments for the quantized linear layers.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        inner_dim=None,
        bias: bool = True,
        dtype=None,
        device=None,
        **kwargs,
    ):
        super(FeedForward, self).__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.net = nn.ModuleList([])
        self.net.append(
            NunchakuGELU(dim, inner_dim, approximate="tanh", bias=bias, dtype=dtype, device=device, **kwargs)
        )
        self.net.append(nn.Dropout(dropout))
        self.net.append(
            SVDQW4A4Linear(
                inner_dim,
                dim_out,
                bias=bias,
                act_unsigned=kwargs["precision"]
                == "int4",  # For int4 quantization, the second linear layer is unsigned as the output of the first is shifted positive in fused_gelu_mlp
                torch_dtype=dtype,
                device=device,
                **kwargs,
            )
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feed-forward network.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor of shape (batch, seq_len, dim).

        Returns
        -------
        torch.Tensor
            Output tensor after feed-forward transformation.
        """
        if isinstance(self.net[0], NunchakuGELU):
            return fused_gelu_mlp(hidden_states, self.net[0].proj, self.net[2])
        else:
            # Fallback to original implementation
            for module in self.net:
                hidden_states = module(hidden_states)
            return hidden_states
    
    def update_lora_params(self, lora_dict: dict[str, torch.Tensor]):
        """
        Update LoRA parameters for the feed-forward network.
        """
        from nunchaku.lora.qwenimage.packer import unpack_lowrank_weight
        import logging
        logger = logging.getLogger(__name__)
        
        # Helper function to apply LoRA to a SVDQW4A4Linear layer
        def apply_lora_to_linear(linear_layer, lora_dict, layer_prefix):
            lora_down_key = None
            lora_up_key = None
            
            # Find lora_down and lora_up for this layer
            for k in lora_dict.keys():
                if layer_prefix in k:
                    if 'lora_down' in k:
                        lora_down_key = k
                    elif 'lora_up' in k:
                        lora_up_key = k
            
            if lora_down_key is None or lora_up_key is None:
                return False
            
            lora_down_packed = lora_dict[lora_down_key]
            lora_up_packed = lora_dict[lora_up_key]
            
            # The LoRA weights are already merged with original low-rank branches in the converter
            # Just directly apply them
            device = linear_layer.proj_down.device
            dtype = linear_layer.proj_down.dtype
            old_rank = linear_layer.rank
            
            # Directly replace parameters with merged weights
            linear_layer.proj_down.data = lora_down_packed.to(device=device, dtype=dtype)
            linear_layer.proj_up.data = lora_up_packed.to(device=device, dtype=dtype)
            linear_layer.rank = lora_down_packed.shape[1]
            
            return True
        
        # Apply LoRA to each SVDQW4A4Linear layer in the network
        for i, module in enumerate(self.net):
            if isinstance(module, SVDQW4A4Linear):
                apply_lora_to_linear(module, lora_dict, f'net.{i}')
            elif isinstance(module, NunchakuGELU) and hasattr(module, 'proj') and isinstance(module.proj, SVDQW4A4Linear):
                # For GELU with proj attribute
                apply_lora_to_linear(module.proj, lora_dict, f'net.{i}.proj')
    
    def restore_original_params(self):
        """
        Restore original parameters for all quantized linear layers in the feed-forward network.
        """
        def restore_linear_layer(linear_layer, layer_prefix):
            if hasattr(linear_layer, '_original_proj_down'):
                linear_layer.proj_down = linear_layer._original_proj_down
                linear_layer.proj_up = linear_layer._original_proj_up
                linear_layer.rank = linear_layer._original_rank
        
        # Restore parameters for each SVDQW4A4Linear layer
        for i, module in enumerate(self.net):
            if isinstance(module, SVDQW4A4Linear):
                restore_linear_layer(module, f'net.{i}')
            elif isinstance(module, NunchakuGELU) and hasattr(module, 'proj') and isinstance(module.proj, SVDQW4A4Linear):
                restore_linear_layer(module.proj, f'net.{i}.proj')


class Attention(nn.Module):
    """
    Double-stream attention module for joint image-text attention.

    This module fuses QKV projections for both image and text streams for improved speed,
    applies Q/K normalization and rotary embeddings, and computes joint attention.

    Parameters
    ----------
    query_dim : int
        Input feature dimension.
    dim_head : int, optional
        Dimension per attention head (default: 64).
    heads : int, optional
        Number of attention heads (default: 8).
    dropout : float, optional
        Dropout probability (default: 0.0).
    bias : bool, optional
        Whether to use bias in projections (default: False).
    eps : float, optional
        Epsilon for normalization layers (default: 1e-5).
    out_bias : bool, optional
        Whether to use bias in output projections (default: True).
    out_dim : int, optional
        Output dimension for image stream.
    out_context_dim : int, optional
        Output dimension for text stream.
    dtype : torch.dtype, optional
        Data type for projections.
    device : torch.device, optional
        Device for projections.
    operations : module, optional
        Module providing normalization and linear layers.
    **kwargs
        Additional arguments for quantized linear layers.
    """

    def __init__(
        self,
        query_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
        eps: float = 1e-5,
        out_bias: bool = True,
        out_dim: int = None,
        out_context_dim: int = None,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim
        self.heads = heads
        self.dim_head = dim_head
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.dropout = dropout

        # Q/K normalization for both streams
        self.norm_q = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)

        # Image stream projections: fused QKV for speed
        self.to_qkv = SVDQW4A4Linear(
            query_dim, self.inner_dim + self.inner_kv_dim * 2, bias=bias, torch_dtype=dtype, device=device, **kwargs
        )

        # Text stream projections: fused QKV for speed
        self.add_qkv_proj = SVDQW4A4Linear(
            query_dim, self.inner_dim + self.inner_kv_dim * 2, bias=bias, torch_dtype=dtype, device=device, **kwargs
        )

        # Output projections
        self.to_out = nn.ModuleList(
            [
                SVDQW4A4Linear(self.inner_dim, self.out_dim, bias=out_bias, torch_dtype=dtype, device=device, **kwargs),
                nn.Dropout(dropout),
            ]
        )
        self.to_add_out = SVDQW4A4Linear(
            self.inner_dim, self.out_context_dim, bias=out_bias, torch_dtype=dtype, device=device, **kwargs
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for double-stream attention.

        Parameters
        ----------
        hidden_states : torch.FloatTensor
            Image stream input tensor of shape (batch, seq_len_img, dim).
        encoder_hidden_states : torch.FloatTensor, optional
            Text stream input tensor of shape (batch, seq_len_txt, dim).
        encoder_hidden_states_mask : torch.FloatTensor, optional
            Mask for encoder hidden states.
        attention_mask : torch.FloatTensor, optional
            Attention mask for joint attention.
        image_rotary_emb : torch.Tensor, optional
            Rotary positional embeddings.

        Returns
        -------
        img_attn_output : torch.Tensor
            Output tensor for image stream.
        txt_attn_output : torch.Tensor
            Output tensor for text stream.
        """
        seq_txt = encoder_hidden_states.shape[1]

        img_qkv = self.to_qkv(hidden_states)
        img_query, img_key, img_value = img_qkv.chunk(3, dim=-1)

        # Compute QKV for text stream (context projections)
        txt_qkv = self.add_qkv_proj(encoder_hidden_states)
        txt_query, txt_key, txt_value = txt_qkv.chunk(3, dim=-1)

        img_query = img_query.unflatten(-1, (self.heads, -1))
        img_key = img_key.unflatten(-1, (self.heads, -1))
        img_value = img_value.unflatten(-1, (self.heads, -1))

        txt_query = txt_query.unflatten(-1, (self.heads, -1))
        txt_key = txt_key.unflatten(-1, (self.heads, -1))
        txt_value = txt_value.unflatten(-1, (self.heads, -1))

        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key = self.norm_added_k(txt_key)

        # Concatenate image and text streams for joint attention
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Apply rotary embeddings
        joint_query = apply_rotary_emb(joint_query, image_rotary_emb)
        joint_key = apply_rotary_emb(joint_key, image_rotary_emb)

        joint_query = joint_query.flatten(start_dim=2)
        joint_key = joint_key.flatten(start_dim=2)
        joint_value = joint_value.flatten(start_dim=2)

        # Compute joint attention
        joint_hidden_states = optimized_attention_masked(
            joint_query, joint_key, joint_value, self.heads, attention_mask
        )

        # Split results back to separate streams
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = self.to_out[0](img_attn_output)
        img_attn_output = self.to_out[1](img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output
    
    def update_lora_params(self, lora_dict: dict[str, torch.Tensor]):
        """
        Update LoRA parameters for the attention module.
        
        This applies LoRA by concatenating LoRA projections with existing low-rank projections
        in SVDQW4A4Linear layers.
        """
        from nunchaku.lora.qwenimage.packer import unpack_lowrank_weight
        import logging
        logger = logging.getLogger(__name__)
        
        
        # Helper function to apply LoRA to a SVDQW4A4Linear layer
        def apply_lora_to_linear(linear_layer, lora_dict, layer_prefix):
            lora_down_key = None
            lora_up_key = None
            
            # Find lora_down/lora_up (Nunchaku format) or lora_A/lora_B (Diffusers format)
            for k in lora_dict.keys():
                if layer_prefix in k:
                    if 'lora_down' in k or 'lora_A' in k:
                        lora_down_key = k
                    elif 'lora_up' in k or 'lora_B' in k:
                        lora_up_key = k
            
            if lora_down_key is None or lora_up_key is None:
                return False  # No LoRA for this layer
            
            lora_down_packed = lora_dict[lora_down_key]
            lora_up_packed = lora_dict[lora_up_key]
            
            # The LoRA weights are already packed and merged in the converter
            # Directly replace proj_down and proj_up (following official implementation)
            # The LoRA weights are already merged with original low-rank branches in the converter
            # Just directly apply them
            device = linear_layer.proj_down.device
            dtype = linear_layer.proj_down.dtype
            old_rank = linear_layer.rank
            
            # Directly replace parameters with merged weights
            linear_layer.proj_down.data = lora_down_packed.to(device=device, dtype=dtype)
            linear_layer.proj_up.data = lora_up_packed.to(device=device, dtype=dtype)
            linear_layer.rank = lora_down_packed.shape[1]
            
            return True
        
        # Apply LoRA to each quantized linear layer
        applied = False
        if isinstance(self.to_qkv, SVDQW4A4Linear):
            applied |= apply_lora_to_linear(self.to_qkv, lora_dict, 'to_qkv')
        
        if isinstance(self.add_qkv_proj, SVDQW4A4Linear):
            applied |= apply_lora_to_linear(self.add_qkv_proj, lora_dict, 'add_qkv_proj')
        
        if isinstance(self.to_out[0], SVDQW4A4Linear):
            applied |= apply_lora_to_linear(self.to_out[0], lora_dict, 'to_out.0')
        
        if isinstance(self.to_add_out, SVDQW4A4Linear):
            applied |= apply_lora_to_linear(self.to_add_out, lora_dict, 'to_add_out')
        
        # Summary log disabled - will show overall count instead
        return applied
    
    def restore_original_params(self):
        """
        Restore original parameters for all quantized linear layers in the attention module.
        """
        def restore_linear_layer(linear_layer, layer_prefix):
            if hasattr(linear_layer, '_original_proj_down'):
                linear_layer.proj_down = linear_layer._original_proj_down
                linear_layer.proj_up = linear_layer._original_proj_up
                linear_layer.rank = linear_layer._original_rank
        
        # Restore parameters for each quantized linear layer
        if isinstance(self.to_qkv, SVDQW4A4Linear):
            restore_linear_layer(self.to_qkv, 'to_qkv')
        
        if isinstance(self.add_qkv_proj, SVDQW4A4Linear):
            restore_linear_layer(self.add_qkv_proj, 'add_qkv_proj')
        
        if isinstance(self.to_out[0], SVDQW4A4Linear):
            restore_linear_layer(self.to_out[0], 'to_out.0')
        
        if isinstance(self.to_add_out, SVDQW4A4Linear):
            restore_linear_layer(self.to_add_out, 'to_add_out')


class NunchakuQwenImageTransformerBlock(nn.Module):
    """
    Transformer block with dual-stream (image/text) processing, modulation, and quantized attention/MLP.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    num_attention_heads : int
        Number of attention heads.
    attention_head_dim : int
        Dimension per attention head.
    eps : float, optional
        Epsilon for normalization layers (default: 1e-6).
    dtype : torch.dtype, optional
        Data type for projections.
    device : torch.device, optional
        Device for projections.
    operations : module, optional
        Module providing normalization and linear layers.
    scale_shift : float, optional
        Value added to scale in modulation (default: 1.0). Nunchaku may have fused the scale's shift into bias.
    **kwargs
        Additional arguments for quantized linear layers.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        dtype=None,
        device=None,
        operations=None,
        scale_shift: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.scale_shift = scale_shift
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Modulation and normalization for image stream
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            AWQW4A16Linear(dim, 6 * dim, bias=True, torch_dtype=dtype, device=device),
        )
        self.img_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_mlp = NunchakuFeedForward(dim=dim, dim_out=dim, dtype=dtype, device=device, **kwargs)

        # Modulation and normalization for text stream
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            AWQW4A16Linear(dim, 6 * dim, bias=True, torch_dtype=dtype, device=device),
        )
        self.txt_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_mlp = NunchakuFeedForward(dim=dim, dim_out=dim, dtype=dtype, device=device, **kwargs)

        self.attn = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=eps,
            dtype=dtype,
            device=device,
            operations=operations,
            **kwargs,
        )

    def _modulate(self, x: torch.Tensor, mod_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply modulation to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, dim).
        mod_params : torch.Tensor
            Modulation parameters of shape (batch, 3*dim).

        Returns
        -------
        modulated_x : torch.Tensor
            Modulated tensor.
        gate : torch.Tensor
            Gate tensor for residual connection.
        """
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if self.scale_shift != 0:
            scale.add_(self.scale_shift)
        return x * scale.unsqueeze(1) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the transformer block.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Image stream input tensor.
        encoder_hidden_states : torch.Tensor
            Text stream input tensor.
        encoder_hidden_states_mask : torch.Tensor
            Mask for encoder hidden states.
        temb : torch.Tensor
            Timestep or conditioning embedding.
        image_rotary_emb : tuple of torch.Tensor, optional
            Rotary positional embeddings.

        Returns
        -------
        encoder_hidden_states : torch.Tensor
            Updated text stream tensor.
        hidden_states : torch.Tensor
            Updated image stream tensor.
        """
        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]
        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # Nunchaku's mod_params is [B, 6*dim] instead of [B, dim*6]
        img_mod_params = (
            img_mod_params.view(img_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(img_mod_params.shape[0], -1)
        )
        txt_mod_params = (
            txt_mod_params.view(txt_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(txt_mod_params.shape[0], -1)
        )

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Joint attention computation (DoubleStreamLayerMegatron logic)
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream ("sample")
            encoder_hidden_states=txt_modulated,  # Text stream ("context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        return encoder_hidden_states, hidden_states

    def update_lora_params(self, lora_dict: dict):
        """
        Update LoRA parameters for the transformer block.
        
        Directly applies LoRA to attention and MLP layers by calling their update methods.
        
        Parameters
        ----------
        lora_dict : dict
            Dictionary containing LoRA weights for this block in Nunchaku format (lora_down/lora_up).
        """
        # Apply LoRA to attention
        if hasattr(self.attn, 'update_lora_params'):
            attn_lora = {k: v for k, v in lora_dict.items() if 'attn' in k}
            if attn_lora:
                self.attn.update_lora_params(attn_lora)
        
        # Apply LoRA to image stream MLP
        if hasattr(self.img_mlp, 'update_lora_params'):
            img_mlp_lora = {k: v for k, v in lora_dict.items() if 'img_mlp' in k}
            if img_mlp_lora:
                self.img_mlp.update_lora_params(img_mlp_lora)
        
        # Apply LoRA to text stream MLP
        if hasattr(self.txt_mlp, 'update_lora_params'):
            txt_mlp_lora = {k: v for k, v in lora_dict.items() if 'txt_mlp' in k}
            if txt_mlp_lora:
                self.txt_mlp.update_lora_params(txt_mlp_lora)
    
    def restore_original_params(self):
        """
        Restore original parameters for all components in this transformer block.
        """
        # Restore attention parameters
        if hasattr(self.attn, 'restore_original_params'):
            self.attn.restore_original_params()
        
        # Restore image MLP parameters
        if hasattr(self.img_mlp, 'restore_original_params'):
            self.img_mlp.restore_original_params()
        
        # Restore text MLP parameters
        if hasattr(self.txt_mlp, 'restore_original_params'):
            self.txt_mlp.restore_original_params()


class NunchakuQwenImageTransformer2DModel(NunchakuModelMixin, QwenImageTransformer2DModel):
    """
    Full transformer model for QwenImage, using Nunchaku-optimized blocks.

    Parameters
    ----------
    patch_size : int, optional
        Patch size for image input (default: 2).
    in_channels : int, optional
        Number of input channels (default: 64).
    out_channels : int, optional
        Number of output channels (default: 16).
    num_layers : int, optional
        Number of transformer layers (default: 60).
    attention_head_dim : int, optional
        Dimension per attention head (default: 128).
    num_attention_heads : int, optional
        Number of attention heads (default: 24).
    joint_attention_dim : int, optional
        Dimension for joint attention (default: 3584).
    pooled_projection_dim : int, optional
        Dimension for pooled projection (default: 768).
    guidance_embeds : bool, optional
        Whether to use guidance embeddings (default: False).
    axes_dims_rope : tuple of int, optional
        Axes dimensions for rotary embeddings (default: (16, 56, 56)).
    image_model : module, optional
        Optional image model.
    dtype : torch.dtype, optional
        Data type for projections.
    device : torch.device, optional
        Device for projections.
    operations : module, optional
        Module providing normalization and linear layers.
    scale_shift : float, optional
        Value added to scale in modulation (default: 1.0).
    **kwargs
        Additional arguments for quantized linear layers.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        image_model=None,
        dtype=None,
        device=None,
        operations=None,
        scale_shift: float = 1.0,
        **kwargs,
    ):
        super(QwenImageTransformer2DModel, self).__init__()
        
        # LoRA support attributes (similar to nunchaku library implementation)
        self._unquantized_part_sd: dict[str, torch.Tensor] = {}
        self._unquantized_part_loras: dict[str, torch.Tensor] = {}
        self._quantized_part_sd: dict[str, torch.Tensor] = {}
        self._quantized_part_vectors: dict[str, torch.Tensor] = {}
        
        # ComfyUI LoRA related attributes
        # Note: comfy_lora_meta_list and comfy_lora_sd_list are now initialized dynamically in _forward
        # to support Flux-style caching. _lora_config_list is set by LoRA Loader nodes.
        
        self.dtype = dtype
        self.patch_size = patch_size
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=10000, axes_dim=list(axes_dims_rope))

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.txt_norm = operations.RMSNorm(joint_attention_dim, eps=1e-6, dtype=dtype, device=device)
        self.img_in = operations.Linear(in_channels, self.inner_dim, dtype=dtype, device=device)
        self.txt_in = operations.Linear(joint_attention_dim, self.inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList(
            [
                NunchakuQwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                    scale_shift=scale_shift,
                    **kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = LastLayer(
            self.inner_dim,
            self.inner_dim,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.proj_out = operations.Linear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.gradient_checkpointing = False

    def process_img(self, x, index=0, h_offset=0, w_offset=0):
        """
        Preprocess an input image tensor for the model.
        
        Overrides the base class method to handle 4D tensors (batch, channels, height, width)
        instead of 5D tensors required by ComfyUI's base implementation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, channels, height, width).
        index : int, optional
            Index for image ID encoding.
        h_offset : int, optional
            Height offset for patch IDs.
        w_offset : int, optional
            Width offset for patch IDs.
        
        Returns
        -------
        img : torch.Tensor
            Rearranged image tensor of shape (batch, num_patches, patch_dim).
        img_ids : torch.Tensor
            Image ID tensor of shape (batch, num_patches, 3).
        orig_shape : tuple
            Original shape (batch, channels, height, width) for unpatchify.
        """
        from comfy.ldm.common_dit import pad_to_patch_size
        from einops import rearrange, repeat
        
        bs, c, h, w = x.shape
        x = pad_to_patch_size(x, (self.patch_size, self.patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=self.patch_size, pw=self.patch_size)
        h_len = (h + (self.patch_size // 2)) // self.patch_size
        w_len = (w + (self.patch_size // 2)) // self.patch_size

        h_offset = (h_offset + (self.patch_size // 2)) // self.patch_size
        w_offset = (w_offset + (self.patch_size // 2)) // self.patch_size

        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 0] = img_ids[:, :, 1] + index
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(
            h_offset, h_len - 1 + h_offset, steps=h_len, device=x.device, dtype=x.dtype
        ).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(
            w_offset, w_len - 1 + w_offset, steps=w_len, device=x.device, dtype=x.dtype
        ).unsqueeze(0)
        
        # Return orig_shape as (batch, channels, height, width) for unpatchify
        return img, repeat(img_ids, "h w c -> b (h w) c", b=bs), (bs, c, h, w)

    def forward(
        self,
        hidden_states=None,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        timestep=None,
        x=None,
        context=None,
        attention_mask=None,
        **kwargs
    ):
        """
        Forward pass adapter for ComfyUI compatibility.
        
        This method handles parameter name conversion between ComfyUI's convention
        (hidden_states, encoder_hidden_states) and the internal implementation
        (x, context).
        
        Parameters can be provided in either naming convention:
        - ComfyUI style: hidden_states, encoder_hidden_states, encoder_hidden_states_mask, timestep
        - Internal style: x, context, attention_mask, timesteps
        
        This method delegates to _forward() with the correct parameter names.
        """
        # Convert parameter names from ComfyUI to internal format
        if x is None and hidden_states is not None:
            x = hidden_states
        if context is None and encoder_hidden_states is not None:
            context = encoder_hidden_states
        if attention_mask is None and encoder_hidden_states_mask is not None:
            attention_mask = encoder_hidden_states_mask
        if 'timesteps' not in kwargs and timestep is not None:
            kwargs['timesteps'] = timestep
        
        # Call internal _forward with correct parameter names
        return self._forward(
            x=x,
            context=context,
            attention_mask=attention_mask,
            **kwargs
        )

    def _forward(
        self,
        x,
        timesteps,
        context,
        attention_mask=None,
        guidance: torch.Tensor = None,
        ref_latents=None,
        transformer_options={},
        control=None,
        **kwargs,
    ):
        """
        Forward pass of the Nunchaku Qwen-Image model.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, channels, height, width).
        timesteps : torch.Tensor or int
            Timestep(s) for diffusion process.
        context : torch.Tensor
            Textual context tensor (e.g., from a text encoder).
        attention_mask : torch.Tensor, optional
            Optional attention mask for the context.
        guidance : torch.Tensor, optional
            Optional guidance tensor for classifier-free guidance.
        ref_latents : list[torch.Tensor], optional
            Optional list of reference latent tensors for multi-image conditioning.
        transformer_options : dict, optional
            Dictionary of options for transformer block patching and replacement.
        **kwargs
            Additional keyword arguments. Supports 'ref_latents_method' to control reference latent handling.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, channels, height, width), matching the input spatial dimensions.

        """
        device = x.device
        if self.offload:
            self.offload_manager.set_device(device)

        # LoRA composition logic with caching (Flux-style)
        # Note: self is always the transformer (not the container)
        # The container injects _lora_config_list into the transformer before calling forward
        
        if hasattr(self, '_lora_config_list'):
            # If config is empty, clear all LoRA parameters
            if len(self._lora_config_list) == 0:
                if hasattr(self, 'comfy_lora_meta_list') and len(self.comfy_lora_meta_list) > 0:
                    self.reset_lora()
                    self.comfy_lora_meta_list = []
                    self.comfy_lora_sd_list = []
            # If config is not empty, execute sync logic
            elif len(self._lora_config_list) > 0:
                from nunchaku.lora.qwenimage import compose_lora
                from nunchaku.utils import load_state_dict_in_safetensors
                
                # Initialize cache lists if not present (on transformer, shared)
                if not hasattr(self, 'comfy_lora_meta_list'):
                    self.comfy_lora_meta_list = []
                if not hasattr(self, 'comfy_lora_sd_list'):
                    self.comfy_lora_sd_list = []
                
                # Smart sync: compare config with applied state
                if self._lora_config_list != self.comfy_lora_meta_list:
                    # Remove excess cache entries if config list shortened
                    for _ in range(max(0, len(self.comfy_lora_meta_list) - len(self._lora_config_list))):
                        self.comfy_lora_meta_list.pop()
                        self.comfy_lora_sd_list.pop()
                    
                    # Sync each LoRA
                    lora_to_be_composed = []
                    for i in range(len(self._lora_config_list)):
                        meta = self._lora_config_list[i]  # (path, strength)
                        
                        # New LoRA: load and cache
                        if i >= len(self.comfy_lora_meta_list):
                            sd = load_state_dict_in_safetensors(meta[0])
                            self.comfy_lora_meta_list.append(meta)
                            self.comfy_lora_sd_list.append(sd)
                        # LoRA config changed
                        elif self.comfy_lora_meta_list[i] != meta:
                            # Path changed: reload file
                            if meta[0] != self.comfy_lora_meta_list[i][0]:
                                sd = load_state_dict_in_safetensors(meta[0])
                                self.comfy_lora_sd_list[i] = sd
                            # Only strength changed: reuse cache
                            self.comfy_lora_meta_list[i] = meta
                        
                        # Add to composition list (always recompose with current strength)
                        lora_to_be_composed.append(({k: v for k, v in self.comfy_lora_sd_list[i].items()}, meta[1]))
                    
                    # Compose all LoRAs
                    composed_lora = compose_lora(lora_to_be_composed)
                    
                    # Apply to model
                    if len(composed_lora) == 0:
                        self.reset_lora()
                    else:
                        self.update_lora_params(composed_lora)
                        
                        # Activate LoRA
                        from nunchaku.models.linear import SVDQW4A4Linear
                        for block in self.transformer_blocks:
                            for module in block.modules():
                                if isinstance(module, SVDQW4A4Linear):
                                    module.lora_strength = 1.0

        timestep = timesteps
        encoder_hidden_states = context
        encoder_hidden_states_mask = attention_mask

        hidden_states, img_ids, orig_shape = self.process_img(x)
        num_embeds = hidden_states.shape[1]

        if ref_latents is not None:
            h = 0
            w = 0
            index = 0
            index_ref_method = kwargs.get("ref_latents_method", "index") == "index"
            for ref in ref_latents:
                if index_ref_method:
                    index += 1
                    h_offset = 0
                    w_offset = 0
                else:
                    index = 1
                    h_offset = 0
                    w_offset = 0
                    if ref.shape[-2] + h > ref.shape[-1] + w:
                        w_offset = w
                    else:
                        h_offset = h
                    h = max(h, ref.shape[-2] + h_offset)
                    w = max(w, ref.shape[-1] + w_offset)

                kontext, kontext_ids, _ = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
                hidden_states = torch.cat([hidden_states, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)

        txt_start = round(
            max(
                ((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2,
                ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2,
            )
        )
        txt_ids = (
            torch.arange(txt_start, txt_start + context.shape[1], device=x.device)
            .reshape(1, -1, 1)
            .repeat(x.shape[0], 1, 3)
        )
        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pe_embedder(ids).squeeze(1).unsqueeze(2).to(x.dtype)
        del ids, txt_ids, img_ids

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})

        # Setup compute stream for offloading
        compute_stream = torch.cuda.current_stream()
        if self.offload:
            self.offload_manager.initialize(compute_stream)

        for i, block in enumerate(self.transformer_blocks):
            with torch.cuda.stream(compute_stream):
                if self.offload:
                    block = self.offload_manager.get_block(i)
                if ("double_block", i) in blocks_replace:

                    def block_wrap(args):
                        out = {}
                        out["txt"], out["img"] = block(
                            hidden_states=args["img"],
                            encoder_hidden_states=args["txt"],
                            encoder_hidden_states_mask=encoder_hidden_states_mask,
                            temb=args["vec"],
                            image_rotary_emb=args["pe"],
                        )
                        return out

                    out = blocks_replace[("double_block", i)](
                        {"img": hidden_states, "txt": encoder_hidden_states, "vec": temb, "pe": image_rotary_emb},
                        {"original_block": block_wrap},
                    )
                    hidden_states = out["img"]
                    encoder_hidden_states = out["txt"]
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                    )
                # ControlNet helpers(device/dtype-safe residual adds)
                _control = (
                    control
                    if control is not None
                    else (transformer_options.get("control", None) if isinstance(transformer_options, dict) else None)
                )
                if isinstance(_control, dict):
                    control_i = _control.get("input")
                    try:
                        _scale = float(_control.get("weight", _control.get("scale", 1.0)))
                    except Exception:
                        _scale = 1.0
                else:
                    control_i = None
                    _scale = 1.0
                if control_i is not None and i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        if (
                            getattr(add, "device", None) != hidden_states.device
                            or getattr(add, "dtype", None) != hidden_states.dtype
                        ):
                            add = add.to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
                        t = min(hidden_states.shape[1], add.shape[1])
                        if t > 0:
                            hidden_states[:, :t].add_(add[:, :t], alpha=_scale)

            if self.offload:
                self.offload_manager.step(compute_stream)

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states[:, :num_embeds].view(
            orig_shape[0], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2
        )
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5)
        return hidden_states.reshape(orig_shape)[:, :, : x.shape[-2], : x.shape[-1]]

    def update_lora_params(self, lora_dict: dict, num_loras: int = 1):
        """
        Update LoRA parameters for the Qwen Image model.
        
        This method applies LoRA weights to the model.
        For ComfyUI-nunchaku, we use a simplified approach that directly applies
        LoRA weights without the complex quantization handling.
        
        Parameters
        ----------
        lora_dict : dict
            Dictionary containing LoRA weights in Diffusers or Nunchaku format.
        num_loras : int, optional
            Number of LoRAs that were composed. If > 1, this is a composed LoRA.
            Used to determine whether to merge with base model.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Import necessary functions
        from nunchaku.lora.qwenimage import is_nunchaku_format, to_nunchaku
        
        # Convert to nunchaku format if needed
        if not is_nunchaku_format(lora_dict):
            logger.debug("Converting LoRA to Nunchaku format")
            
            # Check if this is a composed LoRA
            is_composed = (num_loras > 1)
            
            # Always use skip_base_merge=False (Qwen Image requires base model low-rank branches)
            if is_composed:
                logger.debug(f"Detected composed LoRA ({num_loras} LoRAs)")
            else:
                logger.debug(f"Single LoRA detected")
            
            lora_dict = to_nunchaku(lora_dict, base_sd=self._quantized_part_sd, skip_base_merge=False)
            logger.debug(f"Converted LoRA to Nunchaku format: {len(lora_dict)} keys")
        else:
            logger.debug("LoRA already in Nunchaku format")
        
        # Apply LoRA to transformer blocks
        blocks_updated = 0
        for i, block in enumerate(self.transformer_blocks):
            # Extract LoRA weights for this block
            block_lora = {}
            for k, v in lora_dict.items():
                if f"transformer_blocks.{i}." in k or f"blocks.{i}." in k:
                    # Remove all prefixes to get relative key
                    parts = k.split(f".{i}.")
                    if len(parts) > 1:
                        relative_key = parts[-1]
                        block_lora[relative_key] = v
            
            # Apply LoRA to this block if it has any weights
            if block_lora:
                # Disabled detailed logging - only show final summary
                # if i == 0:  # Only log first block to reduce noise
                #     logger.info(f"  Block {i}: {len(block_lora)} LoRA keys")
                if hasattr(block, 'update_lora_params'):
                    block.update_lora_params(block_lora)
                    blocks_updated += 1
        
        logger.info(f"LoRA applied to {blocks_updated}/{len(self.transformer_blocks)} blocks")
    
    def restore_original_params(self):
        """
        Restore original parameters for all transformer blocks.
        This method should be called when LoRA is no longer needed.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("🔄 Restoring original model parameters...")
        blocks_restored = 0
        for block in self.transformer_blocks:
            if hasattr(block, 'restore_original_params'):
                block.restore_original_params()
                blocks_restored += 1
        
        logger.info(f"Restored original parameters for {blocks_restored}/{len(self.transformer_blocks)} blocks")
    
    def reset_lora(self):
        """
        Reset LoRA parameters to remove all LoRA effects.
        """
        # Import the nunchaku library's transformer model
        from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel as NunchakuQwenImageTransformer2DModelLib
        
        # Check if the nunchaku library's model has the reset_lora method
        if hasattr(NunchakuQwenImageTransformer2DModelLib, 'reset_lora'):
            NunchakuQwenImageTransformer2DModelLib.reset_lora(self)
        else:
            # Fallback: clear LoRA lists
            self.comfy_lora_meta_list = []
            self.comfy_lora_sd_list = []

    def set_offload(self, offload: bool, **kwargs):
        """
        Enable or disable CPU offloading for the transformer blocks.

        Parameters
        ----------
        offload : bool
            If True, enable CPU offloading. If False, disable it.
        **kwargs
            Additional keyword arguments:
                - use_pin_memory (bool): Whether to use pinned memory (default: True).
                - num_blocks_on_gpu (int): Number of transformer blocks to keep on GPU (default: 1).

        Notes
        -----
        - When offloading is enabled, only a subset of modules remain on GPU.
        - When disabling, memory is released and CUDA cache is cleared.
        """
        if offload == self.offload:
            # Nothing changed, just return
            return
        self.offload = offload
        if offload:
            self.offload_manager = CPUOffloadManager(
                self.transformer_blocks,
                use_pin_memory=kwargs.get("use_pin_memory", True),
                on_gpu_modules=[
                    self.img_in,
                    self.txt_in,
                    self.txt_norm,
                    self.time_text_embed,
                    self.norm_out,
                    self.proj_out,
                ],
                num_blocks_on_gpu=kwargs.get("num_blocks_on_gpu", 1),
            )
        else:
            self.offload_manager = None
            gc.collect()
            torch.cuda.empty_cache()

    def set_lora_strength(self, strength: float):
        """
        Sets the LoRA scaling strength for the model.
        
        This method allows dynamic adjustment of LoRA strength, similar to Flux's setLoraScale.
        The strength is applied only to the LoRA part (ranks beyond original_rank), while
        the original low-rank branches remain at strength 1.0.

        Parameters
        ----------
        strength : float, optional
            LoRA scaling strength (default: 1).

        Note: This function will change the strength of all the LoRAs. So only use it when you only have a single LoRA.
        """
        # Set LoRA strength for all SVDQW4A4Linear layers in transformer blocks
        from nunchaku.models.linear import SVDQW4A4Linear
        
        for block in self.transformer_blocks:
            # Set strength for all SVDQW4A4Linear layers in this block
            for module in block.modules():
                if isinstance(module, SVDQW4A4Linear):
                    module.set_lora_strength(strength)
        
        # Handle unquantized part (similar to Flux implementation)
        if len(self._unquantized_part_loras) > 0:
            self._update_unquantized_part_lora_params(strength)
        if len(self._quantized_part_vectors) > 0:
            from nunchaku.lora.qwenimage.utils import fuse_vectors
            vector_dict = fuse_vectors(self._quantized_part_vectors, self._quantized_part_sd, strength)
            for block in self.transformer_blocks:
                if hasattr(block, 'update_lora_params'):
                    block.update_lora_params(vector_dict)