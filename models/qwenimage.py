# adapted from https://github.com/comfyanonymous/ComfyUI/blob/v0.3.51/comfy/ldm/qwen_image/model.py
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
from torch import DeviceObjType, nn

from nunchaku.models.linear import AWQW4A16Linear, SVDQW4A4Linear
from nunchaku.ops.fused import fused_gelu_mlp


class NunchakuGELU(GELU):
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
        self.net.append(SVDQW4A4Linear(inner_dim, dim_out, bias=bias, torch_dtype=dtype, device=device, **kwargs))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(self.net[0], NunchakuGELU):
            return fused_gelu_mlp(hidden_states, self.net[0].proj, self.net[2])
        else:
            # fallback to original implementation
            for module in self.net:
                hidden_states = module(hidden_states)
            return hidden_states


class Attention(nn.Module):
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

        # Q/K normalization
        self.norm_q = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)

        # Image stream projections, nunchaku fuse the qkv projection for better speed (faster execution speed)
        self.to_qkv = SVDQW4A4Linear(
            query_dim, self.inner_dim + self.inner_kv_dim * 2, bias=bias, torch_dtype=dtype, device=device, **kwargs
        )

        # Text stream projections, nunchaku fuse the qkv projection for better speed (faster execution speed)
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
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        joint_query = apply_rotary_emb(joint_query, image_rotary_emb)
        joint_key = apply_rotary_emb(joint_key, image_rotary_emb)

        joint_query = joint_query.flatten(start_dim=2)
        joint_key = joint_key.flatten(start_dim=2)
        joint_value = joint_value.flatten(start_dim=2)

        joint_hidden_states = optimized_attention_masked(
            joint_query, joint_key, joint_value, self.heads, attention_mask
        )

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = self.to_out[0](img_attn_output)
        img_attn_output = self.to_out[1](img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class NunchakuQwenImageTransformerBlock(nn.Module):
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

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            AWQW4A16Linear(dim, 6 * dim, bias=True, torch_dtype=dtype, device=device),
        )
        self.img_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_mlp = NunchakuFeedForward(dim=dim, dim_out=dim, dtype=dtype, device=device, **kwargs)

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
        """Apply modulation to input tensor"""
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
        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]
        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # nunchaku's mod_params is [B, 6*dim] instead of [B, dim*6]
        img_mod_params = (
            img_mod_params.view(img_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(img_mod_params.shape[0], -1)
        )
        txt_mod_params = (
            txt_mod_params.view(txt_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(txt_mod_params.shape[0], -1)
        )

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Split modulation parameters for norm1 and norm2
        # img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        # txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
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


class NunchakuQwenImageTransformer2DModel(QwenImageTransformer2DModel):
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
