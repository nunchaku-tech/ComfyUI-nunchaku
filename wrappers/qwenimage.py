"""
This module provides a wrapper for the :class:`~nunchaku.models.transformers.transformer_qwenimage.NunchakuQwenImageTransformer2DModel`,
enabling integration with ComfyUI forward, LoRA composition, and advanced caching strategies.
"""

from typing import Callable

import torch
from comfy.ldm.common_dit import pad_to_patch_size
from einops import rearrange, repeat
from torch import nn

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.caching.fbcache import cache_context, create_cache_context
from nunchaku.lora.qwenimage.compose import compose_lora
from nunchaku.utils import load_state_dict_in_safetensors


class ComfyQwenImageWrapper(nn.Module):
    """
    Wrapper for :class:`~nunchaku.models.transformers.transformer_qwenimage.NunchakuQwenImageTransformer2DModel`
    to support ComfyUI workflows, LoRA composition, and caching.

    Parameters
    ----------
    model : :class:`~nunchaku.models.transformers.transformer_qwenimage.NunchakuQwenImageTransformer2DModel`
        The underlying Nunchaku model to wrap.
    config : dict
        Model configuration dictionary.
    customized_forward : Callable, optional
        Optional custom forward function.
    forward_kwargs : dict, optional
        Additional keyword arguments for the forward pass.

    Attributes
    ----------
    model : :class:`~nunchaku.models.transformers.transformer_qwenimage.NunchakuQwenImageTransformer2DModel`
        The wrapped model.
    dtype : torch.dtype
        Data type of the model parameters.
    config : dict
        Model configuration.
    loras : list
        List of LoRA metadata for composition.
    customized_forward : Callable or None
        Custom forward function if provided.
    forward_kwargs : dict
        Additional arguments for the forward pass.
    """

    def __init__(
        self,
        model: NunchakuQwenImageTransformer2DModel,
        config,
        customized_forward: Callable = None,
        forward_kwargs: dict | None = {},
    ):
        super(ComfyQwenImageWrapper, self).__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config
        self.loras = []

        self.customized_forward = customized_forward
        self.forward_kwargs = {} if forward_kwargs is None else forward_kwargs

        self._prev_timestep = None  # for first-block cache
        self._cache_context = None

    def to_safely(self, device):
        """
        Safely move the model to the specified device.
        Required by NunchakuModelPatcher for device management.
        """
        if hasattr(self.model, "to_safely"):
            return self.model.to_safely(device)
        else:
            return self.model.to(device)

    def process_img(self, x, index=0, h_offset=0, w_offset=0):
        """
        Preprocess an input image tensor for the model.

        Pads and rearranges the image into patches and generates corresponding image IDs.

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
        """
        bs, c, h, w = x.shape
        patch_size = self.config.get("patch_size", 2)
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        h_len = (h + (patch_size // 2)) // patch_size
        w_len = (w + (patch_size // 2)) // patch_size

        h_offset = (h_offset + (patch_size // 2)) // patch_size
        w_offset = (w_offset + (patch_size // 2)) // patch_size

        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 0] = img_ids[:, :, 1] + index
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(
            h_offset, h_len - 1 + h_offset, steps=h_len, device=x.device, dtype=x.dtype
        ).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(
            w_offset, w_len - 1 + w_offset, steps=w_len, device=x.device, dtype=x.dtype
        ).unsqueeze(0)
        return img, repeat(img_ids, "h w c -> b (h w) c", b=bs)

    def forward(
        self,
        x,
        timestep,
        context=None,
        y=None,
        guidance=None,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        """
        Forward pass for the wrapped model.

        Handles LoRA composition, caching, and dual-stream processing for Qwen Image.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor.
        timestep : float or torch.Tensor
            Diffusion timestep.
        context : torch.Tensor
            Context tensor (e.g., text embeddings).
        y : torch.Tensor
            Pooled projections or additional conditioning.
        guidance : torch.Tensor
            Guidance embedding or value.
        control : dict, optional
            ControlNet input and output samples.
        transformer_options : dict, optional
            Additional transformer options.
        **kwargs
            Additional keyword arguments, e.g., 'ref_latents'.

        Returns
        -------
        out : torch.Tensor
            Output tensor of the same spatial size as the input.
        """
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 1:
                timestep_float = timestep.item()
            else:
                timestep_float = timestep.flatten()[0].item()
        else:
            assert isinstance(timestep, float)
            timestep_float = timestep

        model = self.model
        if model is None:
            raise ValueError("Wrapped model is None!")
        if not (
            type(model).__name__ == "NunchakuQwenImageTransformer2DModel"
            or isinstance(model, NunchakuQwenImageTransformer2DModel)
        ):
            raise TypeError(f"Expected NunchakuQwenImageTransformer2DModel, got {type(model).__name__}")

        # Check if x is already processed or needs processing
        input_is_5d = False
        if x.ndim == 5:
            # x is (batch, channels, 1, height, width) - squeeze the middle dimension
            input_is_5d = True
            x = x.squeeze(2)  # Now (batch, channels, height, width)

        # Keep x in 4D format and let model's _forward handle process_img
        if x.ndim != 4:
            raise ValueError(f"Unexpected input shape: {x.shape}, expected 4D tensor")

        # load and compose LoRA
        if self.loras != model.comfy_lora_meta_list:
            from nunchaku.lora.qwenimage import is_nunchaku_format

            lora_to_be_composed = []
            nunchaku_lora_count = 0

            for _ in range(max(0, len(model.comfy_lora_meta_list) - len(self.loras))):
                model.comfy_lora_meta_list.pop()
                model.comfy_lora_sd_list.pop()

            for i in range(len(self.loras)):
                meta = self.loras[i]
                if i >= len(model.comfy_lora_meta_list):
                    sd = load_state_dict_in_safetensors(meta[0])
                    model.comfy_lora_meta_list.append(meta)
                    model.comfy_lora_sd_list.append(sd)
                elif model.comfy_lora_meta_list[i] != meta:
                    if meta[0] != model.comfy_lora_meta_list[i][0]:
                        sd = load_state_dict_in_safetensors(meta[0])
                        model.comfy_lora_sd_list[i] = sd
                    model.comfy_lora_meta_list[i] = meta

                # Check if this LoRA is already in Nunchaku format
                sd_to_compose = model.comfy_lora_sd_list[i]
                if is_nunchaku_format(sd_to_compose):
                    nunchaku_lora_count += 1
                    # Convert back to Diffusers format for composition
                    from nunchaku.lora.qwenimage import to_diffusers

                    sd_to_compose = to_diffusers(sd_to_compose)
                    # Update the cache with Diffusers version
                    model.comfy_lora_sd_list[i] = sd_to_compose

                lora_to_be_composed.append(({k: v for k, v in sd_to_compose.items()}, meta[1]))

            # Now all LoRAs are in Diffusers format, can safely compose
            composed_lora = compose_lora(lora_to_be_composed)

            if len(composed_lora) == 0:
                # CRITICAL: Manually restore original proj_down/proj_up weights
                import torch.nn as nn

                from nunchaku.models.linear import SVDQW4A4Linear

                restored_count = 0
                for name, module in model.named_modules():
                    if isinstance(module, SVDQW4A4Linear):
                        proj_down_key = f"{name}.proj_down"
                        proj_up_key = f"{name}.proj_up"
                        if proj_down_key in model._quantized_part_sd:
                            original_proj_down = model._quantized_part_sd[proj_down_key]
                            module.proj_down = nn.Parameter(
                                original_proj_down.clone().to(
                                    device=module.proj_down.device, dtype=module.proj_down.dtype
                                ),
                                requires_grad=False,
                            )
                            restored_count += 1
                        if proj_up_key in model._quantized_part_sd:
                            original_proj_up = model._quantized_part_sd[proj_up_key]
                            module.proj_up = nn.Parameter(
                                original_proj_up.clone().to(device=module.proj_up.device, dtype=module.proj_up.dtype),
                                requires_grad=False,
                            )
                            restored_count += 1
                        if proj_down_key in model._quantized_part_sd:
                            original_rank = model._quantized_part_sd[proj_down_key].shape[1]
                            module.rank = original_rank
                            if not hasattr(module, "original_rank"):
                                module.original_rank = original_rank
                        module.lora_strength = 0.0

                model.reset_lora()
            else:
                # Pass number of LoRAs to help detect composed LoRAs
                model.update_lora_params(composed_lora, num_loras=len(self.loras))

                # CRITICAL: For composed LoRAs, strength is already baked by compose_lora
                # Setting lora_strength to a uniform value will destroy the individual strength differences
                # SOLUTION: Calculate average strength or use 1.0 as neutral value
                from nunchaku.models.linear import SVDQW4A4Linear

                # Calculate weighted average strength for composed LoRAs
                if len(self.loras) > 1:
                    avg_strength = sum(s for _, s in self.loras) / len(self.loras)
                else:
                    avg_strength = 1.0

                for block in model.transformer_blocks:
                    for module in block.modules():
                        if isinstance(module, SVDQW4A4Linear):
                            module.lora_strength = avg_strength

        # Note: nunchaku's attention processor doesn't accept 'wrappers' and other ComfyUI-specific keys
        # We handle this by not passing transformer_options to the underlying model

        if getattr(model, "residual_diff_threshold_multi", 0) != 0 or getattr(model, "_is_cached", False):
            # A more robust caching strategy
            cache_invalid = False

            # Check if timestamps have changed or are out of valid range
            if self._prev_timestep is None:
                cache_invalid = True
            elif self._prev_timestep < timestep_float + 1e-5:  # allow a small tolerance to reuse the cache
                cache_invalid = True

            if cache_invalid:
                self._cache_context = create_cache_context()

            # Update the previous timestamp
            self._prev_timestep = timestep_float
            with cache_context(self._cache_context):
                if self.customized_forward is None:
                    out = model(
                        hidden_states=x,
                        encoder_hidden_states=context,
                        encoder_hidden_states_mask=None,  # Qwen Image doesn't use mask in ComfyUI
                        timestep=timestep,
                        ref_latents=kwargs.get("ref_latents"),
                        guidance=guidance if self.config.get("guidance_embed", False) else None,
                        control=control,
                        transformer_options=transformer_options,
                    )
                else:
                    out = self.customized_forward(
                        model,
                        hidden_states=x,
                        encoder_hidden_states=context,
                        encoder_hidden_states_mask=None,
                        timestep=timestep,
                        ref_latents=kwargs.get("ref_latents"),
                        guidance=guidance if self.config.get("guidance_embed", False) else None,
                        control=control,
                        transformer_options=transformer_options,
                        **self.forward_kwargs,
                    )
        else:
            if self.customized_forward is None:
                # Pass original 4D x to model, let model handle process_img
                # Model's forward will convert parameters and call _forward
                out = model(
                    hidden_states=x,  # Pass original 4D x
                    encoder_hidden_states=context,
                    encoder_hidden_states_mask=None,
                    timestep=timestep,
                    ref_latents=kwargs.get("ref_latents"),
                    guidance=guidance if self.config.get("guidance_embed", False) else None,
                    control=control,
                    transformer_options=transformer_options,
                )
            else:
                out = self.customized_forward(
                    model,
                    hidden_states=x,  # Pass original 4D x
                    encoder_hidden_states=context,
                    encoder_hidden_states_mask=None,
                    timestep=timestep,
                    ref_latents=kwargs.get("ref_latents"),
                    guidance=guidance if self.config.get("guidance_embed", False) else None,
                    control=control,
                    transformer_options=transformer_options,
                    **self.forward_kwargs,
                )

        # Model returns a tuple (output,), unpack it
        if isinstance(out, tuple):
            out = out[0]

        # Model already returns unpatchified output (4D)
        # out = out[:, :img_tokens]
        # out = rearrange(
        #     out,
        #     "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        #     h=h_len,
        #     w=w_len,
        #     ph=patch_size,
        #     pw=patch_size,
        # )

        # If input was 5D, unsqueeze output back to 5D
        if input_is_5d:
            out = out.unsqueeze(2)  # (batch, channels, height, width) -> (batch, channels, 1, height, width)
        return out
