from typing import Callable

import torch
from comfy.ldm.common_dit import pad_to_patch_size
from diffusers import FluxPipeline
from einops import rearrange
from torch import nn

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.caching.utils import cache_context, create_cache_context
from nunchaku.lora.flux.compose import compose_lora
from nunchaku.pipeline.pipeline_flux_pulid import PuLIDPipeline
from nunchaku.utils import load_state_dict_in_safetensors


class ComfyFluxWrapper(nn.Module):
    def __init__(
        self,
        model: NunchakuFluxTransformer2dModel,
        config,
        pulid_pipeline: PuLIDPipeline | None = None,
        customized_forward: Callable = None,
        forward_kwargs: dict | None = {},
    ):
        super(ComfyFluxWrapper, self).__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config
        self.loras = []

        self.pulid_pipeline = pulid_pipeline
        self.customized_forward = customized_forward
        self.forward_kwargs = {} if forward_kwargs is None else forward_kwargs

        self._prev_timestep = None  # for first-block cache
        self._cache_context = None

    def forward(
        self,
        x,
        timestep,
        context,
        y,
        guidance,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 1:
                timestep_float = timestep.item()
            else:
                timestep_float = timestep.flatten()[0].item()
        else:
            assert isinstance(timestep, float)
            timestep_float = timestep

        model = self.model
        assert isinstance(model, NunchakuFluxTransformer2dModel)

        bs, c, h, w = x.shape
        patch_size = self.config.get("patch_size", 2)
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = (h + (patch_size // 2)) // patch_size
        w_len = (w + (patch_size // 2)) // patch_size
        img_ids = FluxPipeline._prepare_latent_image_ids(bs, h_len, w_len, x.device, x.dtype)
        txt_ids = torch.zeros((context.shape[1], 3), device=x.device, dtype=x.dtype)

        # load and compose LoRA
        if self.loras != model.comfy_lora_meta_list:
            lora_to_be_composed = []
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
                lora_to_be_composed.append(({k: v for k, v in model.comfy_lora_sd_list[i].items()}, meta[1]))

            composed_lora = compose_lora(lora_to_be_composed)

            if len(composed_lora) == 0:
                model.reset_lora()
            else:
                if "x_embedder.lora_A.weight" in composed_lora:
                    new_in_channels = composed_lora["x_embedder.lora_A.weight"].shape[1]
                    current_in_channels = model.x_embedder.in_features
                    if new_in_channels < current_in_channels:
                        model.reset_x_embedder()
                model.update_lora_params(composed_lora)

        controlnet_block_samples = None if control is None else [y.to(x.dtype) for y in control["input"]]
        controlnet_single_block_samples = None if control is None else [y.to(x.dtype) for y in control["output"]]

        if self.pulid_pipeline is not None:
            self.model.transformer_blocks[0].pulid_ca = self.pulid_pipeline.pulid_ca

        if getattr(model, "_is_cached", False):
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
                        hidden_states=img,
                        encoder_hidden_states=context,
                        pooled_projections=y,
                        timestep=timestep,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        guidance=guidance if self.config["guidance_embed"] else None,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                    ).sample
                else:
                    out = self.customized_forward(
                        model,
                        hidden_states=img,
                        encoder_hidden_states=context,
                        pooled_projections=y,
                        timestep=timestep,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        guidance=guidance if self.config["guidance_embed"] else None,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                        **self.forward_kwargs,
                    ).sample
        else:
            if self.customized_forward is None:
                out = model(
                    hidden_states=img,
                    encoder_hidden_states=context,
                    pooled_projections=y,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance if self.config["guidance_embed"] else None,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                ).sample
            else:
                out = self.customized_forward(
                    model,
                    hidden_states=img,
                    encoder_hidden_states=context,
                    pooled_projections=y,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance if self.config["guidance_embed"] else None,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                    **self.forward_kwargs,
                ).sample
        if self.pulid_pipeline is not None:
            self.model.transformer_blocks[0].pulid_ca = None

        out = rearrange(
            out,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h_len,
            w=w_len,
            ph=patch_size,
            pw=patch_size,
        )
        out = out[:, :, :h, :w]

        self._prev_timestep = timestep_float
        return out
