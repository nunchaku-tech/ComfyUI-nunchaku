"""
Wrapper for :class:`~nunchaku.models.transformers.transformer_flux2.NunchakuFlux2Transformer2DModel`,
enabling integration with ComfyUI's forward calling convention.
"""

import torch
from comfy.ldm.common_dit import pad_to_patch_size
from einops import rearrange, repeat
from torch import nn

from nunchaku import NunchakuFlux2Transformer2DModel


class ComfyFlux2Wrapper(nn.Module):
    """
    Wrapper for :class:`~nunchaku.models.transformers.transformer_flux2.NunchakuFlux2Transformer2DModel`
    to support ComfyUI workflows.

    ComfyUI calls ``diffusion_model(x, t, context=context, y=y, guidance=guidance, ...)``
    where ``y`` (pooled projections) is not used by FLUX.2.

    Parameters
    ----------
    model : NunchakuFlux2Transformer2DModel
        The underlying quantized FLUX.2 transformer.
    config : dict
        Model configuration dict (e.g. from comfy_config["model_config"]).
    """

    def __init__(self, model: NunchakuFlux2Transformer2DModel, config: dict):
        super().__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config

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
        """
        Forward pass matching ComfyUI's diffusion_model calling convention.

        Parameters
        ----------
        x : torch.Tensor
            Noisy latent input of shape ``(B, C, H, W)``.
        timestep : torch.Tensor
            Diffusion timestep.
        context : torch.Tensor
            Encoder hidden states from the text encoder, shape ``(B, T, D)``.
        y : torch.Tensor or None
            Pooled text projections — **ignored** for FLUX.2 (no pooled conditioning).
        guidance : torch.Tensor
            Guidance scale tensor.
        control : dict, optional
            ControlNet outputs (not yet supported; ignored).
        transformer_options : dict, optional
            Additional transformer options from ComfyUI.

        Returns
        -------
        torch.Tensor
            Predicted noise in latent space, shape ``(B, C, H, W)``.
        """
        bs, c, h_orig, w_orig = x.shape
        patch_size = self.config.get("patch_size", 2)
        h_len = (h_orig + (patch_size // 2)) // patch_size
        w_len = (w_orig + (patch_size // 2)) // patch_size

        x = pad_to_patch_size(x, (patch_size, patch_size))
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        # Build image position IDs (h, w grid)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(
            0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype
        ).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(
            0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype
        ).unsqueeze(0)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        # Text position IDs are all zeros (text has no spatial structure)
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)

        use_guidance = self.config.get("guidance_embed", True)

        out = self.model(
            hidden_states=img,
            encoder_hidden_states=context,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance if use_guidance else None,
            return_dict=False,
        )[0]

        out = rearrange(
            out,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h_len,
            w=w_len,
            ph=patch_size,
            pw=patch_size,
        )
        out = out[:, :, :h_orig, :w_orig]
        return out
