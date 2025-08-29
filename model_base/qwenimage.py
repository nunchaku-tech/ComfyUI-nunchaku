"""
Nunchaku Qwen-Image model base.

This module provides a wrapper for ComfyUI's Qwen-Image model base.
"""

import torch
from comfy.model_base import ModelType, QwenImage

from nunchaku.models.linear import SVDQW4A4Linear

from ..models.qwenimage import NunchakuQwenImageTransformer2DModel


class NunchakuQwenImage(QwenImage):
    """
    Wrapper for the Nunchaku Qwen-Image model.

    Parameters
    ----------
    model_config : object
        Model configuration object.
    model_type : ModelType, optional
        Type of the model (default is ModelType.FLUX).
    device : torch.device or str, optional
        Device to load the model onto.
    """

    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        """
        Initialize the NunchakuQwenImage model.

        Parameters
        ----------
        model_config : object
            Model configuration object.
        model_type : ModelType, optional
            Type of the model (default is ModelType.FLUX).
        device : torch.device or str, optional
            Device to load the model onto.
        """
        super(QwenImage, self).__init__(
            model_config, model_type, device=device, unet_model=NunchakuQwenImageTransformer2DModel
        )
        self.memory_usage_factor_conds = ("ref_latents",)

    def load_model_weights(self, sd: dict[str, torch.Tensor], unet_prefix: str = ""):
        """
        Load model weights into the diffusion model.

        Parameters
        ----------
        sd : dict of str to torch.Tensor
            State dictionary containing model weights.
        unet_prefix : str, optional
            Prefix for UNet weights (default is "").

        Raises
        ------
        ValueError
            If a required key is missing from the state dictionary.
        """
        diffusion_model = self.diffusion_model
        state_dict = diffusion_model.state_dict()
        for k in state_dict.keys():
            if k not in sd:
                if ".wcscales" not in k:
                    raise ValueError(f"Key {k} not found in state_dict")
                sd[k] = torch.ones_like(state_dict[k])
        for n, m in diffusion_model.named_modules():
            if isinstance(m, SVDQW4A4Linear):
                if m.wtscale is not None:
                    m.wtscale = sd.pop(f"{n}.wtscale", 1.0)
        diffusion_model.load_state_dict(sd, strict=True)
