"""
This module provides a wrapper for ComfyUI's Qwen-Image model configuration.
"""

import torch
from comfy.supported_models import QwenImage

from .. import model_base


class NunchakuQwenImage(QwenImage):
    """
    Wrapper for the Nunchaku Qwen-Image model configuration.
    """

    def get_model(
        self, state_dict: dict[str, torch.Tensor], prefix: str = "", device=None, **kwargs
    ) -> model_base.NunchakuQwenImage:
        """
        Instantiate and return a NunchakuQwenImage model.

        Parameters
        ----------
        state_dict : dict
            Model state dictionary.
        prefix : str, optional
            Prefix for parameter names (default is "").
        device : torch.device or str, optional
            Device to load the model onto.
        **kwargs
            Additional keyword arguments for model initialization.

        Returns
        -------
        model_base.NunchakuQwenImage
            Instantiated NunchakuQwenImage model.
        """
        out = model_base.NunchakuQwenImage(self, device=device, **kwargs)
        return out
