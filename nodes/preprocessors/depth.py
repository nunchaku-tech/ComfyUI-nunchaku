"""
This module provides a node for depth preprocessing using
`Depth Anything <https://huggingface.co/LiheYoung/depth-anything-large-hf>`_.
It is now deprecated and will be removed in October 2025.
"""

import logging
import os

import folder_paths
import numpy as np
import torch

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FluxDepthPreprocessor:
    """
    Node for applying a depth preprocessor model to an input image.

    .. warning::
        This node will be deprecated in October 2025.
        Please use the ``Depth Anything`` node in `comfyui_controlnet_aux <https://github.com/Fannovel16/comfyui_controlnet_aux>`_.
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types and tooltips for the node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and their descriptions for the node interface.
        """
        model_paths = []
        prefix = os.path.join(folder_paths.models_dir, "checkpoints")
        local_folders = os.listdir(prefix)
        local_folders = sorted(
            [
                folder
                for folder in local_folders
                if not folder.startswith(".") and os.path.isdir(os.path.join(prefix, folder))
            ]
        )
        model_paths = local_folders + model_paths
        return {
            "required": {
                "image": ("IMAGE", {}),
                "model_path": (
                    model_paths,
                    {"tooltip": "Name of the depth preprocessor model."},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "depth_preprocess"
    CATEGORY = "Nunchaku"
    TITLE = "FLUX Depth Preprocessor (Deprecated)"

    def depth_preprocess(self, image, model_path):
        """
        Apply the selected depth preprocessor model to the input image.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            The input image to process.
        model_path : str
            The name of the depth preprocessor model checkpoint.

        Returns
        -------
        tuple
            A tuple containing the depth map as a torch.Tensor.
        """
        logger.warning(
            "`FLUX.1 Depth Preprocessor` is deprecated and will be removed in October 2025. "
            "Please use `Depth Anything` in `comfyui_controlnet_aux` instead."
        )
        prefixes = folder_paths.folder_names_and_paths["checkpoints"][0]
        for prefix in prefixes:
            if os.path.exists(os.path.join(prefix, model_path)):
                model_path = os.path.join(prefix, model_path)
                break
        from image_gen_aux import DepthPreprocessor

        processor = DepthPreprocessor.from_pretrained(model_path)
        np_image = np.asarray(image)
        np_result = np.array(processor(np_image)[0].convert("RGB"))
        out_tensor = torch.from_numpy(np_result.astype(np.float32) / 255.0).unsqueeze(0)
        return (out_tensor,)
