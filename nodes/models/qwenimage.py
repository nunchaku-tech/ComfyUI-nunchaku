"""
This module provides the :class:`NunchakuQwenImageDiTLoader` class for loading Nunchaku Qwen-Image models.
"""

import json
import logging
import os

import comfy.model_patcher
import comfy.utils
import folder_paths
import torch
from comfy import model_detection, model_management

from nunchaku.utils import check_hardware_compatibility, get_precision_from_quantization_config

from ...model_configs.qwenimage import NunchakuQwenImage

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_diffusion_model_state_dict(
    sd: dict[str, torch.Tensor], metadata: dict[str, str] = {}, model_options: dict = {}
):
    """
    Load a Nunchaku-quantized Qwen-Image diffusion model.

    Parameters
    ----------
    sd : dict[str, torch.Tensor]
        The state dictionary of the model.
    metadata : dict[str, str], optional
        Metadata containing quantization configuration (default is empty dict).
    model_options : dict, optional
        Additional model options such as dtype or custom operations.

    Returns
    -------
    comfy.model_patcher.ModelPatcher
        The patched and loaded Qwen-Image model ready for inference.
    """
    quantization_config = json.loads(metadata.get("quantization_config", "{}"))
    precision = get_precision_from_quantization_config(quantization_config)
    rank = quantization_config.get("rank", 32)

    dtype = model_options.get("dtype", None)

    # Allow loading unets from checkpoint files
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd

    parameters = comfy.utils.calculate_parameters(sd)
    weight_dtype = comfy.utils.weight_dtype(sd)

    load_device = model_management.get_torch_device()
    check_hardware_compatibility(quantization_config, load_device)

    # model_config = model_detection.model_config_from_unet_config({"image_model": "qwen_image"}, state_dict=sd)
    model_config = NunchakuQwenImage(
        {"image_model": "qwen_image", "scale_shift": 0, "rank": rank, "precision": precision}
    )
    model_config.optimizations["fp8"] = False

    new_sd = sd

    offload_device = model_management.unet_offload_device()
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.scaled_fp8 is not None:
        weight_dtype = None

    if dtype is None:
        unet_dtype = model_management.unet_dtype(
            model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype
        )
    else:
        unet_dtype = dtype

    manual_cast_dtype = model_management.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get("custom_operations", model_config.custom_operations)
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    return comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)


class NunchakuQwenImageDiTLoader:
    """
    Loader for Nunchaku Qwen-Image models.

    Attributes
    ----------
    RETURN_TYPES : tuple
        Output types for the node ("MODEL",).
    FUNCTION : str
        Name of the function to call ("load_model").
    CATEGORY : str
        Node category ("Nunchaku").
    TITLE : str
        Node title ("Nunchaku Qwen-Image DiT Loader").
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Define the input types and tooltips for the node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and their descriptions for the node interface.
        """
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "The Nunchaku Qwen-Image model."},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Qwen-Image DiT Loader"

    def load_model(self, model_name: str, **kwargs):
        """
        Load the Qwen-Image model from file and return a patched model.

        Parameters
        ----------
        model_name : str
            The filename of the Qwen-Image model to load.

        Returns
        -------
        tuple
            A tuple containing the loaded and patched model.
        """
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        sd, metadata = comfy.utils.load_torch_file(model_path, return_metadata=True)
        model = load_diffusion_model_state_dict(sd, metadata=metadata)
        return (model,)
