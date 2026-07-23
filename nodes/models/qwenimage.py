"""
This module provides the :class:`NunchakuQwenImageDiTLoader` class for loading Nunchaku Qwen-Image models.
"""

import json
import logging

import comfy.utils
import torch
from comfy import model_detection, model_management

from nunchaku.utils import check_hardware_compatibility, get_precision_from_quantization_config

from ...model_configs.qwenimage import NunchakuQwenImage
from ...model_patcher.common import NunchakuModelPatcher
from ..utils import get_filename_list, get_full_path_or_raise, get_nunchaku_model_list, get_nunchaku_model_full_path

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

    offload_device = model_management.unet_offload_device()
    model_config = NunchakuQwenImage(
        {
            "image_model": "qwen_image",
            "scale_shift": 0,
            "rank": rank,
            "precision": precision,
            "transformer_offload_device": offload_device if model_options.get("cpu_offload_enabled", False) else None,
        }
    )
    model_config.optimizations["fp8"] = False

    new_sd = sd

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

    model = model_config.get_model(new_sd, "", load_device)
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    return NunchakuModelPatcher(model, load_device=load_device, offload_device=offload_device)


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

    def __init__(self):
        self.model = None
        self.model_path = None
        self.cpu_offload = None

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
                    get_nunchaku_model_list(),
                    {"tooltip": "The Nunchaku Qwen-Image model."},
                ),
                "cpu_offload": (
                    ["auto", "enable", "disable"],
                    {
                        "default": "auto",
                        "tooltip": "Whether to enable CPU offload for the transformer model."
                        "auto' will enable it if the GPU memory is less than 15G.",
                    },
                ),
            },
            "optional": {
                "num_blocks_on_gpu": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 60,
                        "tooltip": (
                            "When CPU offload is enabled, this option determines how many transformer blocks remain on GPU memory. "
                            "Increasing this value decreases CPU RAM usage but increases GPU memory usage."
                        ),
                    },
                ),
                "use_pin_memory": (
                    ["enable", "disable"],
                    {
                        "default": "disable",
                        "tooltip": (
                            "Enable this to use pinned memory for transformer blocks when CPU offload is enabled. "
                            "This can improve data transfer speed between CPU and GPU, but may increase system memory usage."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Qwen-Image DiT Loader"

    def load_model(
        self, model_name: str, cpu_offload: str, num_blocks_on_gpu: int = 1, use_pin_memory: str = "disable", **kwargs
    ):
        """
        Load the Qwen-Image model from file and return a patched model.

        Parameters
        ----------
        model_name : str
            The filename of the Qwen-Image model to load.
        cpu_offload : str
            Whether to enable CPU offload for the transformer model.
        num_blocks_on_gpu : int
            The number of transformer blocks to keep on GPU when CPU offload is enabled.
        use_pin_memory : str
            Whether to use pinned memory for the transformer blocks when CPU offload is enabled.

        Returns
        -------
        tuple
            A tuple containing the loaded and patched model.
        """
        model_path = get_nunchaku_model_full_path(model_name)

        if cpu_offload == "auto":
            device = model_management.get_torch_device()
            vram_for_weights = model_management.maximum_vram_for_weights(device)
            if vram_for_weights < 12 * 1024**3:  # less than ~12 GiB available for weights
                cpu_offload_enabled = True
                logger.info("VRAM for weights < 12GiB, enabling CPU offload")
            else:
                cpu_offload_enabled = False
                logger.info("VRAM for weights >= 12GiB, disabling CPU offload")
        elif cpu_offload == "enable":
            cpu_offload_enabled = True
            logger.info("Enabling CPU offload")
        else:
            assert cpu_offload == "disable", "Invalid CPU offload option"
            cpu_offload_enabled = False
            logger.info("Disabling CPU offload")

        if (
            self.model_path != model_path
            or self.cpu_offload != cpu_offload_enabled
        ):
            sd, metadata = comfy.utils.load_torch_file(model_path, return_metadata=True)
            self.model = load_diffusion_model_state_dict(
                sd, metadata=metadata, model_options={"cpu_offload_enabled": cpu_offload_enabled}
            )
            self.model_path = model_path
            self.cpu_offload = cpu_offload_enabled

        if cpu_offload_enabled:
            assert use_pin_memory in ["enable", "disable"], "Invalid use_pin_memory option"
            self.model.model.diffusion_model.set_offload(
                cpu_offload_enabled, num_blocks_on_gpu=num_blocks_on_gpu, use_pin_memory=use_pin_memory == "enable"
            )

        return (self.model,)
