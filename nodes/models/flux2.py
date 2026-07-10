"""
This module provides the :class:`NunchakuFlux2DiTLoader` class for loading Nunchaku FLUX.2 Klein models.
"""

import gc
import json
import logging
import os

import comfy.model_management
import comfy.model_patcher
import torch
from comfy.supported_models import Flux2

from nunchaku import NunchakuFlux2Transformer2DModel
from nunchaku.utils import is_turing

from ...wrappers.flux2 import ComfyFlux2Wrapper
from ..utils import get_filename_list, get_full_path_or_raise

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _build_comfy_config_from_metadata(metadata: dict) -> dict:
    """
    Construct a minimal ComfyUI model config dict from safetensors metadata.

    Falls back to FLUX.2 Klein defaults if the metadata lacks specific fields.

    Parameters
    ----------
    metadata : dict
        Raw metadata dict from the safetensors file.

    Returns
    -------
    dict
        A ``comfy_config`` dict with ``model_class`` and ``model_config`` keys.
    """
    diffusers_cfg = json.loads(metadata.get("config", "{}"))

    attention_head_dim = diffusers_cfg.get("attention_head_dim", 128)
    num_attention_heads = diffusers_cfg.get("num_attention_heads", 16)
    hidden_size = attention_head_dim * num_attention_heads  # e.g. 128*16=2048 for Klein

    in_channels = diffusers_cfg.get("in_channels", 64)
    out_channels = diffusers_cfg.get("out_channels", in_channels)
    patch_size = diffusers_cfg.get("patch_size", 2)
    guidance_embeds = diffusers_cfg.get("guidance_embeds", True)

    return {
        "model_class": "Flux2",
        "model_config": {
            "image_model": "flux2",
            "guidance_embed": guidance_embeds,
            "hidden_size": hidden_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "patch_size": patch_size,
            "disable_unet_model_creation": True,
        },
    }


class NunchakuFlux2DiTLoader:
    """
    Loader for Nunchaku FLUX.2 Klein models.

    Loads a Nunchaku-quantized FLUX.2 Klein transformer from a single safetensors
    file and returns a ComfyUI ``MODEL`` ready for use with standard samplers.

    The text encoder (Qwen3) can be loaded separately with ComfyUI's built-in
    dual-encoder loader or any FLUX.2-compatible text encoder node.
    """

    def __init__(self):
        self.transformer = None
        self.metadata = None
        self.model_path = None
        self.device = None
        self.cpu_offload = None
        self.data_type = None
        self.device = comfy.model_management.get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        safetensor_files = get_filename_list("diffusion_models")
        ngpus = torch.cuda.device_count()

        all_turing = all(is_turing(f"cuda:{i}") for i in range(max(1, torch.cuda.device_count())))
        dtype_options = ["float16"] if all_turing else ["bfloat16", "float16"]

        return {
            "required": {
                "model_path": (
                    safetensor_files,
                    {"tooltip": "The Nunchaku FLUX.2 Klein quantized model (.safetensors)."},
                ),
                "cpu_offload": (
                    ["auto", "enable", "disable"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Whether to enable CPU offload for the transformer. "
                            "'auto' enables it when GPU VRAM is less than 14 GiB."
                        ),
                    },
                ),
                "device_id": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": max(0, ngpus - 1),
                        "step": 1,
                        "display": "number",
                        "lazy": True,
                        "tooltip": "GPU device ID to load the model on.",
                    },
                ),
                "data_type": (
                    dtype_options,
                    {
                        "default": dtype_options[0],
                        "tooltip": (
                            "Model data type. Use 'float16' for 20-series GPUs that do not support bfloat16."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku FLUX.2 Klein DiT Loader"

    def load_model(
        self,
        model_path: str,
        cpu_offload: str,
        device_id: int,
        data_type: str,
        **kwargs,
    ):
        """
        Load the FLUX.2 Klein model and return a patched ComfyUI MODEL.

        Parameters
        ----------
        model_path : str
            Filename of the FLUX.2 Klein safetensors model in the diffusion_models folder.
        cpu_offload : str
            CPU offload policy: "auto", "enable", or "disable".
        device_id : int
            CUDA device index.
        data_type : str
            Weight dtype: "bfloat16" or "float16".

        Returns
        -------
        tuple
            A single-element tuple containing the loaded ComfyUI MODEL.
        """
        device = torch.device(f"cuda:{device_id}")
        model_path = get_full_path_or_raise("diffusion_models", model_path)

        if device_id >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid device_id {device_id}: only {torch.cuda.device_count()} GPU(s) available."
            )

        gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 2)

        if cpu_offload == "auto":
            cpu_offload_enabled = gpu_memory < 14336
        elif cpu_offload == "enable":
            cpu_offload_enabled = True
        else:
            cpu_offload_enabled = False

        torch_dtype = torch.float16 if data_type == "float16" else torch.bfloat16

        reload_needed = (
            self.model_path != model_path
            or self.device != device
            or self.cpu_offload != cpu_offload_enabled
            or self.data_type != data_type
        )

        if reload_needed:
            if self.transformer is not None:
                model_size = comfy.model_management.module_size(self.transformer)
                old = self.transformer
                self.transformer = None
                old.to("cpu")
                del old
                gc.collect()
                comfy.model_management.cleanup_models_gc()
                comfy.model_management.soft_empty_cache()
                comfy.model_management.free_memory(model_size, device)

            logger.info(f"Loading FLUX.2 Klein model from {model_path}")
            self.transformer, self.metadata = NunchakuFlux2Transformer2DModel.from_pretrained(
                model_path,
                offload=cpu_offload_enabled,
                device=device,
                torch_dtype=torch_dtype,
                return_metadata=True,
            )
            self.model_path = model_path
            self.device = device
            self.cpu_offload = cpu_offload_enabled
            self.data_type = data_type

        transformer = self.transformer

        # Resolve ComfyUI model config — prefer embedded comfy_config in metadata
        comfy_config = None
        if self.metadata:
            raw = self.metadata.get("comfy_config")
            if raw:
                comfy_config = json.loads(raw)
        if comfy_config is None:
            comfy_config = _build_comfy_config_from_metadata(self.metadata or {})

        model_config_dict = comfy_config["model_config"]
        model_config_dict.setdefault("disable_unet_model_creation", True)

        model_config = Flux2(model_config_dict)
        model_config.set_inference_dtype(torch_dtype, None)
        model_config.custom_operations = None

        model = model_config.get_model({})
        model.diffusion_model = ComfyFlux2Wrapper(transformer, config=model_config_dict)
        patcher = comfy.model_patcher.ModelPatcher(model, device, device_id)
        return (patcher,)
