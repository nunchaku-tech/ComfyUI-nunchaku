"""
Adapted from https://github.com/lldacing/ComfyUI_PuLID_Flux_ll
"""

import logging
import os
from functools import partial
from types import MethodType

import numpy as np
import torch

import folder_paths
from nunchaku.models.pulid.pulid_forward import pulid_forward
from nunchaku.pipeline.pipeline_flux_pulid import PuLIDPipeline

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuPulidApply:
    def __init__(self):
        self.pulid_device = "cuda"
        self.weight_dtype = torch.bfloat16
        self.onnx_provider = "gpu"
        self.pretrained_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pulid": ("PULID", {"tooltip": "from Nunchaku Pulid Loader"}),
                "image": ("IMAGE", {"tooltip": "The image to encode"}),
                "model": ("MODEL", {"tooltip": "The nunchaku model."}),
                "ip_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "ip_weight"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Pulid Apply (Deprecated)"

    def apply(self, pulid, image, model, ip_weight):
        logger.warning(
            'This node is deprecated and will be removed in the v0.5.0. Directly use "Nunchaku FLUX PuLID Apply" instead.'
        )

        image = image.squeeze().cpu().numpy() * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        id_embeddings, _ = pulid.get_id_embedding(image)
        model.model.diffusion_model.model.forward = MethodType(
            partial(pulid_forward, id_embeddings=id_embeddings, id_weight=ip_weight), model.model.diffusion_model.model
        )
        return (model,)


class NunchakuPulidLoader:
    def __init__(self):
        self.pulid_device = "cuda"
        self.weight_dtype = torch.bfloat16
        self.onnx_provider = "gpu"
        self.pretrained_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The nunchaku model."}),
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "PULID",
    )
    FUNCTION = "load"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Pulid Loader (Deprecated)"

    def load(self, model):
        logger.warning(
            'This node is deprecated and will be removed in the v0.5.0. Directly use "Nunchaku FLUX PuLID Apply" instead.'
        )
        pulid_model = PuLIDPipeline(
            dit=model.model.diffusion_model.model,
            device=self.pulid_device,
            weight_dtype=self.weight_dtype,
            onnx_provider=self.onnx_provider,
        )
        pulid_model.load_pretrain(self.pretrained_model)

        return (model, pulid_model)


class NunchakuPuLIDLoaderV2:
    # def __init__(self):
    #     self.pulid_device = "cuda"
    #     self.weight_dtype = torch.bfloat16
    #     self.onnx_provider = "gpu"
    #     self.pretrained_model = None

    @classmethod
    def INPUT_TYPES(s):
        pulid_files = folder_paths.get_filename_list("pulid")
        clip_files = folder_paths.get_filename_list("clip")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The nunchaku model."}),
                "pulid_file": (pulid_files, {"tooltip": "Path to the PuLID model."}),
                "eva_clip_file": (clip_files, {"tooltip": "Path to the EVA clip model."}),
                "insight_face_provider": (
                    ["CPU", "CUDA", "ROCM"],
                    {"default": "gpu", "tooltip": "InsightFace ONNX provider."},
                ),
            }
        }

    RETURN_TYPES = ("model", "pulid_pipeline")
    FUNCTION = "load"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku PuLID Loader V2"

    def load(self, model):

        pulid_pipline = PuLIDPipeline(
            dit=model.model.diffusion_model.model,
            device=self.pulid_device,
            weight_dtype=self.weight_dtype,
            onnx_provider=self.onnx_provider,
        )

        return (model, pulid_pipline)
