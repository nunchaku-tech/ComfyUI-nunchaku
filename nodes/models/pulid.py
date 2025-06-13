import logging
import os
from functools import partial
from types import MethodType

import numpy as np
import torch

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
    TITLE = "Nunchaku Pulid Apply"

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
    TITLE = "Nunchaku Pulid Loader"

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

        return (
            model,
            pulid_model,
        )


class NunchakuFLUXPuLIDApply:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "pulid_flux": ("PULIDFLUX",),
                "eva_clip": ("EVA_CLIP",),
                "face_analysis": ("FACEANALYSIS",),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "attn_mask": ("MASK",),
                "options": ("OPTIONS",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku FLUX PuLID Apply"
