"""
Adapted from https://github.com/lldacing/ComfyUI_PuLID_Flux_ll
"""

import copy
import logging
import os
from functools import partial
from types import MethodType

import comfy
import folder_paths
import numpy as np
import torch

from nunchaku.models.pulid.pulid_forward import pulid_forward
from nunchaku.pipeline.pipeline_flux_pulid import PuLIDPipeline

from ...wrappers.flux import ComfyFluxWrapper

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_extra_config_model_path(extra_config_models_dir_key, models_dir_name: str):
    models_dir_default = os.path.join(folder_paths.models_dir, models_dir_name)
    if extra_config_models_dir_key not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[extra_config_models_dir_key] = (
            [os.path.join(folder_paths.models_dir, models_dir_name)],
            folder_paths.supported_pt_extensions,
        )
    else:
        if not os.path.exists(models_dir_default):
            os.makedirs(models_dir_default, exist_ok=True)
        folder_paths.add_model_folder_path(extra_config_models_dir_key, models_dir_default, is_default=True)


set_extra_config_model_path("pulid", "pulid")
set_extra_config_model_path("insightface", "insightface")
set_extra_config_model_path("facexlib", "facexlib")


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

    RETURN_TYPES = ("MODEL", "PULID")
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


class NunchakuFluxPuLIDApplyV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "pulid_pipline": ("PULID_PIPELINE",),
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
    TITLE = "Nunchaku FLUX PuLID Apply V2"

    def apply(
        self,
        model,
        pulid_pipline: PuLIDPipeline,
        image,
        weight: float,
        start_at: float,
        end_at: float,
        attn_mask=None,
        options=None,
        unique_id=None,
    ):
        image = image.squeeze().cpu().numpy() * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        id_embeddings, _ = pulid_pipline.get_id_embedding(image)

        model_wrapper = model.model.diffusion_model
        assert isinstance(model_wrapper, ComfyFluxWrapper)
        transformer = model_wrapper.model

        model_wrapper.model = None
        ret_model = copy.deepcopy(model)  # copy everything except the model
        ret_model_wrapper = ret_model.model.diffusion_model
        assert isinstance(ret_model_wrapper, ComfyFluxWrapper)
        ret_model_wrapper.model = transformer
        model_wrapper.model = transformer

        ret_model_wrapper.pulid_pipeline = pulid_pipline
        ret_model_wrapper.customized_forward = partial(
            pulid_forward, id_embeddings=id_embeddings, id_weight=weight, start_timestep=start_at, end_timestep=end_at
        )

        if attn_mask is not None:
            raise NotImplementedError("Attn mask is not supported for now in Nunchaku FLUX PuLID Apply V2.")

        return (ret_model,)


class NunchakuPuLIDLoaderV2:
    @classmethod
    def INPUT_TYPES(s):
        pulid_files = folder_paths.get_filename_list("pulid")
        clip_files = folder_paths.get_filename_list("clip")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The nunchaku model."}),
                "pulid_file": (pulid_files, {"tooltip": "Path to the PuLID model."}),
                "eva_clip_file": (clip_files, {"tooltip": "Path to the EVA clip model."}),
                "insight_face_provider": (["gpu", "cpu"], {"default": "gpu", "tooltip": "InsightFace ONNX provider."}),
            }
        }

    RETURN_TYPES = ("MODEL", "PULID_PIPELINE")
    FUNCTION = "load"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku PuLID Loader V2"

    def load(self, model, pulid_file: str, eva_clip_file: str, insight_face_provider: str):
        model_wrapper = model.model.diffusion_model
        assert isinstance(model_wrapper, ComfyFluxWrapper)
        transformer = model_wrapper.model

        device = comfy.model_management.get_torch_device()
        weight_dtype = next(transformer.parameters()).dtype

        pulid_path = folder_paths.get_full_path_or_raise("pulid", pulid_file)
        eva_clip_path = folder_paths.get_full_path_or_raise("clip", eva_clip_file)
        insightface_dirpath = folder_paths.get_folder_paths("insightface")[0]
        facexlib_dirpath = folder_paths.get_folder_paths("facexlib")[0]

        pulid_pipline = PuLIDPipeline(
            dit=transformer,
            device=device,
            weight_dtype=weight_dtype,
            onnx_provider=insight_face_provider,
            pulid_path=pulid_path,
            eva_clip_path=eva_clip_path,
            insightface_dirpath=insightface_dirpath,
            facexlib_dirpath=facexlib_dirpath,
        )

        return (model, pulid_pipline)
