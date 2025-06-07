from functools import partial
from types import MethodType

import numpy as np
import torch
import os

from nunchaku.models.pulid.pulid_forward import pulid_forward
# from nunchaku.pipeline.pipeline_flux_pulid import PuLIDPipeline
from .pulid_flux_pipeline import PuLIDPipeline

import folder_paths

if "ipadapter" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["ipadapter"] = os.path.join(folder_paths.models_dir, "ipadapter")
if "pulid" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["pulid"] = os.path.join(folder_paths.models_dir, "pulid")

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
                "ipadapter_name": (["Auto"] + folder_paths.get_filename_list("ipadapter") + folder_paths.get_filename_list("pulid"), {"default": "Auto"}),
                "clip_vision_name": (["Auto"] + folder_paths.get_filename_list("clip_vision"), {"default": "Auto"}),
                "version": (["v0.9.0", "v0.9.1"], {"default": "v0.9.0", "tooltip": ""})
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "PULID",
    )
    FUNCTION = "load"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Pulid Loader"

    def load(self, model, ipadapter_name, clip_vision_name, version):
        if os.path.exists(os.path.join(folder_paths.models_dir, "pulid_flux_v0.9.0.safetensors")): #Check possible existing 0.9.0 pulid model.
            pulid_path = os.path.join(folder_paths.models_dir, "pulid_flux_v0.9.0.safetensors")
        elif os.path.exists(os.path.join(folder_paths.models_dir, "pulid_flux_v0.9.1.safetensors")): #Check possible existing 0.9.1 pulid model.
            pulid_path = os.path.join(folder_paths.models_dir, "pulid_flux_v0.9.1.safetensors")
        else:
            pulid_path = folder_paths.get_full_path("pulid", f"pulid_flux_{version}.safetensors")
            if pulid_path == None:
                pulid_path = folder_paths.get_full_path("ipadapter", f"pulid_flux_{version}.safetensors")
                if pulid_path == None:
                    pulid_path = folder_paths.get_full_path("ipadapter", ipadapter_name)
                    if pulid_path == None:
                        pulid_path = folder_paths.get_full_path("pulid", ipadapter_name)
            
        clip_vision_name = 'EVA02_CLIP_L_336_psz14_s6B.pt' if clip_vision_name == "Auto" else clip_vision_name
        clip_vision_path = folder_paths.get_full_path("clip_vision", clip_vision_name) #Auto will return None
        if clip_vision_path == None: # Check possible existing eva_clip to avoid duplicate download.
            clip_vision_path = folder_paths.get_full_path("text_encoders", 'EVA02_CLIP_L_336_psz14_s6B.pt')
            if clip_vision_path is None:
                clip_vision_path = folder_paths.get_full_path("clip", 'EVA02_CLIP_L_336_psz14_s6B.pt')
        
        pulid_model = PuLIDPipeline(
            dit=model.model.diffusion_model.model,
            device=self.pulid_device,
            weight_dtype=self.weight_dtype,
            onnx_provider=self.onnx_provider,
            clip_vision_path=clip_vision_path
        )
        pulid_model.load_pretrain(pulid_path, version)

        return (
            model,
            pulid_model,
        )
