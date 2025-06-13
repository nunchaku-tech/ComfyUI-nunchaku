"""
Adapted from https://github.com/lldacing/ComfyUI_PuLID_Flux_ll
"""

import logging
import os
from functools import partial
from types import MethodType

import cv2
import numpy as np
import torch
from insightface.utils import face_align
from torchvision import transforms
from torchvision.transforms import functional

import comfy
import folder_paths
from comfy import model_management
from nunchaku.models.pulid.pulid_forward import pulid_forward
from nunchaku.pipeline.pipeline_flux_pulid import PuLIDPipeline
from .pulid_utils.face_restoration_helper import FaceRestoreHelper, get_face_by_index

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


def tensor_to_image(tensor):
    image = tensor.mul(255).clamp(0, 255).byte().cpu()
    image = image[..., [2, 1, 0]].numpy()
    return image


def image_to_tensor(image):
    tensor = torch.clamp(torch.from_numpy(image).float() / 255.0, 0, 1)
    tensor = tensor[..., [2, 1, 0]]
    return tensor


def to_gray(img):
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.repeat(1, 3, 1, 1)
    return x


FACEXLIB_DIR = folder_paths.get_folder_paths("facexlib")[0]


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

    def apply(
        self,
        model,
        pulid_flux,
        eva_clip,
        face_analysis,
        image,
        weight,
        start_at,
        end_at,
        attn_mask=None,
        options={},
        unique_id=None,
    ):
        import ipdb

        ipdb.set_trace()

        device = comfy.model_management.get_torch_device()
        dtype = model.model.diffusion_model.dtype
        # Issue: https://github.com/balazik/ComfyUI-PuLID-Flux/issues/6
        if model.model.manual_cast_dtype is not None:
            dtype = model.model.manual_cast_dtype

        eva_clip.to(device, dtype=dtype)
        pulid_flux.model.to(dtype=dtype)
        comfy.model_management.load_models_gpu([pulid_flux], force_full_load=True)

        if attn_mask is not None:
            if attn_mask.dim() > 3:
                attn_mask = attn_mask.squeeze(-1)
            elif attn_mask.dim() < 3:
                attn_mask = attn_mask.unsqueeze(0)
            raise NotImplementedError()

        image = tensor_to_image(image)

        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            parsing_model="bisenet",
            save_ext="png",
            device=device,
            model_rootpath=FACEXLIB_DIR,
        )

        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        cond = []

        input_face_sort = options.get("input_faces_order", "large-small")
        input_face_index = options.get("input_faces_index", 0)
        input_face_align_mode = options.get("input_faces_align_mode", 1)
        # Analyse multiple images at multiple sizes and combine largest area embeddings
        for i in range(image.shape[0]):
            # get insightface embeddings
            for size in [(size, size) for size in range(640, 256, -64)]:
                face_analysis.det_model.input_size = size
                face_info = face_analysis.get(image[i])
                if face_info:
                    face_info, index, sorted_faces = get_face_by_index(
                        face_info, face_sort_rule=input_face_sort, face_index=input_face_index
                    )
                    bboxes = [face.bbox for face in sorted_faces]
                    iface_embeds = torch.from_numpy(face_info.embedding).unsqueeze(0).to(device, dtype=dtype)
                    break
            else:
                # No face detected, skip this image
                logging.warning(f"Warning: No face detected in image {str(i)}")
                continue

            if input_face_align_mode == 1:
                image_size = 512
                M = face_align.estimate_norm(face_info.kps, image_size=image_size)
                align_face = cv2.warpAffine(
                    image[i], M, (image_size, image_size), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132)
                )
                # align_face = face_align.norm_crop(image[i], landmark=face_info.kps, image_size=image_size)
                del M
            else:
                # get eva_clip embeddings
                face_helper.clean_all()
                face_helper.read_image(image[i])
                face_helper.get_face_landmarks_5(ref_sort_bboxes=bboxes, face_index=input_face_index)
                face_helper.align_warp_face()

                if len(face_helper.cropped_faces) == 0:
                    # No face detected, skip this image
                    continue

                # Get aligned face image
                align_face = face_helper.cropped_faces[0]
            # Convert bgr face image to tensor
            align_face = image_to_tensor(align_face).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            parsing_out = face_helper.face_parse(
                functional.normalize(align_face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            )[0]
            parsing_out = parsing_out.argmax(dim=1, keepdim=True)
            bg = sum(parsing_out == i for i in bg_label).bool()
            white_image = torch.ones_like(align_face)
            # Only keep the face features
            face_features_image = torch.where(bg, white_image, to_gray(align_face))

            # Transform img before sending to eva_clip
            # Apparently MPS only supports NEAREST interpolation?
            face_features_image = functional.resize(
                face_features_image,
                eva_clip.image_size,
                transforms.InterpolationMode.BICUBIC if "cuda" in device.type else transforms.InterpolationMode.NEAREST,
            ).to(device, dtype=dtype)
            face_features_image = functional.normalize(face_features_image, eva_clip.image_mean, eva_clip.image_std)

            # eva_clip
            id_cond_vit, id_vit_hidden = eva_clip(
                face_features_image, return_all_features=False, return_hidden=True, shuffle=False
            )
            id_cond_vit = id_cond_vit.to(device, dtype=dtype)
            for idx in range(len(id_vit_hidden)):
                id_vit_hidden[idx] = id_vit_hidden[idx].to(device, dtype=dtype)

            id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))

            # Combine embeddings
            id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)

            # Pulid_encoder
            cond.append(pulid_flux.model.get_embeds(id_cond, id_vit_hidden))

        eva_clip.to(torch.device("cpu"))
        if not cond:
            # No faces detected, return the original model
            logging.warning("PuLID warning: No faces detected in any of the given images, returning unmodified model.")
            del eva_clip, face_analysis, pulid_flux, face_helper, attn_mask
            return (model,)

        # average embeddings
        cond = torch.cat(cond).to(device, dtype=dtype)
        if cond.shape[0] > 1:
            cond = torch.mean(cond, dim=0, keepdim=True)

        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        patch_kwargs = {
            "pulid_model": pulid_flux,
            "weight": weight,
            "embedding": cond,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
            "mask": attn_mask,
        }

        # ca_idx = 0
        # for i in range(19):
        #     if i % pulid_flux.model.double_interval == 0:
        #         patch_kwargs["ca_idx"] = ca_idx
        #         set_model_dit_patch_replace(model, patch_kwargs, ("double_block", i))
        #         ca_idx += 1
        # for i in range(38):
        #     if i % pulid_flux.model.single_interval == 0:
        #         patch_kwargs["ca_idx"] = ca_idx
        #         set_model_dit_patch_replace(model, patch_kwargs, ("single_block", i))
        #         ca_idx += 1

        # if len(model.get_additional_models_with_key("pulid_flux_model_patcher")) == 0:
        #     model.set_additional_models("pulid_flux_model_patcher", [pulid_flux])
        #
        # if len(model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, wrappers_name)) == 0:
        #     # Just add it once when connecting in series
        #     model.add_wrapper_with_key(
        #         comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        #         wrappers_name,
        #         pulid_outer_sample_wrappers_with_override,
        #     )
        # if len(model.get_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, wrappers_name)) == 0:
        #     # Just add it once when connecting in series
        #     model.add_wrapper_with_key(
        #         comfy.patcher_extension.WrappersMP.APPLY_MODEL, wrappers_name, pulid_apply_model_wrappers
        #     )
        #
        # del eva_clip, face_analysis, pulid_flux, face_helper, attn_mask
        return (model,)
