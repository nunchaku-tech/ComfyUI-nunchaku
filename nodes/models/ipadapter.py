import logging
import os

import folder_paths
import torch
from torchvision import transforms

from nunchaku.models.IP_adapter.diffusers_adapters import apply_IPA_on_pipe
from nunchaku.models.IP_adapter.utils import undo_all_mods_on_transformer
from nunchaku.pipeline.pipeline_flux_IPA import FluxPipelineWrapper

log_level = os.getenv("LOG_LEVEL", "INFO").upper()

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


set_extra_config_model_path("ipadapter", "ipadapter")
set_extra_config_model_path("clip", "clip")


class NunchakuIPAdapterLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The nunchaku model."}),
            }
        }

    RETURN_TYPES = ("MODEL", "IPADAPTER_PIPELINE")
    FUNCTION = "load"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku IP-Adapter Loader"

    def load(self, model):
        device = model.model.diffusion_model.model.device
        pipeline = FluxPipelineWrapper.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=model.model.diffusion_model.model, torch_dtype=torch.bfloat16
        ).to(device)

        pipeline.load_ip_adapter(
            pretrained_model_name_or_path_or_dict="XLabs-AI/flux-ip-adapter-v2",
            weight_name="ip_adapter.safetensors",
            image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
        )
        return (
            model,
            pipeline,
        )


class NunchakuFluxIPAdapterApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter_pipeline": ("IPADAPTER_PIPELINE",),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipa"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku FLUX IP-Adapter Apply"

    def apply_ipa(
        self,
        model,
        ipadapter_pipeline: FluxPipelineWrapper,
        image,
        weight: float,
    ):
        to_pil_transformer = transforms.ToPILImage()
        image_tensor_chw = image[0].permute(2, 0, 1)
        pil_image = to_pil_transformer(image_tensor_chw)

        image_embeds, _ = ipadapter_pipeline.get_image_embeds(
            ip_adapter_image=pil_image,
        )

        undo_all_mods_on_transformer(ipadapter_pipeline.transformer)
        apply_IPA_on_pipe(ipadapter_pipeline, ip_adapter_scale=weight, repo_id="XLabs-AI/flux-ip-adapter-v2")

        ipadapter_pipeline.transformer.transformer_blocks[0].set_ip_hidden_states(image_embeds=image_embeds)

        model.model.diffusion_model.model = ipadapter_pipeline.transformer

        return (model,)
