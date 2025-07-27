"""
This module provides nodes load and apply IP-Adapter models
to FLUX pipelines, enabling image-based conditioning for generative models.
"""

import logging
import os
from typing import Any, List, Optional

import torch
from diffusers import FluxPipeline
from torchvision import transforms

from nunchaku.models.ip_adapter.diffusers_adapters import apply_IPA_on_pipe
from nunchaku.models.ip_adapter.utils import undo_all_mods_on_transformer

from .utils import set_extra_config_model_path

log_level = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class IPAFluxPipelineWrapper(FluxPipeline):
    """
    FluxPipeline wrapper with IP-Adapter support.
    """

    @torch.no_grad()
    def get_image_embeds(
        self,
        num_images_per_prompt: int = 1,
        ip_adapter_image: Optional[Any] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[Any] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    ) -> (Optional[torch.Tensor], Optional[torch.Tensor]):
        """
        Compute image and negative image embeddings for IP-Adapter.

        Parameters
        ----------
        num_images_per_prompt : int, optional
            Number of images per prompt (default is 1).
        ip_adapter_image : Any, optional
            Input image for positive conditioning.
        ip_adapter_image_embeds : list of torch.Tensor, optional
            Precomputed positive image embeddings.
        negative_ip_adapter_image : Any, optional
            Input image for negative conditioning.
        negative_ip_adapter_image_embeds : list of torch.Tensor, optional
            Precomputed negative image embeddings.

        Returns
        -------
        image_embeds : torch.Tensor or None
            Positive image embeddings.
        negative_image_embeds : torch.Tensor or None
            Negative image embeddings.
        """
        batch_size = 1

        device = self.transformer.device

        image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image=ip_adapter_image,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
                device=device,
                num_images_per_prompt=batch_size * num_images_per_prompt,
            )
            image_embeds = self.transformer.encoder_hid_proj(image_embeds)

        negative_image_embeds = None
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image=negative_ip_adapter_image,
                ip_adapter_image_embeds=negative_ip_adapter_image_embeds,
                device=device,
                num_images_per_prompt=batch_size * num_images_per_prompt,
            )
            negative_image_embeds = self.transformer.encoder_hid_proj(negative_image_embeds)

        return image_embeds, negative_image_embeds


set_extra_config_model_path("ipadapter", "ipadapter")
set_extra_config_model_path("clip", "clip")


class NunchakuIPAdapterLoader:
    """
    Node for loading Nunchaku IP-Adapter pipelines.

    .. warning::
        This node will automatically download the IP-Adapter and associated CLIP models from Hugging Face.
        Custom model paths are not supported for now.
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
        """
        Load the IP-Adapter pipeline and attach it to the given model.

        Parameters
        ----------
        model : object
            The Nunchaku model to which the IP-Adapter will be attached.
            It should be loaded with :class:`~comfyui_nunchaku.nodes.models.flux.NunchakuFluxDiTLoader`.

        Returns
        -------
        tuple
            The original model and the loaded IP-Adapter pipeline.
        """
        device = model.model.diffusion_model.model.device
        pipeline = IPAFluxPipelineWrapper.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=model.model.diffusion_model.model, torch_dtype=torch.bfloat16
        ).to(device)

        pipeline.load_ip_adapter(
            pretrained_model_name_or_path_or_dict="XLabs-AI/flux-ip-adapter-v2",
            weight_name="ip_adapter.safetensors",
            image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
        )
        return (model, pipeline)


class NunchakuFluxIPAdapterApply:
    """
    Node for applying IP-Adapter to a Nunchaku model using a given image and weight.
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
        ipadapter_pipeline: IPAFluxPipelineWrapper,
        image,
        weight: float,
    ):
        """
        Apply the IP-Adapter to the given model using the provided image and weight.

        Parameters
        ----------
        model : object
            The Nunchaku model to modify.
        ipadapter_pipeline : IPAFluxPipelineWrapper
            The IP-Adapter pipeline.
        image : torch.Tensor
            The input image tensor.
        weight : float
            The scale/weight for the IP-Adapter.

        Returns
        -------
        tuple
            The modified model.
        """
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
