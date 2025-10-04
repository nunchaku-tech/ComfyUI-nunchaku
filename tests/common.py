from comfy.model_downloader import add_known_models
from comfy.model_downloader_types import HuggingFile
from nunchaku.utils import get_precision
from PIL import Image
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio
from torchmetrics.multimodal import CLIPImageQualityAssessment
from diffusers.utils import load_image
import torch
import numpy as np


def compute_metrics(gen_image_path: str, ref_image_path: str) -> tuple[float, float, float]:
    # clip_iqa metric
    metric = CLIPImageQualityAssessment(model_name_or_path="openai/clip-vit-large-patch14").to("cuda")
    image = Image.open(gen_image_path).convert("RGB")
    gen_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(torch.float32).unsqueeze(0).to("cuda")
    clip_iqa = metric(gen_tensor).item()
    print(f"CLIP-IQA: {clip_iqa}")

    ref_image = load_image(ref_image_path).convert("RGB")
    metric = LearnedPerceptualImagePatchSimilarity().to("cuda")
    ref_tensor = torch.from_numpy(np.array(ref_image)).permute(2, 0, 1).to(torch.float32)
    ref_tensor = ref_tensor.unsqueeze(0).to("cuda")
    lpips = metric(gen_tensor / 255, ref_tensor / 255).item()
    print(f"LPIPS: {lpips}")

    metric = PeakSignalNoiseRatio(data_range=(0, 255)).cuda()
    psnr = metric(gen_tensor, ref_tensor).item()
    print(f"PSNR: {psnr}")
    return clip_iqa, lpips, psnr


def prepare_models():
    add_known_models(
        "text_encoders",
        HuggingFile(repo_id="nunchaku-tech/nunchaku-t5", filename="awq-int4-flux.1-t5xxl.safetensors"),
    )

    add_known_models(
        "diffusion_models",
        (
            HuggingFile(
                repo_id="nunchaku-tech/nunchaku-flux.1-dev",
                filename=f"svdq-{get_precision()}_r32-flux.1-dev.safetensors",
            ),
            HuggingFile(
                repo_id="nunchaku-tech/nunchaku-flux.1-schnell",
                filename=f"svdq-{get_precision()}_r32-flux.1-schnell.safetensors",
            ),
            HuggingFile(
                repo_id="nunchaku-tech/nunchaku-flux.1-depth-dev",
                filename=f"svdq-{get_precision()}_r32-flux.1-depth-dev.safetensors",
            ),
            HuggingFile(
                repo_id="nunchaku-tech/nunchaku-flux.1-canny-dev",
                filename=f"svdq-{get_precision()}_r32-flux.1-canny-dev.safetensors",
            ),
            HuggingFile(
                repo_id="nunchaku-tech/nunchaku-flux.1-fill-dev",
                filename=f"svdq-{get_precision()}_r32-flux.1-fill-dev.safetensors",
            ),
            HuggingFile(
                repo_id="nunchaku-tech/nunchaku-shuttle-jaguar",
                filename=f"svdq-{get_precision()}_r32-shuttle-jaguar.safetensors",
            ),
            HuggingFile(
                repo_id="nunchaku-tech/nunchaku-flux.1-kontext-dev",
                filename=f"svdq-{get_precision()}_r32-flux.1-kontext-dev.safetensors",
            ),
            HuggingFile(
                repo_id="nunchaku-tech/nunchaku-qwen-image",
                filename=f"svdq-{get_precision()}_r32-qwen-image.safetensors",
            ),
            HuggingFile(
                repo_id="nunchaku-tech/nunchaku-qwen-image",
                filename=f"svdq-{get_precision()}_r128-qwen-image.safetensors",
            ),
            HuggingFile(
                repo_id="nunchaku-tech/nunchaku-qwen-image-edit",
                filename=f"svdq-{get_precision()}_r32-qwen-image-edit.safetensors",
            ),
            HuggingFile(
                repo_id="nunchaku-tech/nunchaku-qwen-image-edit",
                filename=f"svdq-{get_precision()}_r128-qwen-image-edit.safetensors",
            ),
            HuggingFile(
                repo_id="nunchaku-tech/nunchaku-qwen-image-edit-2509",
                filename=f"svdq-{get_precision()}_r32-qwen-image-edit-2509.safetensors",
            ),
            HuggingFile(
                repo_id="nunchaku-tech/nunchaku-qwen-image-edit-2509",
                filename=f"svdq-{get_precision()}_r128-qwen-image-edit-2509.safetensors",
            ),
        ),
    )
