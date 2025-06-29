import gc
import os
import subprocess

import numpy as np
import pytest
import torch
from diffusers.utils import load_image
from PIL import Image
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio
from torchmetrics.multimodal import CLIPImageQualityAssessment

from nunchaku.utils import get_precision

script_dir = os.path.join(os.path.dirname(__file__), "scripts")


@pytest.mark.parametrize(
    "script_name, expected_clip_iqa, expected_lpips, expected_psnr",
    [
        ("nunchaku-flux1-redux-dev.py", 0.9, 0.137, 18.9),
        ("nunchaku-flux1-dev-controlnet_upscaler.py", 0.9, 0.1, 26),
        ("nunchaku-flux1-dev-controlnet_union_pro2.py", 0.9, 0.1, 26),
        ("nunchaku-flux1-depth-lora.py", 0.7, 0.13, 21),
        ("nunchaku-flux1-canny.py", 0.9, 0.1, 26),
        ("nunchaku-flux1-schnell.py", 0.9, 0.29, 19.3),
        ("nunchaku-flux1-depth.py", 0.9, 0.13, 26),
        ("nunchaku-shuttle-jaguar.py", 0.9, 0.157, 23.9),
        ("nunchaku-flux1-fill.py", 0.9, 0.1, 26),
        ("nunchaku-flux1-fill-removalV2.py", 0.56, 0.13, 26),
        ("nunchaku-flux1-dev.py", 0.9, 0.28, 19.7),
        ("nunchaku-flux1-canny-lora.py", 0.9, 0.1, 25),
        ("nunchaku-flux1-dev-qencoder.py", 0.9, 0.27, 16.3),
        ("nunchaku-flux1-dev-hand_drawn_game.py", 0.92, 0.254, 20),
        ("nunchaku-flux1-dev-pulid.py", 0.9, 0.194, 15.8),
        ("nunchaku-flux1-kontext-dev.py", 0.9, 0.1, 18.3),
        ("nunchaku-flux1-kontext-dev-turbo_lora.py", 0.87, 0.13, 18.8),
    ],
)
def test_workflows(script_name: str, expected_clip_iqa: float, expected_lpips: float, expected_psnr: float):
    gc.collect()
    torch.cuda.empty_cache()
    script_path = os.path.join(script_dir, script_name)

    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    print(f"Running {script_path} -> Return code: {result.returncode}")
    assert result.returncode == 0, f"{script_path} failed with code {result.returncode}"

    path = open("image_path.txt", "r").read().strip()

    # clip_iqa metric
    metric = CLIPImageQualityAssessment(model_name_or_path="openai/clip-vit-large-patch14").to("cuda")
    image = Image.open(path).convert("RGB")
    gen_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(torch.float32).unsqueeze(0).to("cuda")
    clip_iqa = metric(gen_tensor).item()
    print(f"CLIP-IQA: {clip_iqa}")

    # lpips metric
    ref_image_url = (
        f"https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/ComfyUI-nunchaku/ref_images/"
        f"{get_precision()}/{script_name.replace('.py', '.png')}"
    )
    ref_image = load_image(ref_image_url).convert("RGB")
    metric = LearnedPerceptualImagePatchSimilarity().to("cuda")
    ref_tensor = torch.from_numpy(np.array(ref_image)).permute(2, 0, 1).to(torch.float32)
    ref_tensor = ref_tensor.unsqueeze(0).to("cuda")
    lpips = metric(gen_tensor / 255, ref_tensor / 255).item()
    print(f"LPIPS: {lpips}")

    metric = PeakSignalNoiseRatio(data_range=(0, 255)).cuda()
    psnr = metric(gen_tensor, ref_tensor).item()
    print(f"PSNR: {psnr}")

    assert clip_iqa >= expected_clip_iqa * 0.85
    assert lpips <= expected_lpips * 1.15
    assert psnr >= expected_psnr * 0.85
