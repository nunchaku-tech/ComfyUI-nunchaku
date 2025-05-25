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
scripts = [f for f in os.listdir(script_dir) if f.endswith(".py")]


@pytest.mark.parametrize("script_name", scripts)
def test_workflows(script_name: str):
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

    if script_name in ["nunchaku_flux1_depth.py", "nunchaku_flux1_depth_lora.py"]:
        assert clip_iqa >= 0.6
    else:
        assert clip_iqa >= 0.8
    assert lpips <= 0.24
    assert psnr >= 19
