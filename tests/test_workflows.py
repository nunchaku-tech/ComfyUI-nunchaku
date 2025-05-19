import gc
import os
import subprocess

import numpy as np
import pytest
import torch
from PIL import Image
from torchmetrics.multimodal import CLIPImageQualityAssessment

script_dir = os.path.join(os.path.dirname(__file__), "scripts")
scripts = [f for f in os.listdir(script_dir) if f.endswith(".py")]


@pytest.mark.parametrize("script_name", scripts)
def test_workflows(script_name: str) -> float:
    gc.collect()
    torch.cuda.empty_cache()
    script_path = os.path.join(script_dir, script_name)

    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    print(f"Running {script_path} -> Return code: {result.returncode}")
    assert result.returncode == 0, f"{script_path} failed with code {result.returncode}"

    path = open("image_path.txt", "r").read().strip()
    metric = CLIPImageQualityAssessment(model_name_or_path="openai/clip-vit-large-patch14").to("cuda")
    image = Image.open(path).convert("RGB")
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(torch.float32)
    clip_iqa = metric(tensor.unsqueeze(0)).item()
    print(f"CLIP-IQA: {clip_iqa}")
    if script_name in ["nunchaku_flux1_depth.py"]:
        assert clip_iqa >= 0.6
    else:
        assert clip_iqa >= 0.8
