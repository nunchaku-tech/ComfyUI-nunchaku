import logging
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from comfy.model_downloader import add_known_models
from comfy.model_downloader_types import HuggingFile
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from PIL import Image
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio
from torchmetrics.multimodal import CLIPImageQualityAssessment

from nunchaku.utils import get_precision

logger = logging.getLogger(__name__)


def compute_metrics(gen_image_path: str, ref_image_path: str) -> tuple[float, float, float]:
    # clip_iqa metric
    metric = CLIPImageQualityAssessment(model_name_or_path="openai/clip-vit-large-patch14").to("cuda")
    image = Image.open(gen_image_path).convert("RGB")
    gen_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(torch.float32).unsqueeze(0).to("cuda")
    clip_iqa = metric(gen_tensor).item()
    logger.info(f"CLIP-IQA: {clip_iqa}")

    ref_image = load_image(ref_image_path).convert("RGB")
    metric = LearnedPerceptualImagePatchSimilarity().to("cuda")
    ref_tensor = torch.from_numpy(np.array(ref_image)).permute(2, 0, 1).to(torch.float32)
    ref_tensor = ref_tensor.unsqueeze(0).to("cuda")
    lpips = metric(gen_tensor / 255, ref_tensor / 255).item()
    logger.info(f"LPIPS: {lpips}")

    metric = PeakSignalNoiseRatio(data_range=(0, 255)).cuda()
    psnr = metric(gen_tensor, ref_tensor).item()
    logger.info(f"PSNR: {psnr}")
    return clip_iqa, lpips, psnr


def prepare_models():
    """Reads models.yaml and adds them to known models."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_yaml_path = os.path.join(current_dir, "..", "test_data", "models.yaml")
    with open(models_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    precision = get_precision()

    # other model files
    for model in data["models"]:
        hugging_files = []
        filename = model["filename"]
        sub_folder = model["sub_folder"]
        if "{precision}" in filename:
            filename = filename.format(precision=precision)

        new_filename = model.get("new_filename")

        if new_filename:
            hugging_files.append(
                HuggingFile(repo_id=model["repo_id"], filename=filename, save_with_filename=new_filename)
            )
        else:
            hugging_files.append(HuggingFile(repo_id=model["repo_id"], filename=filename))

        add_known_models(sub_folder, *hugging_files)


def set_nested_value(d: dict, key: str, value: Any):
    keys = key.split(",")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def prepare_inputs(inputs_dir: str):
    """Downloads test input files from inputs.yaml to the specified directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    inputs_yaml_path = os.path.join(current_dir, "..", "test_data", "inputs.yaml")
    with open(inputs_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    inputs_dir = Path(inputs_dir)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "nunchaku-tech/test-data"

    with tempfile.TemporaryDirectory() as tmpdir:
        for item in data["inputs"]:
            for file in item["files"]:
                dst_path = inputs_dir / file

                if dst_path.exists(follow_symlinks=True):
                    continue

                src_path = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=f"inputs/{file}",
                )
                src_path = Path(src_path)

                try:
                    os.symlink(src_path, dst_path)
                except Exception:
                    shutil.copyfile(src_path, dst_path)
