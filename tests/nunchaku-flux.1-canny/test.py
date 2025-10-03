import json
from pathlib import Path

import pytest
import pytest_asyncio
from comfy.api.components.schema.prompt import Prompt
from comfy.client.embedded_comfy_client import Comfy
from comfy.model_downloader import KNOWN_LORAS, add_known_models
from comfy.model_downloader_types import CivitFile, HuggingFile

from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio
from torchmetrics.multimodal import CLIPImageQualityAssessment
from diffusers.utils import load_image
import torch
from PIL import Image
import numpy as np
from nunchaku.utils import get_precision, is_turing


precision = get_precision()
torch_dtype = torch.float16 if is_turing() else torch.bfloat16
dtype_str = "fp16" if torch_dtype == torch.float16 else "bf16"


@pytest_asyncio.fixture(scope="module")
async def client() -> Comfy:
    async with Comfy() as client_instance:
        yield client_instance


class Case:
    def __init__(
        self,
        ref_image_url: str = "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/ComfyUI-nunchaku/ref_images/int4/nunchaku-flux1-canny.png",
        expected_clip_iqa: dict[str, float] = {},
        expected_lpips: dict[str, float] = {},
        expected_psnr: dict[str, float] = {},
    ):
        self.ref_image_url = ref_image_url
        self.expected_clip_iqa = expected_clip_iqa
        self.expected_lpips = expected_lpips
        self.expected_psnr = expected_psnr


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [Case()])
async def test(case: Case):
    api_file = Path(__file__).parent / "api.json"
    # Read and parse the workflow file
    workflow = json.loads(api_file.read_text(encoding="utf8"))
    prompt = Prompt.validate(workflow)
    outputs = await client.queue_prompt(prompt)
    save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")
    path = outputs[save_image_node_id]["images"][0]["abs_path"]
    print(path)

    # clip_iqa metric
    metric = CLIPImageQualityAssessment(model_name_or_path="openai/clip-vit-large-patch14").to("cuda")
    image = Image.open(path).convert("RGB")
    gen_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(torch.float32).unsqueeze(0).to("cuda")
    clip_iqa = metric(gen_tensor).item()
    print(f"CLIP-IQA: {clip_iqa}")

    ref_image = load_image(case.ref_image_url).convert("RGB")
    metric = LearnedPerceptualImagePatchSimilarity().to("cuda")
    ref_tensor = torch.from_numpy(np.array(ref_image)).permute(2, 0, 1).to(torch.float32)
    ref_tensor = ref_tensor.unsqueeze(0).to("cuda")
    lpips = metric(gen_tensor / 255, ref_tensor / 255).item()
    print(f"LPIPS: {lpips}")

    metric = PeakSignalNoiseRatio(data_range=(0, 255)).cuda()
    psnr = metric(gen_tensor, ref_tensor).item()
    print(f"PSNR: {psnr}")

    assert clip_iqa >= case.expected_clip_iqa[f"{precision}-{dtype_str}"] * 0.85
    assert lpips <= case.expected_lpips[f"{precision}-{dtype_str}"] * 1.15
    assert psnr >= case.expected_psnr[f"{precision}-{dtype_str}"] * 0.85
