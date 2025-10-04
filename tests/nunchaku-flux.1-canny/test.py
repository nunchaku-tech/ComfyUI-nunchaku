import asyncio
import json
from pathlib import Path
from typing import Any

import torch
from comfy.api.components.schema.prompt import Prompt
from comfy.client.embedded_comfy_client import Comfy
from comfy.model_downloader import KNOWN_UNET_MODELS, add_known_models
from comfy.model_downloader_types import HuggingFile

from nunchaku.utils import get_precision, is_turing

# from ..common import compute_metrics, prepare_models, set_nested_value


def prepare_models():
    add_known_models(
        "text_encoders",
        None,
        HuggingFile(repo_id="nunchaku-tech/nunchaku-t5", filename="awq-int4-flux.1-t5xxl.safetensors"),
    )

    add_known_models(
        "diffusion_models",
        None,
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
    )


def set_nested_value(d: dict, keys: tuple, value: Any):
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


precision = get_precision()
torch_dtype = torch.float16 if is_turing() else torch.bfloat16
dtype_str = "fp16" if torch_dtype == torch.float16 else "bf16"
prepare_models()


class Case:
    def __init__(
        self,
        ref_image_url: str = "https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/ComfyUI-nunchaku/ref_images/int4/nunchaku-flux1-canny.png",
        expected_clip_iqa: dict[str, float] = {},
        expected_lpips: dict[str, float] = {},
        expected_psnr: dict[str, float] = {},
        inputs: dict[tuple, Any] = {},
    ):
        self.ref_image_url = ref_image_url
        self.expected_clip_iqa = expected_clip_iqa
        self.expected_lpips = expected_lpips
        self.expected_psnr = expected_psnr
        self.inputs = inputs


# @pytest_asyncio.fixture(scope="module")
# async def client() -> Comfy:
#     async with Comfy() as client_instance:
#         yield client_instance


# @pytest.mark.asyncio
# @pytest.mark.parametrize(
#     "case",
#     [
#         Case(
#             expected_clip_iqa={"int4-bf16": 0.5},
#             expected_lpips={"int4-bf16": 0.5},
#             expected_psnr={"int4-bf16": 15},
#             inputs={("39", "inputs", "model_path"): f"svdq-{precision}-r32-flux.1-canny-dev.safetensors"},
#         )
#     ],
# )
# async def test(case: Case):
#     api_file = Path(__file__).parent / "api.json"
#     # Read and parse the workflow file
#     workflow = json.loads(api_file.read_text(encoding="utf8"))
#     for key, value in case.inputs.items():
#         set_nested_value(workflow, key, value)
#     prompt = Prompt.validate(workflow)
#     outputs = await client.queue_prompt(prompt)
#     save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")
#     path = outputs[save_image_node_id]["images"][0]["abs_path"]
#     print(path)

#     clip_iqa, lpips, psnr = compute_metrics(path, case.ref_image_url)

#     assert clip_iqa >= case.expected_clip_iqa[f"{precision}-{dtype_str}"] * 0.85
#     assert lpips <= case.expected_lpips[f"{precision}-{dtype_str}"] * 1.15
#     assert psnr >= case.expected_psnr[f"{precision}-{dtype_str}"] * 0.85

case = Case(
    inputs={("39", "inputs", "model_path"): f"svdq-{precision}_r32-flux.1-canny-dev.safetensors"},
)


async def main():
    print(KNOWN_UNET_MODELS)
    api_file = Path(__file__).parent / "api.json"
    # Read and parse the workflow file
    workflow = json.loads(api_file.read_text(encoding="utf8"))
    for key, value in case.inputs.items():
        set_nested_value(workflow, key, value)
    prompt = Prompt.validate(workflow)
    async with Comfy() as client:
        outputs = await client.queue_prompt(prompt)
        save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")
        print(outputs)
        path = outputs[save_image_node_id]["images"][0]["abs_path"]
        print(path)


if __name__ == "__main__":
    # Since our main function is async, it must be run as async too.
    asyncio.run(main())
