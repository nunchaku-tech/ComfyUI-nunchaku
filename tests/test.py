import json
from pathlib import Path

import pytest
import pytest_asyncio
import torch
from comfy.api.components.schema.prompt import Prompt
from comfy.client.embedded_comfy_client import Comfy

from nunchaku.utils import get_precision, is_turing

from .case import Case, cases, ids
from .utils import compute_metrics, prepare_models, set_nested_value

precision = get_precision()
torch_dtype = torch.float16 if is_turing() else torch.bfloat16
dtype_str = "fp16" if torch_dtype == torch.float16 else "bf16"
prepare_models()


@pytest_asyncio.fixture(scope="module")
async def client() -> Comfy:
    async with Comfy() as client_instance:
        yield client_instance


@pytest.mark.asyncio
@pytest.mark.parametrize("case", cases, ids=ids)
async def test(case: Case, client: Comfy):
    api_file = Path(__file__).parent / case.workflow_name / "api.json"
    # Read and parse the workflow file
    workflow = json.loads(api_file.read_text(encoding="utf8"))
    for key, value in case.inputs.items():
        set_nested_value(workflow, key, value)
    prompt = Prompt.validate(workflow)
    outputs = await client.queue_prompt(prompt)
    save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")
    path = outputs[save_image_node_id]["images"][0]["abs_path"]
    print(path)

    clip_iqa, lpips, psnr = compute_metrics(path, case.ref_image_url)

    assert clip_iqa >= case.expected_clip_iqa[f"{precision}-{dtype_str}"] * 0.85
    assert lpips <= case.expected_lpips[f"{precision}-{dtype_str}"] * 1.15
    assert psnr >= case.expected_psnr[f"{precision}-{dtype_str}"] * 0.85
