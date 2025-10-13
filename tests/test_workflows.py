import json
import logging
from pathlib import Path

import pytest
import pytest_asyncio
import torch
from comfy.api.components.schema.prompt import Prompt
from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import Comfy

from nunchaku.utils import get_precision, is_turing

from .case import Case, cases, ids
from .utils import compute_metrics, prepare_inputs, prepare_models, set_nested_value

logger = logging.getLogger(__name__)

precision = get_precision()
torch_dtype = torch.float16 if is_turing() else torch.bfloat16
dtype_str = "fp16" if torch_dtype == torch.float16 else "bf16"


@pytest_asyncio.fixture(scope="module", autouse=False)
async def inputs(tmp_path_factory):
    config = Configuration()
    input_path = config.input_directory = str(tmp_path_factory.mktemp("nunchaku_input"))
    prepare_inputs(input_path)
    yield config


@pytest_asyncio.fixture(scope="function", autouse=False)
async def client(tmp_path_factory, inputs) -> Comfy:
    async with Comfy(inputs) as client_instance:
        prepare_models()
        yield client_instance


@pytest.mark.asyncio
@pytest.mark.parametrize("case", cases, ids=ids)
async def test(case: Case, client: Comfy):
    api_file = Path(__file__).parent / "workflows" / case.workflow_name / "api.json"
    # Read and parse the workflow file
    workflow = json.loads(api_file.read_text(encoding="utf8"))
    for key, value in case.inputs.items():
        set_nested_value(workflow, key, value)
    prompt = Prompt.validate(workflow)
    outputs = await client.queue_prompt(prompt)
    save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")
    path = outputs[save_image_node_id]["images"][0]["abs_path"]
    logger.info("Generated image path: %s", path)

    clip_iqa, lpips, psnr = compute_metrics(path, case.ref_image_url)

    assert clip_iqa >= min(case.expected_clip_iqa[f"{precision}-{dtype_str}"], 1) * 0.9
    assert lpips <= max(case.expected_lpips[f"{precision}-{dtype_str}"], 0.1) * 1.15
    assert psnr >= min(case.expected_psnr[f"{precision}-{dtype_str}"], 24) * 0.85
