import json
from pathlib import Path

import pytest
import pytest_asyncio
from comfy.api.components.schema.prompt import Prompt
from comfy.client.embedded_comfy_client import Comfy


@pytest_asyncio.fixture(scope="module")
async def client() -> Comfy:
    async with Comfy() as client_instance:
        yield client_instance


class Case:
    def __init__(self, workflow_name: str, workflow_file: Path):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [Case()])
async def test(case: Case):
    api_file = Path(__file__).parent / "api.json"
    # Read and parse the workflow file
    workflow = json.loads(api_file.read_text(encoding="utf8"))
    prompt = Prompt.validate(workflow)
    outputs = await client.queue_prompt(prompt)
    save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")
    print(outputs[save_image_node_id]["images"][0]["abs_path"])
