import importlib.resources
import json
import shutil
from pathlib import Path
import pytest

from comfy.api.components.schema.prompt import Prompt
from comfy.client.embedded_comfy_client import Comfy
from comfy.cmd import folder_paths
from comfy.model_downloader import add_known_models, KNOWN_LORAS
from comfy.model_downloader_types import CivitFile, HuggingFile
from comfy_extras.nodes.nodes_audio import TorchAudioNotFoundError

@pytest.fixture(scope="module", autouse=False)
async def client(tmp_path_factory) -> Comfy:
    async with Comfy() as client:
        yield client


def _prepare_for_custom_workflows() -> dict[str, Path]:
    """Prepare and discover custom workflow files"""
    add_known_models("loras", KNOWN_LORAS, CivitFile(13941, 16576, "epi_noiseoffset2.safetensors"))
    add_known_models("checkpoints", HuggingFile("autismanon/modeldump", "cardosAnime_v20.safetensors"))

    # Get the custom_workflows directory path
    custom_workflows_dir = Path(__file__).parent / "workflows_api"
    
    # Return dict of workflow files
    workflow_files = {}
    if custom_workflows_dir.exists():
        for file_path in custom_workflows_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith(".json"):
                workflow_files[file_path.name] = file_path
    
    return workflow_files


def _prepare_test_image() -> None:
    """Copy a test image to the input directory for workflows that need it"""
    # Get the input directory
    input_dir = Path(folder_paths.get_input_directory())
    input_dir.mkdir(exist_ok=True)
    
    # Source test image from images_input directory - try PNG first, fallback to JPG
    source_png = Path(__file__).parent / "images_input" / "input.png"
    source_jpg = Path(__file__).parent / "images_input" / "moon.jpg"
    target_image = input_dir / "input.png"
    
    # Choose the source image (prefer PNG)
    source_image = source_png if source_png.exists() else source_jpg
    
    # Copy the test image if it doesn't exist or if source is newer
    if source_image.exists() and (not target_image.exists() or 
                                 source_image.stat().st_mtime > target_image.stat().st_mtime):
        shutil.copy2(source_image, target_image)


def _workflow_needs_image_input(workflow: dict) -> bool:
    """Check if workflow contains LoadImage nodes that need input images"""
    for node_data in workflow.values():
        if isinstance(node_data, dict) and node_data.get("class_type") == "LoadImage":
            return True
    return False


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_name, workflow_file", _prepare_for_custom_workflows().items())
async def test_custom_workflow(workflow_name: str, workflow_file: Path, has_gpu: bool, client: Comfy):
    """Test custom workflows from the custom_workflows directory"""
    if not has_gpu:
        pytest.skip("requires gpu")

    # Read and parse the workflow file
    workflow = json.loads(workflow_file.read_text(encoding="utf8"))

    # Check if this workflow needs image input and prepare test image if needed
    if _workflow_needs_image_input(workflow):
        _prepare_test_image()

    # Validate the workflow as a prompt (same as test_workflows.py)
    prompt = Prompt.validate(workflow)
    
    # Execute the workflow
    try:
        outputs = await client.queue_prompt(prompt)
    except TorchAudioNotFoundError:
        pytest.skip("requires torchaudio")
    except Exception as e:
        # Skip if model files are missing or other expected errors
        error_msg = str(e).lower()
        if any(skip_phrase in error_msg for skip_phrase in [
            "model file not found", "checkpoint", "safetensors", "model not found",
            "no such file", "file not found", "missing model", "could not load model"
        ]):
            pytest.skip(f"Missing model files: {e}")
        else:
            # Re-raise unexpected errors
            raise

    # Validate outputs based on the node types present in the workflow
    if any(v.class_type == "SaveImage" for v in prompt.values()):
        save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")
        assert outputs[save_image_node_id]["images"][0]["abs_path"] is not None
    elif any(v.class_type == "SaveAudio" for v in prompt.values()):
        save_audio_node_id = next(key for key in prompt if prompt[key].class_type == "SaveAudio")
        assert outputs[save_audio_node_id]["audio"][0]["filename"] is not None
    elif any(v.class_type == "PreviewString" for v in prompt.values()):
        preview_string_node_id = next(key for key in prompt if prompt[key].class_type == "PreviewString")
        output_str = outputs[preview_string_node_id]["string"][0]
        assert output_str is not None
        assert len(output_str) > 0
    else:
        # For workflows without specific output nodes, just ensure we got some output
        assert outputs is not None
        assert len(outputs) > 0
