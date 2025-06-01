import os
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download, snapshot_download

from nunchaku.utils import get_precision


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data


def download_file(
    repo_id: str,
    filename: str,
    sub_folder: str,
    new_filename: str | None = None,
) -> str:
    os.makedirs(os.path.join("models", sub_folder), exist_ok=True)
    target_folder = os.path.join("models", sub_folder)
    target_file = os.path.join(target_folder, filename if new_filename is None else new_filename)
    if not os.path.exists(target_file):
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=target_folder)
        if new_filename is not None:
            os.rename(os.path.join(target_folder, filename), os.path.join(target_folder, new_filename))
    return target_file


def download_original_models():
    download_file(
        repo_id="comfyanonymous/flux_text_encoders", filename="clip_l.safetensors", sub_folder="text_encoders"
    )
    download_file(
        repo_id="comfyanonymous/flux_text_encoders", filename="t5xxl_fp16.safetensors", sub_folder="text_encoders"
    )
    download_file(repo_id="black-forest-labs/FLUX.1-dev", filename="ae.safetensors", sub_folder="vae")
    download_file(
        repo_id="black-forest-labs/FLUX.1-dev", filename="flux1-dev.safetensors", sub_folder="diffusion_models"
    )
    download_file(
        repo_id="black-forest-labs/FLUX.1-schnell", filename="flux1-schnell.safetensors", sub_folder="diffusion_models"
    )


def download_nunchaku_models():
    precision = get_precision()
    svdquant_models = [
        f"mit-han-lab/svdq-{precision}-shuttle-jaguar",
        f"mit-han-lab/svdq-{precision}-flux.1-schnell",
        f"mit-han-lab/svdq-{precision}-flux.1-dev",
        f"mit-han-lab/svdq-{precision}-flux.1-schnell",
        f"mit-han-lab/svdq-{precision}-flux.1-canny-dev",
        f"mit-han-lab/svdq-{precision}-flux.1-depth-dev",
        f"mit-han-lab/svdq-{precision}-flux.1-fill-dev",
    ]
    os.makedirs(os.path.join("models", "diffusion_models"), exist_ok=True)
    for model_path in svdquant_models:
        snapshot_download(
            model_path, local_dir=os.path.join("models", "diffusion_models", os.path.basename(model_path))
        )


def download_from_yaml():
    data = load_yaml(Path(__file__).resolve().parent.parent / "test_data" / "models.yaml")
    for model_info in data["models"]:
        repo_id = model_info["repo_id"]
        filename = model_info["filename"].format(precision=get_precision())
        sub_folder = model_info["sub_folder"]
        new_filename = model_info.get("new_filename", None)
        download_file(repo_id=repo_id, filename=filename, sub_folder=sub_folder, new_filename=new_filename)


def download_other():
    download_file(
        repo_id="Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
        filename="diffusion_pytorch_model.safetensors",
        sub_folder="controlnet",
        new_filename="controlnet-union-pro-2.0.safetensors",
    )
    download_file(
        repo_id="jasperai/Flux.1-dev-Controlnet-Upscaler",
        filename="diffusion_pytorch_model.safetensors",
        sub_folder="controlnet",
        new_filename="controlnet-upscaler.safetensors",
    )
    download_file(
        repo_id="black-forest-labs/FLUX.1-Redux-dev",
        filename="flux1-redux-dev.safetensors",
        sub_folder="style_models",
        new_filename="flux1-redux-dev.safetensors",
    )
    download_file(
        repo_id="Comfy-Org/sigclip_vision_384",
        filename="sigclip_vision_patch14_384.safetensors",
        sub_folder="clip_vision",
    )


if __name__ == "__main__":
    download_original_models()
    download_nunchaku_models()
    download_from_yaml()
    download_other()
