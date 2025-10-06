import os
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

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


def download_from_yaml():
    data = load_yaml(Path(__file__).resolve().parent.parent / "test_data" / "models.yaml")
    for model_info in data["models"]:
        repo_id = model_info["repo_id"]
        filename = model_info["filename"].format(precision=get_precision())
        sub_folder = model_info["sub_folder"]
        new_filename = model_info.get("new_filename", None)
        download_file(repo_id=repo_id, filename=filename, sub_folder=sub_folder, new_filename=new_filename)


if __name__ == "__main__":
    download_from_yaml()
