from pathlib import Path

import folder_paths
from safetensors.torch import save_file

from nunchaku.merge_safetensors import merge_safetensors


class NunchakuModelMerger:
    @classmethod
    def INPUT_TYPES(s):
        prefixes = folder_paths.folder_names_and_paths["diffusion_models"][0]
        local_folders = set()
        for prefix in prefixes:
            prefix = Path(prefix)
            if prefix.exists() and prefix.is_dir():
                local_folders_ = [
                    folder.name for folder in prefix.iterdir() if folder.is_dir() and not folder.name.startswith(".")
                ]
                local_folders.update(local_folders_)
        model_paths = sorted(list(local_folders))
        return {
            "required": {
                "model_folder": (model_paths, {"tooltip": "Nunchaku FLUX.1 model folder."}),
                "save_name": ("STRING", {"tooltip": "Filename to save the merged model as."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"
    CATEGORY = "Nunchaku"
    TITLE = "Merge Nunchaku Model"

    def run(self, model_folder: str, save_name: str):
        prefixes = folder_paths.folder_names_and_paths["diffusion_models"][0]
        model_path = None
        for prefix in prefixes:
            prefix = Path(prefix)
            model_path = prefix / model_folder
            if model_path.exists() and model_path.is_dir():
                break

        comfy_config_path = None
        if not (model_path / "comfy_config.json").exists():
            default_config_root = Path(__file__).parent.parent / "models" / "configs"
            config_name = model_path.name.replace("svdq-int4-", "").replace("svdq-fp4-", "")
            comfy_config_path = default_config_root / f"{config_name}.json"

        state_dict, metadata = merge_safetensors(
            pretrained_model_name_or_path=model_path, comfy_config_path=comfy_config_path
        )
        save_name = save_name.strip()
        if not save_name.endswith((".safetensors", ".sft")):
            save_name += ".safetensors"
        save_path = model_path.parent / save_name
        save_file(state_dict, save_path, metadata=metadata)
        return (f"✅ Merge `{model_path}` to `{save_path}`.",)
