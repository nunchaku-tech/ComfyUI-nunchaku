"""
This module provides a utility node for merging Nunchaku model folders (deprecated format)
into a single safetensors file.
"""

from pathlib import Path

import folder_paths
from safetensors.torch import save_file

from nunchaku.merge_safetensors import merge_safetensors


class NunchakuModelMerger:
    """
    Node for merging a Nunchaku FLUX.1 model folder into a single safetensors file.

    This node scans available model folders, merges the selected folder using
    `nunchaku.merge_safetensors.merge_safetensors`, and saves the result as a safetensors file.

    Attributes
    ----------
    RETURN_TYPES : tuple of str
        The return types of the node ("STRING",).
    RETURN_NAMES : tuple of str
        The names of the returned values ("status",).
    FUNCTION : str
        The function to execute ("run").
    CATEGORY : str
        The node category ("Nunchaku").
    TITLE : str
        The display title of the node.
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Returns the input types required for the node.

        Returns
        -------
        dict
            Dictionary specifying required inputs: model_folder and save_name.
        """
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
    TITLE = "Nunchaku Model Merger"

    def run(self, model_folder: str, save_name: str):
        """
        Merge the specified Nunchaku model folder and save as a safetensors file.

        Parameters
        ----------
        model_folder : str
            Name of the Nunchaku FLUX.1 model folder to merge.
        save_name : str
            Filename to save the merged model as.

        Returns
        -------
        tuple of str
            Status message indicating the result of the merge operation.
        """
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
