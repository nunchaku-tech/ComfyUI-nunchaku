import os

import folder_paths


def set_extra_config_model_path(extra_config_models_dir_key, models_dir_name: str):
    """
    Register an extra model directory (e.g., ``pulid``, ``insightface``, ``facexlib``) with ComfyUI's folder_paths.

    Parameters
    ----------
    extra_config_models_dir_key : str
        The key to register the model directory under.
    models_dir_name : str
        The name of the subdirectory to use for models.
    """
    models_dir_default = os.path.join(folder_paths.models_dir, models_dir_name)
    if extra_config_models_dir_key not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[extra_config_models_dir_key] = (
            [os.path.join(folder_paths.models_dir, models_dir_name)],
            folder_paths.supported_pt_extensions,
        )
    else:
        if not os.path.exists(models_dir_default):
            os.makedirs(models_dir_default, exist_ok=True)
        folder_paths.add_model_folder_path(extra_config_models_dir_key, models_dir_default, is_default=True)
