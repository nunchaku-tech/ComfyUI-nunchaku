import sys

folder_paths = None
try:
    from comfy.cmd import folder_paths
    from comfy.model_downloader import get_filename_list, get_full_path_or_raise
except (ImportError, ModuleNotFoundError):
    folder_paths = sys.modules["folder_paths"]
    from folder_paths import get_filename_list, get_full_path_or_raise

get_filename_list = get_filename_list
get_full_path_or_raise = get_full_path_or_raise


def get_nunchaku_model_list():
    files = get_filename_list("diffusion_models")
    for f in get_filename_list("checkpoints"):
        if f not in files:
            files.append(f)
    return files


def get_nunchaku_model_full_path(filename):
    for folder in ("diffusion_models", "checkpoints"):
        try:
            return get_full_path_or_raise(folder, filename)
        except Exception:
            continue
    raise FileNotFoundError(f"Model {filename} not found in diffusion_models or checkpoints")


__all__ = ["get_filename_list", "get_full_path_or_raise", "folder_paths", "get_nunchaku_model_list", "get_nunchaku_model_full_path"]
