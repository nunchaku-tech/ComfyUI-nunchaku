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

__all__ = ["get_filename_list", "get_full_path_or_raise", "folder_paths"]
