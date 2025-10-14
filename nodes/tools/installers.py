"""
This module provides an advanced utility node for installing the Nunchaku Python wheel.
It operates with a 100% offline startup using a local cache file ('nunchaku_versions.json').
The online update URL for the version file is: https://nunchaku.tech/cdn/nunchaku_versions.json
The node features separate dropdowns for official and development versions. Selecting
'latest' triggers an online update of the local version lists from a centralized
CDN, ensuring a simple, reliable, and error-free user experience for everyone.
"""

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from packaging.version import parse as parse_version

# --- Configuration and Constants ---

LOCAL_VERSIONS_FILE = "nunchaku_versions.json"
NODE_DIR = Path(__file__).parent.parent.parent

# Centralized CDN URL for the versions file, accessible to all users.
NUNCHAKU_CDN_URL = "https://nunchaku.tech/cdn/nunchaku_versions.json"


# --- Network Fetching and Config Management ---


def _get_json_from_url(url: str) -> List[Dict] | Dict:
    """
    Fetch and parse JSON data from a URL.

    Parameters
    ----------
    url : str
        The URL to fetch JSON from.

    Returns
    -------
    list[dict] or dict
        Parsed JSON data, or empty dict/list on error.
    """
    try:
        headers = {"User-Agent": "ComfyUI-Nunchaku-InstallerNode"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                return json.loads(response.read())
            print(f"Warning: Received status code {response.status} from {url}")
            return {}
    except Exception as e:
        print(f"Error fetching data from {url}: {e}")
        return {}


def generate_and_save_config() -> Dict:
    """
    Fetch the centralized versions file from the CDN and update the local cache.

    Returns
    -------
    dict
        The updated configuration dictionary, or empty dict on failure.
    """
    print(f"Checking for new versions from CDN: {NUNCHAKU_CDN_URL}...")
    config = _get_json_from_url(NUNCHAKU_CDN_URL)

    if not config:
        print("Could not fetch the version list. Network might be down.")
        return {}

    try:
        file_path = NODE_DIR / LOCAL_VERSIONS_FILE
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"Successfully created/updated '{LOCAL_VERSIONS_FILE}' from CDN.")
        return config
    except Exception as e:
        print(f"Error writing '{LOCAL_VERSIONS_FILE}': {e}")
        return {}


# --- Core Helper Functions ---


def is_nunchaku_installed() -> bool:
    """
    Check if Nunchaku is currently installed.

    Returns
    -------
    bool
        True if installed, False otherwise.
    """
    try:
        importlib.metadata.version("nunchaku")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def load_version_config() -> Dict:
    """
    Load the local Nunchaku version configuration file.

    Returns
    -------
    dict
        The configuration dictionary, or empty dict if not found.
    """
    try:
        file_path = os.path.join(NODE_DIR, LOCAL_VERSIONS_FILE)
        return json.load(open(file_path, "r", encoding="utf-8")) if os.path.exists(file_path) else {}
    except Exception as e:
        print(f"Error reading or parsing '{LOCAL_VERSIONS_FILE}': {e}")
        return {}


def prepare_all_version_lists(version_config: Dict) -> Tuple[List[str], List[str]]:
    """
    Prepare dropdown lists for official and development versions.

    Parameters
    ----------
    version_config : dict
        The loaded version configuration.

    Returns
    -------
    tuple of list[str]
        (official_versions, dev_versions)
    """
    official_list = ["none"] + version_config.get("versions", [])
    dev_list = ["none"] + version_config.get("dev_versions", [])
    return official_list, dev_list


def get_system_info() -> Dict[str, str]:
    """
    Detect the current system's OS, platform tag, Python, and torch version.

    Returns
    -------
    dict
        System information for wheel selection.
    """
    os_name = platform.system().lower()
    os_key = "linux" if os_name == "linux" else "win" if os_name == "windows" else "unsupported"
    platform_tag = "linux_x86_64" if os_key == "linux" else "win_amd64" if os_key == "win" else "unsupported"
    torch_version = None
    try:
        torch_v_str = importlib.metadata.version("torch")
        v_parts = torch_v_str.split(".")
        torch_version = f"torch{v_parts[0]}.{v_parts[1]}"
    except importlib.metadata.PackageNotFoundError:
        pass

    return {
        "os": os_key,
        "platform_tag": platform_tag,
        "python_version": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "torch_version": torch_version,
    }


def get_install_backend() -> str:
    """
    Detect the available installer backend.

    Returns
    -------
    str
        "uv" if available, otherwise "pip".
    """
    try:
        subprocess.run([sys.executable, "-m", "uv", "--version"], check=True, capture_output=True)
        return "uv"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "pip"


def construct_compatible_wheel_info(
    version: str, source: str, sys_info: Dict[str, str], config: Dict
) -> Optional[Dict]:
    """
    Construct the download URL and filename for a compatible Nunchaku wheel.

    Parameters
    ----------
    version : str
        The Nunchaku version to install.
    source : str
        The source to use ("github", "huggingface", or "modelscope").
    sys_info : dict
        System information as returned by :func:`get_system_info`.
    config : dict
        The version configuration.

    Returns
    -------
    dict or None
        Dictionary with "url" and "name" keys, or None if not compatible.
    """
    url_template = config.get("url_templates", {}).get(source)
    if not url_template or sys_info["python_version"] not in config.get("supported_python", []):
        return None
    supported_torch = config.get("supported_torch", [])
    if not supported_torch:
        return None
    compatible_torch = None
    if sys_info["torch_version"] in supported_torch:
        compatible_torch = sys_info["torch_version"]
    else:
        user_torch_obj = parse_version(sys_info["torch_version"].replace("torch", ""))
        available = sorted(
            [v for v in supported_torch if parse_version(v.replace("torch", "")) <= user_torch_obj],
            key=lambda v: parse_version(v.replace("torch", "")),
            reverse=True,
        )
        if available:
            compatible_torch = available[0]
    if not compatible_torch:
        return None
    template = config.get("filename_template")
    if not template:
        return None
    filename = template.format(
        version=version,
        torch_version=compatible_torch,
        python_version=sys_info["python_version"],
        platform=sys_info["platform_tag"],
    )
    version_tag = "v" + version.replace(".dev", "dev") if "dev" in version else "v" + version
    url = url_template.format(version_tag=version_tag, filename=filename)
    return {"url": url, "name": filename}


def install_wheel(wheel_url: str, backend: str) -> str:
    """
    Download and install a wheel file using the specified backend.

    Parameters
    ----------
    wheel_url : str
        The URL of the wheel file.
    backend : str
        The installer backend ("pip" or "uv").

    Returns
    -------
    str
        The installation log output.

    Raises
    ------
    RuntimeError
        If installation fails.
    """
    command = (
        [sys.executable, "-m", "uv", "pip", "install", wheel_url]
        if backend == "uv"
        else [sys.executable, "-m", "pip", "install", wheel_url]
    )
    try:
        req = urllib.request.Request(wheel_url, method="HEAD", headers={"User-Agent": "ComfyUI-Nunchaku-InstallerNode"})
        with urllib.request.urlopen(req, timeout=15) as response:
            if response.status not in (200, 302):
                raise urllib.error.URLError(f"File not found (status: {response.status})")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
        )
        log = "".join(iter(process.stdout.readline, ""))
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, output=log)
        return log
    except Exception as e:
        raise RuntimeError(f"Installation failed for {wheel_url}. Error: {e}") from e


# --- ComfyUI Node Definition ---

VERSION_CONFIG = load_version_config()
if not VERSION_CONFIG:
    print(f"'{LOCAL_VERSIONS_FILE}' not found. Node will start in minimal mode. Use 'update node' to fetch versions.")
    OFFICIAL_VERSIONS = ["none"]
    DEV_VERSIONS = ["none"]
else:
    OFFICIAL_VERSIONS, DEV_VERSIONS = prepare_all_version_lists(VERSION_CONFIG)


class NunchakuWheelInstaller:
    """
    This node allows users to install or uninstall the Nunchaku Python package
    directly from the ComfyUI interface. It supports both official and development
    versions, and can update its version list from online sources.
    """

    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Installer"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Node change detection stub.

        Returns
        -------
        float
            Always returns NaN (no change detection).
        """
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        """
        Returns the input types required for the node.

        Returns
        -------
        dict
            Dictionary specifying required inputs: version, dev_version, and mode.
        """
        inputs = {
            "required": {
                "version": (
                    OFFICIAL_VERSIONS,
                    {
                        "tooltip": (
                            "Official Nunchaku version to install. Use 'update node' mode to get the latest list."
                            "If dev_version is also selected, it will take priority."
                        ),
                    },
                ),
                "dev_version": (
                    DEV_VERSIONS,
                    {
                        "default": "none",
                        "tooltip": (
                            "Development Nunchaku version to install. Use 'update node' mode to get the latest list."
                            "This option has priority over the official version."
                        ),
                    },
                ),
                "mode": (
                    ["install", "uninstall", "update node"],
                    {"default": "install", "tooltip": "Install, uninstall, or update the version list."},
                ),
            }
        }

        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)

    def run(self, version: str, mode: str, dev_version: str = "none"):
        """
        Execute the install or uninstall operation.

        Parameters
        ----------
        version : str
            The official Nunchaku version to install.
        mode : str
            "install", "uninstall", or "update node".
        dev_version : str, optional
            The development Nunchaku version to install, by default "none".

        Returns
        -------
        tuple[str]
            Status message as a tuple.
        """
        global VERSION_CONFIG, OFFICIAL_VERSIONS, DEV_VERSIONS

        try:
            if mode == "uninstall":
                if is_nunchaku_installed():
                    subprocess.run(
                        [sys.executable, "-m", "pip", "uninstall", "nunchaku", "-y"], check=True, capture_output=True
                    )
                    return (
                        "✅ Existing Nunchaku uninstalled.\n**Please restart ComfyUI completely.**\nThen, run again to install.",
                    )
                else:
                    return ("Nunchaku is not installed. Nothing to do.",)

            elif mode == "update node":
                updated_config = generate_and_save_config()
                if updated_config:
                    VERSION_CONFIG = updated_config
                    OFFICIAL_VERSIONS, DEV_VERSIONS = prepare_all_version_lists(updated_config)
                    return (
                        "✅ Version list updated.\nPlease refresh the web UI (press 'r') to see the new version list.",
                    )
                else:
                    return ("❌ Update failed. Check internet connection and logs.",)

            else:  # install mode
                if not VERSION_CONFIG:
                    raise RuntimeError(
                        "Local version list not found. Please run in 'update node' mode first to fetch versions."
                    )

                # Step 2: Determine the final version to install
                final_version = None
                sources_to_try = []
                if dev_version != "none":
                    final_version = dev_version
                    sources_to_try = ["github"]
                elif version != "none":
                    final_version = version
                    sources_to_try = ["modelscope", "huggingface", "github"]

                if not final_version:
                    return ("No version selected. Please choose a version to install or update the node.",)

                # Step 3: Find compatible wheel and install
                sys_info = get_system_info()
                if sys_info["os"] == "unsupported":
                    raise RuntimeError(f"Unsupported OS: {platform.system()}")

                backend = get_install_backend()
                print(f"Using installer backend: {backend}")

                wheel_to_install = None
                last_error = ""
                for source in sources_to_try:
                    print(f"\n--- Trying source: {source} for version {final_version} ---")
                    try:
                        wheel_info = construct_compatible_wheel_info(final_version, source, sys_info, VERSION_CONFIG)
                        if not wheel_info:
                            last_error = f"No compatible wheel found on '{source}' for your system."
                            print(last_error)
                            continue

                        print(f"Attempting to install: {wheel_info['name']}")
                        final_log = install_wheel(wheel_info["url"], backend)
                        wheel_to_install = wheel_info
                        print(f"--- Successfully installed from {source} ---")
                        break
                    except Exception as e:
                        print(f"Failed to install from {source}: {e}. Trying next source...")
                        last_error = str(e)

                if not wheel_to_install:
                    raise RuntimeError(f"Failed to install from all available sources.\n\nLast error: {last_error}")

                return (
                    f"✅ Success! Installed: {wheel_to_install['name']}\n\nRestart ComfyUI completely to apply changes.\n\n--- LOG ---\n{final_log}",
                )

        except Exception as e:
            return (f"❌ ERROR:\n{e}",)


NODE_CLASS_MAPPINGS = {"NunchakuWheelInstaller": NunchakuWheelInstaller}
NODE_DISPLAY_NAME_MAPPINGS = {"NunchakuWheelInstaller": "Nunchaku Installer"}
