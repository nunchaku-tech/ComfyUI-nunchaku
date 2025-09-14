"""
This module provides an advanced utility node for installing the Nunchaku Python wheel.
It operates with a 100% offline startup using a local cache file ('nunchaku_versions.json').
The node features separate dropdowns for official and development versions. Selecting
'latest' triggers an online update of the local version lists before
installing, ensuring a simple, reliable, and error-free user experience.
"""

import importlib.metadata
import json
import os
import platform
import re
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

GITHUB_API_URL = "https://api.github.com/repos/nunchaku-tech/nunchaku"
HF_API_URL = "https://huggingface.co/api/models/nunchaku-tech/nunchaku/tree/main"
MODEL_SCOPE_API_URL = (
    "https://modelscope.cn/api/v1/models/nunchaku-tech/nunchaku/repo/files?Revision=master&PageSize=500"
)

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


def get_nunchaku_versions_from_sources() -> Tuple[set, set]:
    """
    Fetch all unique Nunchaku version numbers from all sources.

    Returns
    -------
    tuple of set
        (official_versions, dev_versions)
    """
    official_tags, dev_tags = set(), set()
    wheel_regex = re.compile(r"nunchaku-([^-+]+)")

    # GitHub (Official + Dev) - Parsing from asset filenames for accuracy
    releases = _get_json_from_url(f"{GITHUB_API_URL}/releases")
    if isinstance(releases, list):
        for release in releases:
            for asset in release.get("assets", []):
                filename = asset.get("name", "")
                if filename.endswith(".whl"):
                    match = wheel_regex.search(filename)
                    if match:
                        version_str = match.group(1)
                        if "dev" in version_str:
                            dev_tags.add(version_str)
                        else:
                            official_tags.add(version_str)
                        break

    # Hugging Face / ModelScope
    sources = {"huggingface": (HF_API_URL, "path"), "modelscope": (MODEL_SCOPE_API_URL, "Name")}
    for source_name, (url, path_key) in sources.items():
        api_response = _get_json_from_url(url)
        if not api_response:
            print(f"Could not get response from {source_name}, skipping.")
            continue

        file_list = []
        if source_name == "modelscope" and isinstance(api_response, dict):
            file_list = api_response.get("Data", {}).get("Files", [])
        elif source_name == "huggingface" and isinstance(api_response, list):
            file_list = api_response

        for file_info in file_list:
            filename = file_info.get(path_key)
            if filename and filename.endswith(".whl"):
                match = wheel_regex.search(filename)
                if match:
                    version_str = match.group(1)
                    if "dev" in version_str:
                        dev_tags.add(version_str)
                    else:
                        official_tags.add(version_str)

    return official_tags, dev_tags


def generate_and_save_config() -> Dict:
    """
    Fetch all available versions and update the local ``nunchaku_versions.json`` file.

    Returns
    -------
    dict
        The updated configuration dictionary, or empty dict on failure.
    """
    print("Checking for new versions from internet sources...")
    official_versions, dev_versions = get_nunchaku_versions_from_sources()

    if not official_versions and not dev_versions:
        print("Could not fetch any version information. Network might be down.")
        return {}

    config = {
        "versions": sorted(list(official_versions), key=parse_version, reverse=True),
        "dev_versions": sorted(list(dev_versions), key=parse_version, reverse=True),
        "supported_torch": ["torch2.5", "torch2.6", "torch2.7", "torch2.8", "torch2.9"],
        "supported_python": ["cp310", "cp311", "cp312", "cp313"],
        "filename_template": "nunchaku-{version}+{torch_version}-{python_version}-{python_version}-{platform}.whl",
        "url_templates": {
            "github": "https://github.com/nunchaku-tech/nunchaku/releases/download/{version_tag}/{filename}",
            "huggingface": "https://huggingface.co/nunchaku-tech/nunchaku/resolve/main/{filename}",
            "modelscope": "https://modelscope.cn/models/nunchaku-tech/nunchaku/resolve/master/{filename}",
        },
    }

    try:
        file_path = NODE_DIR / LOCAL_VERSIONS_FILE
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"Successfully created/updated '{LOCAL_VERSIONS_FILE}'")
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
    official_list = ["none", "latest"] + version_config.get("versions", [])
    dev_list = ["none", "latest"] + version_config.get("dev_versions", [])
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
    print(f"'{LOCAL_VERSIONS_FILE}' not found. Node will start in minimal mode.")
    OFFICIAL_VERSIONS = ["latest"]
    DEV_VERSIONS = ["none", "latest"]
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
        return {
            "required": {
                "version": (
                    OFFICIAL_VERSIONS,
                    {
                        "tooltip": (
                            "Official Nunchaku version to install. Use 'lastest' to pull the latest version list. "
                            "When you specify both version and dev_version, the dev_version will be used."
                        ),
                    },
                ),
                "dev_version": (
                    DEV_VERSIONS,
                    {
                        "default": "none",
                        "tooltip": (
                            "Development Nunchaku version to install. Use 'lastest' to pull the latest version list. "
                            "When you specify both version and dev_version, the dev_version will be used."
                        ),
                    },
                ),
                "mode": (
                    ["install", "uninstall"],
                    {"default": "install", "tooltip": "Install or uninstall Nunchaku."},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)

    def run(self, version: str, dev_version: str, mode: str):
        """
        Execute the install or uninstall operation.

        Parameters
        ----------
        version : str
            The official Nunchaku version to install.
        dev_version : str
            The development Nunchaku version to install.
        mode : str
            "install" or "uninstall".

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
                current_config = VERSION_CONFIG
                # Step 1: Check if an online update is needed
                if version == "latest" or dev_version == "latest":
                    updated_config = generate_and_save_config()
                    if updated_config:
                        VERSION_CONFIG = updated_config
                        OFFICIAL_VERSIONS, DEV_VERSIONS = prepare_all_version_lists(updated_config)
                        current_config = updated_config
                        print("Version lists updated. Please restart or refresh web UI to see changes.")
                    elif not current_config:
                        raise RuntimeError("Update check failed and no local cache exists. Check internet connection.")

                # Step 2: Determine the final version to install
                if dev_version not in ["none", "latest"]:
                    final_version = dev_version
                    sources_to_try = ["github"]
                elif dev_version == "latest":
                    if not current_config.get("dev_versions"):
                        raise RuntimeError("No dev versions found. Run with 'latest' first or check GitHub.")
                    final_version = current_config["dev_versions"][0]
                    sources_to_try = ["github"]
                else:  # Official version
                    if not current_config.get("versions"):
                        raise RuntimeError("No official versions found. Run with 'latest' to fetch them.")
                    final_version = current_config["versions"][0] if version == "latest" else version
                    sources_to_try = ["modelscope", "huggingface", "github"]

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
                        wheel_info = construct_compatible_wheel_info(final_version, source, sys_info, current_config)
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
