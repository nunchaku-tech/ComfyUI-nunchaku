"""
This module provides an advanced utility node for installing the Nunchaku Python wheel.
It dynamically fetches available versions from GitHub, Hugging Face, and ModelScope,
allows the user to select an installer backend (pip or uv), and automatically finds
the most compatible wheel. The installation status is displayed directly on the node UI.
"""

import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

from packaging.version import parse as parse_version

# --- Helper Functions ---

# CHANGE: Defined separate, direct API URLs for each source.
GITHUB_API_URL = "https://api.github.com/repos/nunchaku-tech/nunchaku"
HF_API_URL = "https://huggingface.co/api/models/mit-han-lab/nunchaku/tree/main"
# CHANGE: Added the direct ModelScope API URL, replacing the previous mirror logic.
MODEL_SCOPE_API_URL = (
    "https://modelscope.cn/api/v1/models/nunchaku-tech/nunchaku/repo/files?Revision=master&PageSize=500"
)


def is_nunchaku_installed() -> bool:
    try:
        importlib.metadata.version("nunchaku")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _get_json_from_url(url: str) -> List[Dict] | Dict:
    try:
        headers = {"User-Agent": "ComfyUI-Nunchaku-InstallerNode"}
        req = urllib.request.Request(url, headers=headers)
        # Added a timeout for network robustness.
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                return json.loads(response.read())
            print(f"Warning: Received status code {response.status} from {url}")
            return []
    except Exception as e:
        print(f"Error fetching data from {url}: {e}")
        return []


def get_nunchaku_releases_from_github() -> List[Dict]:
    releases = _get_json_from_url(f"{GITHUB_API_URL}/releases")
    if isinstance(releases, list):
        for release in releases:
            release["source"] = "github"
        return releases
    return []


# CHANGE: Uses a more robust regex to extract the version.
def _parse_wheels_from_file_list(file_list: List[Dict], source_name: str, path_key: str, url_prefix: str) -> List[Dict]:
    releases = {}

    # This new regex `nunchaku-([^-+]+)` stops at the first `-` or `+`,
    # making it more reliable for various wheel filename formats.
    wheel_regex = re.compile(r"nunchaku-([^-+]+)")

    for file_info in file_list:
        filename = file_info.get(path_key)
        if filename and filename.endswith(".whl"):
            match = wheel_regex.search(filename)
            if match:
                version_str = match.group(1)
                tag_name = f"v{version_str}"
                if tag_name not in releases:
                    releases[tag_name] = {
                        "tag_name": tag_name,
                        "name": f"Release {tag_name}",
                        "assets": [],
                        "source": source_name,
                    }
                releases[tag_name]["assets"].append(
                    {"name": filename, "browser_download_url": f"{url_prefix}{filename}"}
                )

    return list(releases.values())


def get_nunchaku_releases_from_huggingface() -> List[Dict]:
    api_response = _get_json_from_url(HF_API_URL)
    if not isinstance(api_response, list):
        return []
    return _parse_wheels_from_file_list(
        api_response, "huggingface", "path", "https://huggingface.co/mit-han-lab/nunchaku/resolve/main/"
    )


# NEW: Function to fetch releases directly from the ModelScope API.
def get_nunchaku_releases_from_modelscope() -> List[Dict]:
    api_response = _get_json_from_url(MODEL_SCOPE_API_URL)

    # Navigate the specific ModelScope API response structure: Data -> Files
    if isinstance(api_response, dict):
        inner_data = api_response.get("Data", {})
        file_list = inner_data.get("Files") if isinstance(inner_data, dict) else None
    else:
        file_list = None

    if not isinstance(file_list, list):
        return []

    # Use the "Name" key (capitalized) to get the filename.
    return _parse_wheels_from_file_list(
        file_list,
        "modelscope",
        "Name",
        "https://modelscope.cn/models/nunchaku-tech/nunchaku/resolve/master/",
    )


def fetch_and_structure_all_releases() -> Dict[str, Dict[str, Dict]]:
    structured_releases = {"github": {}, "huggingface": {}, "modelscope": {}}

    source_map = {
        "github": get_nunchaku_releases_from_github,
        "huggingface": get_nunchaku_releases_from_huggingface,
        "modelscope": get_nunchaku_releases_from_modelscope,  # Calls the new ModelScope function.
    }

    for source_name, fetch_func in source_map.items():
        for release in fetch_func():
            if tag := release.get("tag_name"):
                structured_releases[source_name][tag] = release

    if not any(structured_releases.values()):
        return {"github": {"latest": {"tag_name": "latest"}}}

    return structured_releases


def prepare_version_lists(structured_data: Dict[str, Dict[str, Dict]]) -> Tuple[List[str], List[str]]:
    official_tags, dev_tags = set(), set()
    for source_data in structured_data.values():
        for tag in source_data.keys():
            if "dev" not in tag:
                official_tags.add(tag.lstrip("v"))
    for tag in structured_data.get("github", {}).keys():
        if "dev" in tag:
            dev_tags.add(tag.lstrip("v"))
    return ["latest"] + sorted(list(official_tags), key=parse_version, reverse=True), sorted(
        list(dev_tags), key=parse_version, reverse=True
    )


def get_torch_version_string() -> Optional[str]:
    try:
        version = importlib.metadata.version("torch")
        version_parts = version.split(".")
        return f"torch{version_parts[0]}.{version_parts[1]}"
    except importlib.metadata.PackageNotFoundError:
        return None


def get_system_info() -> Dict[str, str]:
    os_name = platform.system().lower()
    os_key = "linux" if os_name == "linux" else "win" if os_name == "windows" else "unsupported"
    return {
        "os": os_key,
        "python_version": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "torch_version": get_torch_version_string(),
    }


def find_compatible_wheel(assets: List[Dict], sys_info: Dict[str, str]) -> Optional[Dict]:
    compatible_wheels = []
    wheel_regex = re.compile(r"nunchaku-.+\+(torch[\d.]+)-(cp\d+)-.+-(linux_x86_64|win_amd64)\.whl")
    for asset in assets:
        match = wheel_regex.match(asset.get("name", ""))
        if match:
            torch_v, python_v, _ = match.groups()
            os_key = "linux" if "linux" in asset["name"] else "win"
            if sys_info["os"] == os_key and sys_info["python_version"] == python_v:
                compatible_wheels.append(
                    {
                        "url": asset["browser_download_url"],
                        "name": asset["name"],
                        "torch_version_str": torch_v,
                        "torch_version_obj": parse_version(torch_v.replace("torch", "")),
                    }
                )

    if not compatible_wheels:
        return None
    if sys_info["torch_version"]:
        for wheel in compatible_wheels:
            if wheel["torch_version_str"] == sys_info["torch_version"]:
                return wheel
    return max(compatible_wheels, key=lambda w: w["torch_version_obj"])


def install_wheel(wheel_url: str, backend: str) -> str:
    if backend == "uv":
        command = [sys.executable, "-m", "uv", "pip", "install", wheel_url]
    else:  # Default to pip
        command = [sys.executable, "-m", "pip", "install", wheel_url]

    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
        )
        output_log = []
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
            output_log.append(line)
        process.wait()
        full_log = "".join(output_log)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, output=full_log)
        return full_log
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Command '{backend}' not found. Is it in your PATH?")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Installation failed (exit code {e.returncode}).\n\n--- LOG ---\n{e.output}") from e


# --- ComfyUI Node Definition ---

# Pre-fetch all release data on startup to improve performance.
ALL_RELEASES_DATA = fetch_and_structure_all_releases()
OFFICIAL_VERSIONS, DEV_VERSIONS = prepare_version_lists(ALL_RELEASES_DATA)
DEV_CHOICES = ["None"] + DEV_VERSIONS


class NunchakuWheelInstaller:
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Installer"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        from time import time

        return time()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (["github", "huggingface", "modelscope"], {}),
                "version": (OFFICIAL_VERSIONS, {}),
                "dev_version_github": (DEV_CHOICES, {"default": "None"}),
                "backend": (["pip", "uv"], {}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)

    def run(self, source: str, version: str, dev_version_github: str, backend: str):
        try:
            # CHANGE: Added automatic uninstallation of any pre-existing nunchaku version.
            if is_nunchaku_installed():
                print("An existing version of Nunchaku was detected. Attempting to uninstall automatically...")
                uninstall_command = [sys.executable, "-m", "pip", "uninstall", "nunchaku", "-y"]
                process = subprocess.Popen(
                    uninstall_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                output_log = []
                for line in iter(process.stdout.readline, ""):
                    print(line, end="")
                    output_log.append(line)
                process.wait()
                if process.returncode != 0:
                    full_log = "".join(output_log)
                    raise subprocess.CalledProcessError(process.returncode, uninstall_command, output=full_log)

                status_message = (
                    "✅ An existing version of Nunchaku was detected and uninstalled.\n\n"
                    "**Please restart ComfyUI completely.**\n\n"
                    "Then, run this node again to install the desired version."
                )
                return (status_message,)

            if dev_version_github != "None":
                final_version_tag = f"v{dev_version_github}"
                source = "github"
            else:
                final_version_tag = "latest" if version == "latest" else f"v{version}"

            sys_info = get_system_info()
            if sys_info["os"] == "unsupported":
                raise RuntimeError(f"Unsupported OS: {platform.system()}")

            source_versions = ALL_RELEASES_DATA.get(source, {})

            if final_version_tag == "latest":
                official_tags = [v.lstrip("v") for v in source_versions.keys() if "dev" not in v]
                if not official_tags:
                    raise RuntimeError(f"No official versions found on source '{source}'.")
                final_version_tag = f"v{sorted(official_tags, key=parse_version, reverse=True)[0]}"

            release_data = source_versions.get(final_version_tag)
            if not release_data:
                available_on = [s for s, data in ALL_RELEASES_DATA.items() if final_version_tag in data]
                msg = f"Version '{final_version_tag}' not available from '{source}'."
                if available_on:
                    msg += f" Try sources: {available_on}"
                raise RuntimeError(msg)

            assets = release_data.get("assets", [])
            if not assets:
                raise RuntimeError(f"No downloadable files found for version '{final_version_tag}'.")

            wheel_to_install = find_compatible_wheel(assets, sys_info)
            if not wheel_to_install:
                raise RuntimeError("Could not find a compatible wheel for your system.")

            log = install_wheel(wheel_to_install["url"], backend)
            status_message = f"✅ Success! Installed: {wheel_to_install['name']}\n\nRestart completely ComfyUI to apply changes.\n\n--- LOG ---\n{log}"

        except Exception as e:
            print(f"\n❌ An error occurred during installation:\n{e}")
            status_message = f"❌ ERROR:\n{str(e)}"

        return (status_message,)


NODE_CLASS_MAPPINGS = {"NunchakuWheelInstaller": NunchakuWheelInstaller}
NODE_DISPLAY_NAME_MAPPINGS = {"NunchakuWheelInstaller": "Nunchaku Installer"}
