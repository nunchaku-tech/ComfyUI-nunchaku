"""
This module provides an advanced utility node for installing the Nunchaku Python wheel.
It dynamically fetches available versions from GitHub, allows the user to select an
installer backend (pip or uv), and automatically finds the most compatible wheel.
The installation status is displayed directly on the node UI.
"""

import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import urllib.request
import urllib.error
from packaging.version import parse as parse_version
from typing import List, Dict, Optional

# --- Helper Functions ---
API_URL = "https://api.github.com/repos/nunchaku-tech/nunchaku"

def is_nunchaku_installed() -> bool:
    """Checks if nunchaku is already installed in the environment."""
    try:
        importlib.metadata.version("nunchaku")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False

def _get_json_from_url(url: str) -> List[Dict] | Dict:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'ComfyUI-Nunchaku-InstallerNode'})
        with urllib.request.urlopen(req) as response:
            if response.status == 200: return json.loads(response.read())
            print(f"Error: GitHub API returned status {response.status}")
            return []
    except Exception as e:
        print(f"Error fetching data from GitHub API: {e}")
        return []

def get_nunchaku_releases() -> List[Dict]:
    print("Fetching Nunchaku versions from GitHub...")
    releases = _get_json_from_url(f"{API_URL}/releases")
    if not releases or not isinstance(releases, list):
        print("Warning: Could not fetch Nunchaku versions. Using a fallback value.")
        return [{"tag_name": "latest"}]
    return releases

def get_torch_version_string() -> Optional[str]:
    try:
        version = importlib.metadata.version("torch")
        if not version: return None
        version_parts = version.split('.')
        return f"torch{version_parts[0]}.{version_parts[1]}"
    except importlib.metadata.PackageNotFoundError: return None

def get_system_info() -> Dict[str, str]:
    os_name = platform.system().lower()
    os_key = "linux" if os_name == "linux" else "win" if os_name == "windows" else "unsupported"
    return {"os": os_key, "python_version": f"cp{sys.version_info.major}{sys.version_info.minor}", "torch_version": get_torch_version_string()}

def find_compatible_wheel(assets: List[Dict], sys_info: Dict[str, str]) -> Optional[Dict]:
    compatible_wheels = []
    wheel_regex = re.compile(r"nunchaku-.+\+(torch[\d.]+)-(cp\d+)-.+-(linux_x86_64|win_amd64)\.whl")
    for asset in assets:
        match = wheel_regex.match(asset.get('name', ''))
        if match:
            torch_v, python_v, _ = match.groups()
            os_key = "linux" if "linux" in asset['name'] else "win"
            if sys_info["os"] == os_key and sys_info["python_version"] == python_v:
                compatible_wheels.append({
                    "url": asset["browser_download_url"], "name": asset["name"],
                    "torch_version_str": torch_v, "torch_version_obj": parse_version(torch_v.replace("torch", ""))
                })
    if not compatible_wheels: return None
    if sys_info["torch_version"]:
        for wheel in compatible_wheels:
            if wheel["torch_version_str"] == sys_info["torch_version"]:
                print(f"✅ Found wheel perfectly matching your PyTorch version ({sys_info['torch_version']}).")
                return wheel
        print(f"⚠️ No wheel exactly matches PyTorch version ({sys_info['torch_version']}). Selecting best alternative.")
    return max(compatible_wheels, key=lambda w: w["torch_version_obj"])

def install_wheel(wheel_url: str, backend: str) -> str:
    print(f"\n🚀 Starting installation of {wheel_url} using '{backend}'...")
    command = [sys.executable, "-m", backend, "pip", "install", wheel_url]
    output_log = []
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            output_log.append(line)
        process.stdout.close()
        return_code = process.wait()
        full_log = ''.join(output_log)
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command, output=full_log)
        return full_log
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Command '{backend}' not found. Is it installed and in your PATH?")
    except subprocess.CalledProcessError as e:
        error_message = f"Installation failed (exit code {e.returncode}).\n\n--- LOG ---\n{e.output}"
        raise RuntimeError(error_message) from e

# --- ComfyUI Node Definition ---
ALL_RELEASES = get_nunchaku_releases()
AVAILABLE_VERSIONS = [release['tag_name'] for release in ALL_RELEASES if 'tag_name' in release] or ["latest"]

class NunchakuWheelInstaller:
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Installer"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        from time import time
        return time()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "version": (AVAILABLE_VERSIONS, {}),
                "backend": (["pip", "uv"], {}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)

    def run(self, version: str, backend: str):
        status_message = "Ready. Waiting to start..."
        try:
            print("--- Nunchaku Installer Node ---")
            
            if is_nunchaku_installed():
                raise RuntimeError(
                    "Nunchaku is already installed.\n"
                    "Please uninstall it manually with the command:\n\n"
                    "pip uninstall nunchaku\n\n"
                    "Then, re-run this node."
                )

            sys_info = get_system_info()
            print(f"🔍 Detecting system config: OS={sys_info['os']}, Python={sys_info['python_version']}, PyTorch={sys_info['torch_version'] or 'Not found'}")
            if sys_info['os'] == 'unsupported':
                raise RuntimeError(f"Unsupported OS: {platform.system()}")

            print(f"⬇️ Finding release data for tag '{version}'...")
            release_data = next((r for r in ALL_RELEASES if r.get('tag_name') == version), None)
            if not release_data:
                url_to_fetch = f"{API_URL}/releases/latest" if version == "latest" else f"{API_URL}/releases/tags/{version}"
                release_data = _get_json_from_url(url_to_fetch)
                if isinstance(release_data, list): release_data = release_data[0] if release_data else None
            if not release_data: raise RuntimeError(f"Could not find release info for version '{version}'.")

            print(f"Found release: {release_data.get('name', version)}")
            assets = release_data.get("assets", [])
            wheel_to_install = find_compatible_wheel(assets, sys_info)
            if not wheel_to_install: raise RuntimeError("❌ Could not find a compatible wheel for your system.")

            print(f"🎯 Compatible wheel found: {wheel_to_install['name']}")
            log = install_wheel(wheel_to_install['url'], backend)

            status_message = (
                f"✅ Success! Installed:\n{wheel_to_install['name']}\n\n"
                "IMPORTANT: Please close ComfyUI completely and restart it for the changes to take effect.\n\n"
                f"--- INSTALLATION LOG ---\n{log}"
            )
        except Exception as e:
            print(f"\n❌ An error occurred during installation:\n{e}")
            status_message = f"❌ ERROR:\n{str(e)}"

        return (status_message,)

NODE_CLASS_MAPPINGS = {"NunchakuWheelInstaller": NunchakuWheelInstaller}
NODE_DISPLAY_NAME_MAPPINGS = {"NunchakuWheelInstaller": "Nunchaku Installer"}
