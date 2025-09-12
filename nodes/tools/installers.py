"""
This module provides an advanced utility node for installing the Nunchaku Python wheel.
It dynamically constructs the wheel filename and URL based on a local configuration file
('nunchaku_versions.json') and the user's system environment (OS, Python, Torch).
It automatically detects the installer backend (uv or pip) and attempts to download
from a prioritized list of sources. The installation status is displayed on the node UI.
"""

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import urllib.error
from typing import Dict, List, Optional, Tuple

from packaging.version import parse as parse_version

# --- Helper Functions ---

# Defines the name of the local JSON file containing version compatibility info.
LOCAL_VERSIONS_FILE = "nunchaku_versions.json"


def is_nunchaku_installed() -> bool:
    try:
        importlib.metadata.version("nunchaku")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def load_version_config() -> Dict:
    """Loads the version compatibility configuration from the JSON file."""
    try:
        node_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(node_dir, LOCAL_VERSIONS_FILE)
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading or parsing '{LOCAL_VERSIONS_FILE}': {e}")
        return {}


# CHANGE: This function now reads versions directly from the top-level of the config.
def prepare_version_lists(version_config: Dict) -> Tuple[List[str], List[str]]:
    """Prepares official and dev version lists from the loaded configuration."""
    official_tags = version_config.get("versions", [])
    dev_tags = version_config.get("dev_versions", [])

    return ["latest"] + sorted(list(official_tags), key=parse_version, reverse=True), sorted(
        list(dev_tags), key=parse_version, reverse=True
    )


def get_torch_version_string() -> Optional[str]:
    try:
        version = importlib.metadata.version("torch")
        version_parts = version.split(".")
        # Returns in the format 'torch2.1'
        return f"torch{version_parts[0]}.{version_parts[1]}"
    except importlib.metadata.PackageNotFoundError:
        return None


def get_system_info() -> Dict[str, str]:
    os_name = platform.system().lower()
    os_key = "linux" if os_name == "linux" else "win" if os_name == "windows" else "unsupported"
    platform_tag = "linux_x86_64" if os_key == "linux" else "win_amd64" if os_key == "win" else "unsupported"

    return {
        "os": os_key,
        "platform_tag": platform_tag,
        "python_version": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "torch_version": get_torch_version_string(),
    }


# NEW: Function to automatically detect the best available installation backend.
def get_install_backend() -> str:
    """Checks for 'uv' and falls back to 'pip' if not found."""
    try:
        # Check if 'uv' is available and executable
        subprocess.run(
            [sys.executable, "-m", "uv", "--version"],
            check=True,
            capture_output=True,
        )
        print("Detected 'uv' backend.")
        return "uv"
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Could not find 'uv', falling back to 'pip' backend.")
        return "pip"


# CHANGE: This function now gets compatibility info from the top-level of the config.
def construct_compatible_wheel_info(
    version: str, source: str, sys_info: Dict[str, str], config: Dict
) -> Optional[Dict]:
    url_template = config.get("url_templates", {}).get(source)
    if not url_template:
        return None

    # Check for Python version compatibility from the top-level config
    if sys_info["python_version"] not in config.get("supported_python", []):
        return None

    supported_torch_versions = config.get("supported_torch", [])
    if not supported_torch_versions:
        return None

    # Find the best compatible Torch version
    compatible_torch_version = None
    if sys_info["torch_version"] in supported_torch_versions:
        compatible_torch_version = sys_info["torch_version"]
    else:
        # Fallback: find the latest supported torch version that is not newer
        # than the user's installed version.
        user_torch_obj = parse_version(sys_info["torch_version"].replace("torch", ""))
        available = sorted(
            [v for v in supported_torch_versions if parse_version(v.replace("torch", "")) <= user_torch_obj],
            key=lambda v: parse_version(v.replace("torch", "")),
            reverse=True,
        )
        if available:
            compatible_torch_version = available[0]

    if not compatible_torch_version:
        return None

    # Build the filename from the template
    filename_template = config.get("filename_template")
    if not filename_template:
        return None

    filename = filename_template.format(
        version=version,
        torch_version=compatible_torch_version,
        python_version=sys_info["python_version"],
        platform=sys_info["platform_tag"],
    )

    # CHANGE: New logic to correctly format the version_tag for URLs.
    # This logic now correctly handles GitHub tags for both official and dev releases.
    version_tag = ""
    if "dev" in version:
        # For a dev version like "1.0.0.dev123", create tag "v1.0.0dev123"
        version_tag = "v" + version.replace(".dev", "dev")
    else:
        # For an official version like "1.0.0", create tag "v1.0.0"
        version_tag = "v" + version

    download_url = url_template.format(version_tag=version_tag, filename=filename)

    return {"url": download_url, "name": filename}


def install_wheel(wheel_url: str, backend: str) -> str:
    if backend == "uv":
        command = [sys.executable, "-m", "uv", "pip", "install", wheel_url]
    else:  # Default to pip
        command = [sys.executable, "-m", "pip", "install", wheel_url]

    try:
        # We add a check for the URL status before attempting to install
        req = urllib.request.Request(wheel_url, method="HEAD", headers={"User-Agent": "ComfyUI-Nunchaku-InstallerNode"})
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status not in (200, 302): # 302 for GitHub redirects
                raise urllib.error.URLError(f"File not found at URL (status: {response.status})")

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
    except urllib.error.URLError as e:
         raise RuntimeError(f"Failed to download wheel. URL: {wheel_url}\nError: {e}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Installation failed (exit code {e.returncode}).\n\n--- LOG ---\n{e.output}") from e


# --- ComfyUI Node Definition ---

# Pre-load version config from the local file on startup.
try:
    VERSION_CONFIG = load_version_config()
    if not VERSION_CONFIG or not VERSION_CONFIG.get("versions"):
        raise FileNotFoundError(f"'{LOCAL_VERSIONS_FILE}' not found or is empty/invalid.")

    OFFICIAL_VERSIONS, DEV_VERSIONS = prepare_version_lists(VERSION_CONFIG)
    DEV_CHOICES = ["None"] + DEV_VERSIONS

except Exception as e:
    print(f"File/Config error during initialization: {e}. Node will run in offline mode.")
    VERSION_CONFIG = {}
    OFFICIAL_VERSIONS = ["config file error"]
    DEV_VERSIONS = []
    DEV_CHOICES = ["None"]


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
    # CHANGE: Removed 'source' and 'backend' from the user inputs.
    def INPUT_TYPES(cls):
        return {
            "required": {
                "version": (OFFICIAL_VERSIONS, {}),
                "dev_version": (DEV_CHOICES, {"default": "None"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)

    # CHANGE: Signature updated to remove 'source' and 'backend'.
    def run(self, version: str, dev_version: str):

        if version == "config file error":
            status_message = (
                f"❌ '{LOCAL_VERSIONS_FILE}' not found or invalid on startup.\n\n"
                f"Please create/fix this file in the node's directory "
                f"and then restart ComfyUI."
            )
            return (status_message,)

        try:
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

            # Determine the final version string to use
            if dev_version != "None":
                final_version = dev_version
                sources_to_try = ["github"]  # Dev versions are only on GitHub
            else:
                if version == "latest":
                    final_version = sorted(VERSION_CONFIG["versions"], key=parse_version, reverse=True)[0]
                else:
                    final_version = version
                # NEW: Define the prioritized list of sources for official versions
                sources_to_try = ["modelscope", "huggingface", "github"]
            
            sys_info = get_system_info()
            if sys_info["os"] == "unsupported":
                raise RuntimeError(f"Unsupported OS: {platform.system()}")

            # NEW: Auto-detect the installation backend
            backend = get_install_backend()

            # NEW: Loop through sources and try to install
            wheel_to_install = None
            final_log = ""
            last_error = ""

            for source in sources_to_try:
                print(f"\n--- Attempting to use source: {source} ---")
                
                # First, construct wheel info to see if it's even possible for this system
                wheel_info = construct_compatible_wheel_info(final_version, source, sys_info, VERSION_CONFIG)
                
                if not wheel_info:
                    print(f"No compatible wheel configuration found for your system from source '{source}'.")
                    last_error = (
                        "Could not find a compatible wheel for your system based on the config file.\n"
                        f"Your System: Python {sys_info['python_version']}, Torch {sys_info['torch_version']}, OS: {sys_info['os']}"
                    )
                    continue # Try next source

                try:
                    print(f"Attempting to download and install: {wheel_info['name']}")
                    final_log = install_wheel(wheel_info["url"], backend)
                    wheel_to_install = wheel_info  # Success!
                    print(f"--- Successfully installed from {source} ---")
                    break  # Exit the loop on success
                except (RuntimeError, urllib.error.URLError) as e:
                    print(f"Failed to install from {source}: {e}")
                    last_error = str(e)
                    # Continue to the next source
            
            if not wheel_to_install:
                raise RuntimeError(
                    f"Failed to install from all available sources.\n\nLast error:\n{last_error}"
                )

            status_message = f"✅ Success! Installed: {wheel_to_install['name']}\n\nRestart completely ComfyUI to apply changes.\n\n--- LOG ---\n{final_log}"

        except Exception as e:
            print(f"\n❌ An error occurred during installation:\n{e}")
            status_message = f"❌ ERROR:\n{str(e)}"

        return (status_message,)


NODE_CLASS_MAPPINGS = {"NunchakuWheelInstaller": NunchakuWheelInstaller}
NODE_DISPLAY_NAME_MAPPINGS = {"NunchakuWheelInstaller": "Nunchaku Installer"}
