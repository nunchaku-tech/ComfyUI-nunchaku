import platform
import subprocess
import sys

import torch


def install(package: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


class NunchakuWheelInstaller:

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Wheel Installer"

    @classmethod
    def INPUT_TYPES(s):
        support_versions = ["v0.3.1"]

        return {
            "required": {
                "source": (
                    ["GitHub Release", "HuggingFace", "ModelScope"],
                    {"tooltip": "Source for downloading nunchaku wheels."},
                ),
                "version": (support_versions, {"tooltip": "Specific version to install."}),
            }
        }

    def run(self, source: str, version: str):
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        torch_version = torch.__version__.split("+")[0]
        torch_major_minor_version = ".".join(torch_version.split(".")[:2])
        torch_tag = f"torch{torch_major_minor_version}"  # e.g., 2.5

        if platform.system() == "Windows":
            platform_tag = "win_amd64"
        elif platform.system() == "Linux":
            platform_tag = "linux_x86_64"
        else:
            return (f"Unsupported platform: {platform.system()}",)
        wheel_name = f"nunchaku-{version[1:]}+{torch_tag}-{python_version}-{python_version}-{platform_tag}.whl"

        if source == "GitHub Release":
            url_prefix = f"https://github.com/mit-han-lab/nunchaku/releases/download/{version}/"
        elif source == "HuggingFace":
            url_prefix = "https://huggingface.co/mit-han-lab/nunchaku/resolve/main/"
        elif source == "ModelScope":
            url_prefix = "https://modelscope.cn/models/Lmxyy1999/nunchaku/resolve/master/"
        else:
            raise NotImplementedError(f"Unsupported source: {source}")

        url = url_prefix + wheel_name

        try:
            install(url)
            return (f"✅ Successfully installed {wheel_name} from {source}.",)
        except Exception as e:
            return (f"❌ Failed to install {wheel_name} from {source}: {str(e)}",)
