import logging
import os
from pathlib import Path

import yaml
from packaging.version import InvalidVersion, Version

# Register model folder paths for ComfyUI's model scanner
try:
    import folder_paths

    models_yaml_path = Path(__file__).parent / "test_data" / "models.yaml"
    with open(models_yaml_path, "r") as f:
        nunchaku_models_yaml = yaml.safe_load(f)

    for model in nunchaku_models_yaml["models"]:
        folder_name = model.get("sub_folder", "diffusion_models")
        if folder_name not in folder_paths.folder_names_and_paths:
            default_dir = os.path.join(folder_paths.models_dir, folder_name)
            folder_paths.folder_names_and_paths[folder_name] = (
                [default_dir],
                folder_paths.supported_pt_extensions,
            )
except Exception:
    pass

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("=" * 40 + " ComfyUI-nunchaku Initialization " + "=" * 40)

from .utils import get_package_version, get_plugin_version

nunchaku_full_version = get_package_version("nunchaku")
nunchaku_version = nunchaku_full_version.split("+")[0].strip()
nunchaku_major_minor_patch_version = ".".join(nunchaku_version.split(".")[:3])

logger.info(f"Nunchaku version: {nunchaku_full_version}")
logger.info(f"ComfyUI-nunchaku version: {get_plugin_version()}")


min_nunchaku_version = "1.0.0"

try:
    if Version(nunchaku_major_minor_patch_version) < Version(min_nunchaku_version):
        logger.warning(
            f"ComfyUI-nunchaku {get_plugin_version()} requires nunchaku >= v{min_nunchaku_version}, "
            f"but found nunchaku {nunchaku_full_version}. Please update nunchaku."
        )
except InvalidVersion:
    logger.warning(
        f"Could not parse nunchaku version: {nunchaku_full_version}. "
        f"Please ensure you have at least v{min_nunchaku_version}."
    )

NODE_CLASS_MAPPINGS = {}

try:
    from .nodes.models.flux import NunchakuFluxDiTLoader

    NODE_CLASS_MAPPINGS["NunchakuFluxDiTLoader"] = NunchakuFluxDiTLoader
except ImportError:
    logger.exception("Node `NunchakuFluxDiTLoader` import failed:")

try:
    from .nodes.models.qwenimage import NunchakuQwenImageDiTLoader

    NODE_CLASS_MAPPINGS["NunchakuQwenImageDiTLoader"] = NunchakuQwenImageDiTLoader
except ImportError:
    logger.exception("Node `NunchakuQwenImageDiTLoader` import failed:")

try:
    from .nodes.lora.flux import NunchakuFluxLoraLoader, NunchakuFluxLoraStack

    NODE_CLASS_MAPPINGS["NunchakuFluxLoraLoader"] = NunchakuFluxLoraLoader
    NODE_CLASS_MAPPINGS["NunchakuFluxLoraStack"] = NunchakuFluxLoraStack
except ImportError:
    logger.exception("Nodes `NunchakuFluxLoraLoader` and `NunchakuFluxLoraStack` import failed:")

try:
    from .nodes.models.text_encoder import NunchakuTextEncoderLoaderV2

    NODE_CLASS_MAPPINGS["NunchakuTextEncoderLoaderV2"] = NunchakuTextEncoderLoaderV2
except ImportError:
    logger.exception("Node `NunchakuTextEncoderLoaderV2` import failed:")

try:
    from .nodes.models.pulid import NunchakuFluxPuLIDApplyV2, NunchakuPuLIDLoaderV2

    NODE_CLASS_MAPPINGS["NunchakuPuLIDLoaderV2"] = NunchakuPuLIDLoaderV2
    NODE_CLASS_MAPPINGS["NunchakuFluxPuLIDApplyV2"] = NunchakuFluxPuLIDApplyV2
except ImportError:
    logger.exception("Nodes `NunchakuPuLIDLoaderV2` and `NunchakuFluxPuLIDApplyV2` import failed:")
try:
    from .nodes.models.ipadapter import NunchakuFluxIPAdapterApply, NunchakuIPAdapterLoader

    NODE_CLASS_MAPPINGS["NunchakuFluxIPAdapterApply"] = NunchakuFluxIPAdapterApply
    NODE_CLASS_MAPPINGS["NunchakuIPAdapterLoader"] = NunchakuIPAdapterLoader
except ImportError:
    logger.exception("Nodes `NunchakuFluxIPAdapterApply` and `NunchakuIPAdapterLoader` import failed:")

try:
    from .nodes.models.zimage import NunchakuZImageDiTLoader

    NODE_CLASS_MAPPINGS["NunchakuZImageDiTLoader"] = NunchakuZImageDiTLoader
except ImportError:
    logger.exception("Nodes `NunchakuZImageDiTLoader` import failed:")

try:
    from .nodes.tools.merge_safetensors import NunchakuModelMerger

    NODE_CLASS_MAPPINGS["NunchakuModelMerger"] = NunchakuModelMerger
except ImportError:
    logger.exception("Node `NunchakuModelMerger` import failed:")

try:
    from .nodes.tools.installers import NunchakuWheelInstaller

    NODE_CLASS_MAPPINGS["NunchakuWheelInstaller"] = NunchakuWheelInstaller
except ImportError:
    logger.exception("Node `NunchakuWheelInstaller` import failed:")

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
logger.info("=" * (80 + len(" ComfyUI-nunchaku Initialization ")))
