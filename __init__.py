import logging
import os

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from .utils import get_package_metadata, get_plugin_version

logger.info(get_package_metadata("nunchaku"))
logger.info(f"ComfyUI-nunchaku version: {get_plugin_version()}")

from .nodes.lora.flux import NunchakuFluxLoraLoader
from .nodes.models.flux import NunchakuFluxDiTLoader

NODE_CLASS_MAPPINGS = {
    "NunchakuFluxDiTLoader": NunchakuFluxDiTLoader,
    "NunchakuFluxLoraLoader": NunchakuFluxLoraLoader,
}

try:
    from .nodes.models.text_encoder import NunchakuTextEncoderLoader, NunchakuTextEncoderLoaderV2

    NODE_CLASS_MAPPINGS["NunchakuTextEncoderLoader"] = NunchakuTextEncoderLoader
    NODE_CLASS_MAPPINGS["NunchakuTextEncoderLoaderV2"] = NunchakuTextEncoderLoaderV2
except ImportError:
    logger.exception("Optional nodes `NunchakuTextEncoderLoader` and `NunchakuTextEncoderLoaderV2` import failed:")

try:
    from .nodes.preprocessors.depth import FluxDepthPreprocessor

    NODE_CLASS_MAPPINGS["NunchakuDepthPreprocessor"] = FluxDepthPreprocessor
except ImportError:
    logger.exception("Optional node `NunchakuDepthPreprocessor` import failed:")

try:
    from .nodes.models.pulid import NunchakuPulidApply, NunchakuPulidLoader

    NODE_CLASS_MAPPINGS["NunchakuPulidApply"] = NunchakuPulidApply
    NODE_CLASS_MAPPINGS["NunchakuPulidLoader"] = NunchakuPulidLoader
except ImportError:
    logger.exception("Optional nodes `NunchakuPulidApply` and `NunchakuPulidLoader` import failed:")

try:
    from .nodes.tools.merge_safetensors import NunchakuModelMerger

    NODE_CLASS_MAPPINGS["NunchakuModelMerger"] = NunchakuModelMerger
except ImportError:
    logger.exception("Optional node `NunchakuModelMerger` import failed:")

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
