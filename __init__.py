import logging
import os

from .nodes.lora.flux import NunchakuFluxLoraLoader
from .nodes.models.flux import NunchakuFluxDiTLoader

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


NODE_CLASS_MAPPINGS = {
    "NunchakuFluxDiTLoader": NunchakuFluxDiTLoader,
    "NunchakuFluxLoraLoader": NunchakuFluxLoraLoader,
}

try:
    from .nodes.models.text_encoder import NunchakuTextEncoderLoader, NunchakuTextEncoderLoaderV2

    NODE_CLASS_MAPPINGS["NunchakuTextEncoderLoader"] = NunchakuTextEncoderLoader
    NODE_CLASS_MAPPINGS["NunchakuTextEncoderLoaderV2"] = NunchakuTextEncoderLoaderV2
except ImportError as e:
    logger.warning(f"Optional nodes `NunchakuTextEncoderLoader` and `NunchakuTextEncoderLoaderV2` import failed:\n{e}")

try:
    from .nodes.preprocessors.depth import FluxDepthPreprocessor

    NODE_CLASS_MAPPINGS["NunchakuDepthPreprocessor"] = FluxDepthPreprocessor
except ImportError as e:
    logger.warning(f"Optional node `NunchakuDepthPreprocessor` import failed:\n{e}")

try:
    from .nodes.models.pulid import NunchakuPulidApply, NunchakuPulidLoader

    NODE_CLASS_MAPPINGS["NunchakuPulidApply"] = NunchakuPulidApply
    NODE_CLASS_MAPPINGS["NunchakuPulidLoader"] = NunchakuPulidLoader
except ImportError as e:
    logger.warning(f"Optional nodes `NunchakuPulidApply` and `NunchakuPulidLoader` import failed:\n{e}")

try:
    from .nodes.tools.merge_safetensors import NunchakuModelMerger

    NODE_CLASS_MAPPINGS["NunchakuModelMerger"] = NunchakuModelMerger
except ImportError as e:
    logger.warning(f"Optional node `NunchakuModelMerger` import failed:\n{e}")

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
