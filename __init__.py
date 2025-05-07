# only import if running as a custom node
from .nodes.lora import NunchakuFluxLoraLoader
from .nodes.models import NunchakuFluxDiTLoader, NunchakuTextEncoderLoader
from .nodes.preprocessors import FluxDepthPreprocessor
from .nodes.models import NunchakuPulidApply


NODE_CLASS_MAPPINGS = {
    "NunchakuFluxDiTLoader": NunchakuFluxDiTLoader,
    "NunchakuTextEncoderLoader": NunchakuTextEncoderLoader,
    "NunchakuFluxLoraLoader": NunchakuFluxLoraLoader,
    "NunchakuDepthPreprocessor": FluxDepthPreprocessor,
    "NunchakuPulidApply": NunchakuPulidApply
}
NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
