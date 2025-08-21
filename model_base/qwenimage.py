from comfy.model_base import QwenImage, ModelType
from models.qwenimage import NunchakuQwenImageTransformer2DModel


class NunchakuQwenImage(QwenImage):
    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        super(QwenImage, self).__init__(
            model_config, model_type, device=device, unet_model=NunchakuQwenImageTransformer2DModel
        )
        self.memory_usage_factor_conds = ("ref_latents",)
