from comfy.model_base import QwenImage, ModelType
import comfy.ldm.qwen_image.model


class NunchakuQwenImage(QwenImage):
    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        super(QwenImage, self).__init__(
            model_config, model_type, device=device, unet_model=comfy.ldm.qwen_image.model.QwenImageTransformer2DModel
        )
        self.memory_usage_factor_conds = ("ref_latents",)
