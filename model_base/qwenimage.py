import torch
from comfy.model_base import ModelType, QwenImage

from ..models.qwenimage import NunchakuQwenImageTransformer2DModel


class NunchakuQwenImage(QwenImage):
    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        super(QwenImage, self).__init__(
            model_config, model_type, device=device, unet_model=NunchakuQwenImageTransformer2DModel
        )
        self.memory_usage_factor_conds = ("ref_latents",)

    def load_model_weights(self, sd: dict[str, torch.Tensor], unet_prefix: str = ""):
        super().load_model_weights(sd, unet_prefix)
        diffusion_model = self.diffusion_model
        state_dict = diffusion_model.state_dict()
        for k in state_dict.keys():
            if k not in sd:
                assert ".wtscale" in k or ".wcscales" in k
                sd[k] = torch.ones_like(state_dict[k])
            else:
                assert state_dict[k].dtype == sd[k].dtype
        diffusion_model.load_state_dict(sd)
