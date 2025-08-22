import torch
from comfy.model_base import ModelType, QwenImage

from ..models.qwenimage import NunchakuQwenImageTransformer2DModel

# from diffusers.models import QwenImageTransformer2DModel


class NunchakuQwenImage(QwenImage):
    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        super(QwenImage, self).__init__(
            model_config, model_type, device=device, unet_model=NunchakuQwenImageTransformer2DModel
        )
        self.memory_usage_factor_conds = ("ref_latents",)

    def load_model_weights(self, sd: dict[str, torch.Tensor], unet_prefix: str = ""):
        diffusion_model = self.diffusion_model
        state_dict = diffusion_model.state_dict()
        for k in state_dict.keys():
            if k not in sd:
                if ".wtscale" in k or ".wcscales" in k:
                    raise ValueError(f"Key {k} not found in state_dict")
                sd[k] = torch.ones_like(state_dict[k])
        diffusion_model.load_state_dict(sd, strict=True)
        # bf16_diffusion_model = QwenImageTransformer2DModel.from_pretrained(
        #     "Qwen/Qwen-Image", subfolder="transformer", torch_dtype=torch.bfloat16
        # )
        # for i in range(len(diffusion_model.transformer_blocks)):
        #     block = diffusion_model.transformer_blocks[i]
        #     bf16_block = bf16_diffusion_model.transformer_blocks[i]
        #     block.attn.to_out[0] = bf16_block.attn.to_out[0]
