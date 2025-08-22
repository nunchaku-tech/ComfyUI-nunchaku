from comfy.supported_models import QwenImage

from .. import model_base


class NunchakuQwenImage(QwenImage):
    def get_model(self, state_dict, prefix="", device=None, **kwargs) -> model_base.NunchakuQwenImage:
        out = model_base.NunchakuQwenImage(self, device=device, **kwargs)
        return out
