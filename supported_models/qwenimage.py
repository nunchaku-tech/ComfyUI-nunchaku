from comfy.supported_models import QwenImage

import model_base


class NunchakuQwenImage(QwenImage):
    def get_model(self, state_dict, prefix="", device=None) -> model_base.NunchakuQwenImage:
        out = model_base.NunchakuQwenImage(self, device=device)
        return out
