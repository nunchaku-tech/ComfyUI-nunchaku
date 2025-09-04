import comfy.model_patcher


class NunchakuModelPatcher(comfy.model_patcher.ModelPatcher):
    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        with self.use_ejected():
            self.model.diffusion_model.to(device_to)

    def detach(self, unpatch_all: bool = True):
        self.eject_model()
        self.model.diffusion_model.to(self.offload_device)
