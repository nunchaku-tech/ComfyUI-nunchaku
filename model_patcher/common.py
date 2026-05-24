"""
This module wraps the ComfyUI model patcher for Nunchaku models to load and unload the model correctly.
"""

try:
    from comfy.model_patcher import ModelPatcherDynamic as ModelPatcherBase
except ImportError:
    from comfy.model_patcher import ModelPatcher as ModelPatcherBase


class NunchakuModelPatcher(ModelPatcherBase):
    """
    This class extends the ComfyUI ModelPatcher to provide custom logic for loading and unloading the model correctly.
    """

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False, **kwargs):
        """
        Load the diffusion model onto the specified device.

        Parameters
        ----------
        device_to : torch.device or str, optional
            The device to which the diffusion model should be moved.
        lowvram_model_memory : int, optional
            Not used in this implementation.
        force_patch_weights : bool, optional
            Not used in this implementation.
        full_load : bool, optional
            Not used in this implementation.
        """
        with self.use_ejected():
            self.model.diffusion_model.to_safely(device_to)

    def detach(self, unpatch_all: bool = True, **kwargs):
        """
        Detach the model and move it to the offload device.

        Parameters
        ----------
        unpatch_all : bool, optional
            If True, unpatch all model components (default is True).
        """
        self.eject_model()
        self.model.diffusion_model.to_safely(self.offload_device)

    def is_dynamic(self):
        """
        Hack to prevent ComfyUI's comfy_aimdo from attempting to manage HostBuffers for this model,
        which leads to heap corruption and CUDA unregister failures, as Nunchaku uses its own memory pinning.
        We return False for memory pinning functions, but True otherwise to keep smart caching enabled.
        """
        import inspect
        frame = inspect.currentframe()
        try:
            while frame is not None:
                if frame.f_code.co_name in ("reset_cast_buffers", "free_pins", "ensure_pin_budget", "ensure_pin_registerable"):
                    return False
                frame = frame.f_back
        except Exception:
            pass
        finally:
            del frame
        return True
