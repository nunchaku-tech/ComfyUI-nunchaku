"""
This module wraps the ComfyUI model patcher for Nunchaku models to load and unload the model correctly.
"""

import comfy.model_management
from comfy.model_patcher import ModelPatcher

try:
    from comfy.model_patcher import ModelPatcherDynamic as ModelPatcherBase
except ImportError:
    from comfy.model_patcher import ModelPatcher as ModelPatcherBase


class NunchakuModelPatcher(ModelPatcherBase):
    """
    This class extends the ComfyUI ModelPatcher to provide custom logic for loading and unloading the model correctly.
    """

    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        ModelPatcher.__init__(self, model, load_device, offload_device, size, weight_inplace_update=False)
        self.non_dynamic_delegate_model = None

    def is_dynamic(self):
        """Return False to prevent ComfyUI from creating HostBuffer objects in reset_cast_buffers,
        which causes heap corruption on Windows with Nunchaku's custom load/detach.
        Nunchaku manages its own VRAM — no ComfyUI pin management needed.
        """
        return False

    def _vbar_get(self, create=False):
        """Nunchaku manages its own VRAM — no vbar needed."""
        return

    def partially_unload_ram(self, *args, **kwargs):
        """Nunchaku manages its own pin memory — no-op."""
        return 0

    def unregister_inactive_pins(self, *args, **kwargs):
        """Nunchaku manages its own pin memory — no-op."""
        return 0

    def loaded_ram_size(self):
        """Nunchaku doesn't use pinned host buffers."""
        return 0

    def pinned_memory_size(self):
        """Nunchaku doesn't use pinned host buffers."""
        return 0

    def get_non_dynamic_delegate(self):
        """Nunchaku manages its own memory — no non-dynamic delegate needed."""
        return self

    def __del__(self):
        """Override to prevent ModelPatcherDynamic.__del__ from running cleanup on Nunchaku models.
        Nunchaku manages its own memory lifecycle — no ComfyUI cleanup needed.
        """

    def pin_weight_to_device(self, key):
        pass

    def unpin_weight(self, key):
        pass

    def unpin_all_weights(self):
        pass

    def patch_cached_hook_weights(self, cached_weights, key, memory_counter):
        pass

    def patch_hook_weight_to_device(self, hooks, combined_patches, key, original_weights, memory_counter):
        pass

    @staticmethod
    def _to_safely(module, device, non_blocking=True):
        try:
            module.to_safely(device, non_blocking=non_blocking)
        except TypeError:
            module.to_safely(device)

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
        if device_to is not None:
            model_size = getattr(self.model, "model_size", 0)
            if model_size > 0:
                comfy.model_management.free_memory(model_size, device_to)

        with self.use_ejected():
            self._to_safely(self.model.diffusion_model, device_to)

        self.model.comfy_patched_weights = True

    def detach(self, unpatch_all: bool = True, **kwargs):
        """
        Detach the model and move it to the offload device.

        Parameters
        ----------
        unpatch_all : bool, optional
            If True, unpatch all model components (default is True).
        """
        self.eject_model()
        self._to_safely(self.model.diffusion_model, self.offload_device)
        self.model.comfy_patched_weights = False

