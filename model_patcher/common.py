"""
This module wraps the ComfyUI model patcher for Nunchaku models to load and unload the model correctly.
"""

import comfy.model_management
from comfy.model_patcher import ModelPatcher

try:
    from comfy.model_patcher import ModelPatcherDynamic as ModelPatcherBase
    _HAS_DYNAMIC = True
except ImportError:
    from comfy.model_patcher import ModelPatcher as ModelPatcherBase
    _HAS_DYNAMIC = False


class NunchakuModelPatcher(ModelPatcherBase):
    """
    This class extends the ComfyUI ModelPatcher to provide custom logic for loading and unloading the model correctly.
    """

    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        if _HAS_DYNAMIC:
            # Bypass ModelPatcherDynamic.__init__ to avoid HostBuffer(0,0,0) allocations
            # that cause Windows heap corruption (0xc0000374) when GC'd with Nunchaku's
            # custom load/detach. We still enable smart caching via is_dynamic().
            ModelPatcher.__init__(self, model, load_device, offload_device, size, weight_inplace_update=False)
            # Initialize minimal structures expected by ComfyUI's dynamic-aware code paths
            if not hasattr(self.model, "dynamic_vbars"):
                self.model.dynamic_vbars = {}
            if not hasattr(self.model, "dynamic_pins"):
                self.model.dynamic_pins = {}
            if self.load_device not in self.model.dynamic_pins:
                # Use None instead of HostBuffer to avoid crash — reset_cast_buffers
                # will replace with properly-allocated HostBuffers later if needed
                self.model.dynamic_pins[self.load_device] = {
                    "weights": None,
                    "patches": None,
                    "hostbufs_initialized": True,
                    "failed": False,
                    "active": False,
                }
            self.non_dynamic_delegate_model = None
            self.register_load_device(self.load_device)
        else:
            super().__init__(model, load_device, offload_device, size, weight_inplace_update=False)

    def is_dynamic(self):
        """Enable smart caching in ComfyUI — non-dynamic models get aggressively evicted."""
        return _HAS_DYNAMIC

    def _vbar_get(self, create=False):
        """Nunchaku manages its own VRAM — no vbar needed."""
        return None

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

