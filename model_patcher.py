"""
This module wraps the ComfyUI model patcher for Nunchaku models to load and unload the model correctly.
"""
from typing import Optional

import torch

from .mixins.model import NunchakuModelMixin
from comfy.model_base import BaseModel


class NunchakuModelPatcher:
    """
    This class extends the ComfyUI ModelPatcher to provide custom logic for loading and unloading the model correctly.
    """

    def __init__(self, model: BaseModel, load_device: torch.device, offload_device: torch.device):
        self.model: BaseModel = model
        self.load_device = load_device
        self.offload_device = offload_device

    def _to(self, device_to: torch.device):
        if hasattr(self.model.diffusion_model, "to_safely"):
            diffusion_model: NunchakuModelMixin = self.model.diffusion_model
            diffusion_model.to_safely(device_to)
        else:
            self.model.to(device_to)

    # required
    def patch_model(self, device_to: torch.device | None = None, lowvram_model_memory: int = 0, load_weights: bool = True, force_patch_weights: bool = False) -> torch.nn.Module:
        self._to(device_to=device_to)
        return self.model

    # required
    def unpatch_model(self, device_to: torch.device | None = None, unpatch_weights: Optional[bool] = False) -> torch.nn.Module:
        self._to(device_to=device_to)
        return self.model

    def is_clone(self, other: "NunchakuModelPatcher") -> bool:
        return other.model is self.model

    def clone_has_same_weights(self, clone: "NunchakuModelPatcher") -> bool:
        return clone.model is self.model

    def model_size(self) -> int:
        from comfy.model_management import module_size
        return module_size(self.model)

    def model_patches_to(self, arg: torch.device | torch.dtype):
        pass

    def model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def lowvram_patch_counter(self) -> int:
        """
        Returns a counter related to low VRAM patching, used to decide if a reload is necessary.
        """
        return 0

    def partially_load(self, device_to: torch.device, extra_memory: int = 0, force_patch_weights: bool = False):
        self.patch_model(device_to=device_to)
        return self.model_size()

    def partially_unload(self, device_to: torch.device, memory_to_free: int = 0):
        self.unpatch_model(device_to)
        return self.model_size()

    def memory_required(self, input_shape) -> int:
        return self.model.memory_required(input_shape=input_shape)

    def loaded_size(self) -> int:
        # todo: this will depend on cpu offloading
        if self.current_loaded_device() == self.load_device:
            return self.model_size()
        return 0

    def current_loaded_device(self) -> torch.device:
        return self.current_device

    def get_model_object(self, name: str) -> torch.nn.Module:
        from . import utils
        return utils.get_attr(self.model, name)

    @property
    def model_options(self):
        return self.model_options

    @model_options.setter
    def model_options(self, value):
        self.model_options = value

    def __del__(self):
        if hasattr(self.model, "__del__"):
            del self.model

    @property
    def parent(self) -> None:
        """
        Used for tracking a parent model from which this was cloned
        :return:
        """
        return None

    def detach(self, unpatch_all: bool = True):
        """
        Unloads the model
        :param unpatch_all:
        :return:
        """
        self.unpatch_model(self.offload_device, unpatch_weights=unpatch_all)
        return self.model

    def model_patches_models(self) -> list["NunchakuModelPatcher"]:
        """
        Used to implement Qwen DiffSynth Controlnets (?)
        :return:
        """
        return []
