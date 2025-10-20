from typing import Callable, List, Tuple, Union
from pathlib import Path

import torch
from torch import nn

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.caching.fbcache import cache_context, create_cache_context
from ..nunchaku_code.lora_qwen import compose_loras_v2, reset_lora_v2


class ComfyQwenImageWrapper(nn.Module):
    """
    Wrapper for NunchakuQwenImageTransformer2DModel to support ComfyUI workflows.

    This wrapper separates LoRA composition from the forward pass for maximum efficiency.
    It detects changes to its `loras` attribute and recomposes the underlying model
    lazily when the forward pass is executed.
    """

    def __init__(
        self,
        model: NunchakuQwenImageTransformer2DModel,
        config,
        customized_forward: Callable = None,
        forward_kwargs: dict | None = None,
    ):
        super().__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config
        # This list is the authoritative state, modified by LoRA loader nodes
        self.loras: List[Tuple[Union[str, Path, dict], float]] = []
        # This tracks the LoRAs currently composed into the model to detect changes
        self._applied_loras: List[Tuple[Union[str, Path, dict], float]] = None

        self.customized_forward = customized_forward
        self.forward_kwargs = forward_kwargs or {}

        self._prev_timestep = None
        self._cache_context = None

        # Reusable tensor caches keyed by (H, W, device, dtype, index, offsets)
        self._img_ids_cache = {}
        # Cache for txt ids keyed by (batch, seq_len, device, dtype)
        self._txt_ids_cache = {}
        # Base linspace caches keyed by (length, device, dtype)
        self._linspace_cache_h = {}
        self._linspace_cache_w = {}

    def to_safely(self, device):
        """Safely move the model to the specified device."""
        if hasattr(self.model, "to_safely"):
            self.model.to_safely(device)
        else:
            self.model.to(device)
        return self


    def forward(
        self,
        x,
        timestep,
        context=None,
        y=None,
        guidance=None,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        """
        Forward pass for the wrapped model.

        Detects changes to the `self.loras` list and recomposes the model
        on-the-fly before inference.
        """

        timestep_float = timestep.item() if isinstance(timestep, torch.Tensor) else float(timestep)

        # Check if the LoRA stack has been changed by a loader node
        if self._applied_loras != self.loras:
            # The compose function handles resetting before applying the new stack
            reset_lora_v2(self.model)
            self._applied_loras = self.loras.copy()
            compose_loras_v2(self.model, self.loras)

            # BUG FIX v2: Force a full teardown and rebuild of the OffloadManager
            # This ensures all old buffers are deallocated before creating new ones.
            if hasattr(self.model, "offload_manager") and self.model.offload_manager is not None:
                # Store the settings from the old manager before destroying it
                manager = self.model.offload_manager
                offload_settings = {
                    "num_blocks_on_gpu": manager.num_blocks_on_gpu,
                    "use_pin_memory": manager.use_pin_memory,
                }

                # Step 1: Completely disable and clear the old offloader
                self.model.set_offload(False)

                # Step 2: Re-enable offloading, forcing it to rebuild
                # with the new tensor shapes and the original settings.
                self.model.set_offload(True, **offload_settings)

        # Caching logic
        use_caching = getattr(self.model, "residual_diff_threshold_multi", 0) != 0 or getattr(self.model, "_is_cached", False)
        if use_caching:
            cache_invalid = self._prev_timestep is None or self._prev_timestep < timestep_float + 1e-5
            if cache_invalid:
                self._cache_context = create_cache_context()
            self._prev_timestep = timestep_float

            with cache_context(self._cache_context):
                out = self._execute_model(x, timestep, context, guidance, control, transformer_options, **kwargs)
        else:
            out = self._execute_model(x, timestep, context, guidance, control, transformer_options, **kwargs)

        if isinstance(out, tuple):
            out = out[0]

        if x.ndim == 5 and out.ndim == 4:
            out = out.unsqueeze(2)

        return out

    def _execute_model(self, x, timestep, context, guidance, control, transformer_options, **kwargs):
        """Helper function to run the model's forward pass."""
        model_device = next(self.model.parameters()).device

        # Move input tensors to the model's device
        if x.device != model_device:
            x = x.to(model_device)
        if context is not None and context.device != model_device:
            context = context.to(model_device)

        # Keep original input shape check
        input_is_5d = x.ndim == 5
        if input_is_5d:
            x = x.squeeze(2)

        if self.customized_forward:
            with torch.inference_mode():
                return self.customized_forward(
                    self.model,
                    hidden_states=x,
                    encoder_hidden_states=context,
                    timestep=timestep,
                    guidance=guidance if self.config.get("guidance_embed", False) else None,
                    control=control,
                    transformer_options=transformer_options,
                    **self.forward_kwargs,
                    **kwargs,
                )
        else:
            with torch.inference_mode():
                return self.model(
                    hidden_states=x,
                    encoder_hidden_states=context,
                    timestep=timestep,
                    guidance=guidance if self.config.get("guidance_embed", False) else None,
                    control=control,
                    transformer_options=transformer_options,
                    **kwargs,
                )