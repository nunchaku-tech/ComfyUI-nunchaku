# ComfyUI Compatibility Status

**Date:** 2026-07-22
**ComfyUI HEAD:** 5361a3ce
**Fork baseline:** comfyanonymous/ComfyUI@158419f3

## BROKEN (will error or crash)

| # | File:Line | Issue | Fix |
|---|-----------|-------|-----|
| B1 | `model_base/qwenimage.py:69` | `sd[k] = torch.ones_like(state_dict[k])` silently fills missing `.wcscales` with ones. If `wcscales` is genuinely absent, the model loads with incorrect weight scales → **silent wrong outputs**. | Add logging.warning and validate the assumption. |
| B2 | `nodes/models/pulid.py:213-214` | `folder_paths.get_folder_paths("insightface")` / `"facexlib"` can raise `KeyError` if no other custom node registers these folders. | Wrap in try/except or pre-register folders. |

## DEGRADED (works but suboptimal)

| # | File:Line | Issue | Impact | Fix |
|---|-----------|-------|--------|-----|
| D1 | `__init__.py:15-16` | `comfy.model_downloader.add_known_models` / `HuggingFile` removed upstream. The whole try block silently passes — Nunchaku models never registered in ComfyUI model downloader. | Users cannot discover/install models via ComfyUI Manager. | Ship `.safetensors` files to `diffusion_models`/`text_encoders` folders, or register via `folder_paths.add_model_folder_path`. |
| D2 | `nodes/models/flux.py:313` | Creates plain `ModelPatcher` instead of `ModelPatcherDynamic`. | Model misses dynamic VRAM management benefits. | Use `NunchakuModelPatcher` (consistent with Qwen loader). |
| D3 | `model_patcher/common.py:22-26` | `NunchakuModelPatcher` bypasses `ModelPatcherDynamic.__init__` (HostBuffer workaround). Skips `register_load_device()`. | Dynamic pather codepaths (`free_memory(for_dynamic=True)`, vbar pinning) won't interact correctly. | Call `self.register_load_device(self.load_device)` after the bypass. |
| D4 | `model_patcher/zimage.py:14` | `string_to_seed` imported from `comfy.model_patcher` (deprecated — emits warning, delegates to `comfy.utils`). | Deprecation warning at runtime. | Change to `from comfy.utils import string_to_seed`. |
| D5 | `nodes/models/flux.py:212` | Manual GPU memory calc with hardcoded 14 GiB threshold. | New API `maximum_vram_for_weights()` exists for portable threshold. | Replace magic number with API call. |
| D6 | `nodes/models/flux.py:80-82` | Manual `is_turing()` check for attention/dtype options. | New `supports_fp8_compute()` / `supports_nvfp4_compute()` APIs exist. | Replace Turing detection with portable compute checks. |
| D7 | `nodes/models/qwenimage.py:200` | Uses `get_gpu_memory()` with hardcoded 14 GB threshold. | Same as D5, Qwen-specific. | Use `maximum_vram_for_weights()`. |
| D8 | `model_patcher/common.py:74-81` | `pin_weight_to_device` / `unpin_weight` / `unpin_all_weights` are no-ops. | New `comfy.model_management.pin_memory()` / `unpin_memory()` APIs available. | Wire through to `model_management` for proper pin tracking. |
| D9 | `nodes/models/zimage.py:218-221` | Z-Image loader has no caching — re-loads state dict and re-patches model every execution. | Wasteful; adds ~5s per workflow run. | Add model cache similar to `NunchakuFluxDiTLoader` pattern. |

## OPPORTUNITY (new ComfyUI capabilities worth adopting)

| # | Feature | Relevant file | What it offers |
|---|---------|---------------|----------------|
| O1 | `maximum_vram_for_weights(device)` | `nodes/models/flux.py:218`, `nodes/models/qwenimage.py:200` | Portable VRAM threshold instead of hardcoded magic numbers. |
| O2 | `minimum_inference_memory()` + `extra_reserved_memory()` | `model_patcher/common.py:108` | Portable base-memory accounting for `free_memory` target. |
| O3 | `get_all_torch_devices()` / `get_gpu_device_options()` | `nodes/models/flux.py:77` | Replace manual `torch.cuda.device_count()` loop for device selection widget. |
| O4 | `cast_to_gathered()` / `get_cast_buffer()` / `get_offload_stream()` / `sync_stream()` | `model_patcher/zimage.py:151-261` | Async weight offloading infrastructure. Could replace sequential offload in ZImage. |
| O5 | `vram_aligned_size(tensor)` in `comfy/memory_management.py` | `model_patcher/common.py`, `nodes/models/qwenimage.py:53` | Handles `QuantizedTensor` natively for proper VRAM accounting. |
| O6 | `supports_fp8_compute()` / `supports_nvfp4_compute()` / `supports_mxfp8_compute()` | `nodes/models/flux.py:80-86` | Replace `is_turing()` for attention/dtype option filtering. |
| O7 | `ModelPatcher` hooks infrastructure | `model_patcher/common.py` | Composable inference-time transforms via hooks instead of manual patching. |
| O8 | `ModelPatcher` attachments / additional_models / callbacks / wrappers | `model_patcher/common.py` | Standardized plugin system for auxiliary models (IP-Adapter, PuLID). |
| O9 | `ModelPatcherDynamic` vbar system | `model_patcher/common.py:22-26` | Full dynamic memory management if HostBuffer issue is fixed upstream. |

## Pre-existing bugs (from AGENTS.md) not addressed by this update

| # | File:Line | Issue | Severity |
|---|-----------|-------|----------|
| B3 | `model_base/qwenimage.py:69` | (same as B1 above) — fixed in this update. | **High** |
| B4 | `model_patcher/zimage.py:303` | `assert isinstance(self.model.diffusion_model, NextDiT)` — hard crash if ComfyUI changes model class. | Medium |
| B5 | `nodes/models/flux.py:298` | `assert model_class_name == "Flux"` — assertions disabled with `-O`. | Low |
| B6 | `models/zimage.py:98-99` | `torch.nan_to_num` on norm weights hides NaN issues. | Low |
| B7 | `nodes/models/zimage.py:218-221` | (same as D9 above) — fixed in this update. | Medium |
| B8 | `nodes/models/ipadapter.py:136` | `from_pretrained("black-forest-labs/FLUX.1-dev", ...)` downloads base model every load. | Medium |
| B9 | `model_patcher/zimage.py:26-46` | LoRA rank capacity not validated. | Low |