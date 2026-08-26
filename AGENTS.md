# ComfyUI-nunchaku — Agent Memory

## Project Structure

```
__init__.py              # Plugin entry: log config, version check, node registration (8 try/except blocks)
utils.py                 # Helper: get_package_version, get_plugin_version (reads pyproject.toml)
pyproject.toml           # v1.2.1, setuptools build, uv config, ruff/isort/black

model_base/
├── __init__.py          # Re-exports NunchakuQwenImage
└── qwenimage.py         # QwenImage subclass, loads SVDQW4A4Linear weights, fills missing wcscales with ones

model_configs/
├── __init__.py          # Empty
├── qwenimage.py         # NunchakuQwenImage(QwenImage) — just returns NunchakuQwenImage model
└── zimage.py            # NunchakuZImage(ZImageModelConfig) — patches NextDiT via patch_model()

model_patcher/
├── __init__.py          # Empty
├── common.py            # NunchakuModelPatcher — bypasses HostBuffer on Windows, custom load/detach
└── zimage.py            # ZImageModelPatcher — LoRA for SVDQW4A4Linear, svdq_backup ops

models/
├── __init__.py          # Empty
├── qwenimage.py         # Nunchaku Qwen-Image Attention/FF/Transformer2D (915 lines)
└── zimage.py            # NextDiT patching: fused QKV+RoPE, attention/FF replacement (328 lines)

wrappers/
├── __init__.py          # Empty
└── flux.py              # ComfyFluxWrapper — wraps NunchakuFluxTransformer2dModel (371 lines)

nodes/
├── __init__.py          # Empty
├── utils.py             # Re-exports get_filename_list, get_full_path_or_raise from folder_paths
├── lora/
│   └── flux.py          # NunchakuFluxLoraLoader, NunchakuFluxLoraStack (271 lines)
├── models/
│   ├── __init__.py      # Empty
│   ├── utils.py         # set_extra_config_model_path helper
│   ├── flux.py          # NunchakuFluxDiTLoader (314 lines)
│   ├── qwenimage.py     # NunchakuQwenImageDiTLoader (231 lines)
│   ├── zimage.py        # NunchakuZImageDiTLoader, state_dict patching (223 lines)
│   ├── text_encoder.py  # NunchakuTextEncoderLoaderV2, T5 forward wrapper (413 lines)
│   ├── ipadapter.py     # NunchakuIPAdapterLoader, NunchakuFluxIPAdapterApply (217 lines)
│   ├── pulid.py         # NunchakuPuLIDLoaderV2, NunchakuFluxPuLIDApplyV2 (230 lines)
│   └── configs/         # ComfyUI model config JSONs
├── tools/
│   ├── installers.py    # NunchakuWheelInstaller — pip/uv wheel install (464 lines)
│   └── merge_safetensors.py  # NunchakuModelMerger (107 lines)
└── preprocessors/       # Empty (no __init__.py content)

mixins/
└── model.py             # NunchakuModelMixin — offload flag, to_safely() guard
```

## 📋 Known Issues

### Bugs / Correctness

| # | File | Line(s) | Issue | Severity |
|---|------|---------|-------|----------|
| B1 | `__init__.py` | 44-47 | Broad `except (ImportError, ModuleNotFoundError): pass` swallows all compatibility imports silently — if `comfy_compatibility` partially imports, no logging | Low |
| B2 | `nodes/models/flux.py` | 231-236 | Model cache guard (`self.model_path != model_path or ...`) doesn't include `attention` or `cache_threshold`. User changing these without changing model path re-applies `apply_cache_on_transformer` + `set_attention_impl` on stale model, but the model won't be reloaded. If `NunchakuFluxTransformer2dModel` maintains internal state from init that affects attention, this is a correctness bug. | Medium |
| B3 | `model_base/qwenimage.py` | 69 | `sd[k] = torch.ones_like(state_dict[k])` silently fills missing `.wcscales` keys with ones. If `wcscales` is genuinely missing (not just excluded from state dict for size), the model loads with incorrect weight scales — silent wrong outputs. | **High** |
| B4 | `model_patcher/zimage.py` | 303 | `assert isinstance(self.model.diffusion_model, NextDiT)` — hard crash if ComfyUI changes model class. Use duck-typing or `try/except` instead. | Medium |
| B5 | `nodes/models/flux.py` | 298 | `assert model_class_name == "Flux", f"Unknown model class {model_class_name}."` — assertions can be disabled with `-O` flag. Should be `raise ValueError`. | Low |
| B6 | `models/zimage.py` | 98-99 | `torch.nan_to_num` on norm weights hides NaN issues. If quantized weights produce NaN during dtype conversion, this silently papers over it. | Low |
| B7 | `nodes/models/zimage.py` | 218-221 | Z-Image loader has no caching at all — every workflow execution re-loads the state dict and re-patches the model. | Medium |
| B8 | `nodes/models/ipadapter.py` | 136 | `from_pretrained("black-forest-labs/FLUX.1-dev", ...)` downloads base model every load if not HF-cached. Adds ~30s latency per workflow run. | Medium |
| B9 | `model_patcher/zimage.py` | 26-46 | `apply_lora_to_svdq_linear` applies LoRA by concatenating to `proj_down`/`proj_up` but never checks if the concatenation exceeds the module's designed rank capacity. No upper bound validation. | Low |

### Performance

| # | File | Line(s) | Issue | Impact |
|---|------|---------|-------|--------|
| P1 | `wrappers/flux.py` | 214-243 | `if self.loras != model.comfy_lora_meta_list` runs on every `forward()` call. When it triggers (LoRA change), it calls `load_state_dict_in_safetensors` + `compose_lora`. Should be gated to run only once per LoRA update. | Medium (only affects LoRA-heavy workflows) |
| P2 | `wrappers/flux.py` | 269-322 | Both caching and non-caching forward paths are duplicated (~25 lines each) with identical model call. Only difference is `with cache_context(...)` wrapper. | Low (code quality, 0.5% perf from duplication) |
| P3 | `nodes/models/flux.py` | 259-261 | `apply_cache_on_transformer` called on every invocation regardless of whether threshold changed. If cache_threshold=0, it still applies the cache transform every time. | Low (cheap operation, but unnecessary) |
| P4 | `nodes/models/flux.py` | 242 | `del transformer` after moving to CPU is a no-op (variable goes out of scope). | Negligible |

### Code Quality / Maintainability

| # | File | Line(s) | Issue | Recommendation |
|---|------|---------|-------|----------------|
| Q1 | `__init__.py` | 84-147 | 8 nearly-identical try/except blocks for node imports. ~50 lines of repetition. | DRY with helper: `_register_node(name, import_path)` |
| Q2 | `nodes/utils.py` | 11-12 | `get_filename_list = get_filename_list` — self-reassignment no-ops | Delete lines 11-12 |
| Q3 | `nodes/tools/installers.py` | ~55,58,71,75,82,85,121,291,433 | 10+ `print()` calls instead of logger | Replace all with `logger.info`/`logger.warning`/`logger.error` |
| Q4 | `nodes/tools/installers.py` | 383 | `global VERSION_CONFIG, OFFICIAL_VERSIONS, DEV_VERSIONS` — mutable module globals | Use a Config dataclass or module-level Singleton |
| Q5 | `nodes/models/text_encoder.py` | 354 | `EmptyClass` defined inside `load_text_encoder_state_dicts` — re-created on every call | Move to module level |
| Q6 | `model_patcher/zimage.py` | 151-261 | `partially_unload` is 110 lines with deep nesting | Split into `_try_unload_module`, `_handle_svdq_unload`, `_apply_lowvram_patches` |
| Q7 | `model_patcher/zimage.py` | 293-430 | `patch_weight_to_device` is 140 lines with 3 deep branches (qkv/w13/other) | Extract `_patch_qkv`, `_patch_fused_ff`, `_patch_single` methods |
| Q8 | `models/qwenimage.py` | 684-874 | `_forward` is 190 lines — handles ref_latents, ControlNet, offload streaming, patch replacement | Break into `_prepare_hidden_states`, `_process_blocks`, `_finalize_output` |
| Q9 | `nodes/models/flux.py` line 218 vs `nodes/models/qwenimage.py` line 200 | GPU memory threshold: one uses `total_memory / (1024**2)` in MiB, the other uses `get_gpu_memory()` in GB. Different API, same intent (~14 GB). | Inconsistent — confusing for maintainers |
| Q10 | `nodes/models/flux.py` | 287 | `comfy_config = json.loads(comfy_config_str)` — no error handling if metadata has malformed JSON | Add try/except with fallback to config file |
| Q11 | `nodes/models/flux.py` | 303-313 | Creates `ComfyFluxWrapper` directly but other nodes (lora/pulid) use `copy_with_ctx`. Inconsistent patterns. | Either use `copy_with_ctx` everywhere or document the exception |
| Q12 | `mixins/model.py` | 37-96 | `to_safely` method has complex type-checking logic for args. Could be simplified with `torch.is_tensor`-style checks. | Refactor device detection |

### Missing Features / UX

| # | Description |
|---|-------------|
| F1 | `NunchakuFluxDiTLoader` doesn't expose `i2f_mode` parameter usage — it's accepted via `**kwargs` but never consumed in `load_model`. Dead parameter. |
| F2 | No node-level input validation for `device_id` — `max` bound is `ngpus - 1` but if `ngpus == 0` (CPU-only), `torch.cuda.device_count()` returns 0 and the dropdown breaks (max < min). |
| F3 | `NunchakuPuLIDApplyV2` raises `NotImplementedError` for `attn_mask` but takes it as optional input — users won't know it's unsupported until runtime. |

## 🎯 Priority Fix Order

1. **B3** — Silent wrong outputs from missing `wcscales` 
2. **B7** — Z-Image loads model on every execution (wasteful)
3. **B5/Q9** — Assertion safety + inconsistent GPU memory detection
4. **Q3** — Replace `print()` with logger in installer
5. **Q2** — Remove dead reassignments in `nodes/utils.py`
6. **Q1** — DRY node registration in `__init__.py`
7. **P1** — LoRA re-composition on every forward
8. **B9** — Missing rank capacity check in LoRA apply

## 🔧 Dev Commands

```bash
# Lint
ruff check . --fix

# Format
black -l 120 .

# Sort imports
isort --profile black -l 120 .

# Pre-commit
pre-commit run --all-files

# Test (if CI env available)
pytest tests/
```

## 📐 Conventions

- **Line length**: 120
- **Formatter**: black
- **Imports**: isort with `profile = "black"`, known first-party: `nunchaku`
- **Linter**: ruff
- **Python**: 3.10–3.13
- **Nunchaku min version**: 1.0.0
- **Model patcher base**: prefers `ModelPatcherDynamic`, falls back to `ModelPatcher`
- **Logging**: `logging.getLogger(__name__)` — use `logger.info/warning/error/exception`, never `print()`
- **Docstrings**: NumPy-style with Parameters/Returns/Raises sections

## 🔗 Key External Dependencies

| Package | Min Version | Used In |
|---------|-------------|---------|
| nunchaku | 1.0.0 | Core quantized ops, transformer models |
| diffusers | 0.35 | IP-Adapter pipeline |
| transformers | 4.54 | Text encoders |
| peft | 0.17 | LoRA adapter handling |
| accelerate | 1.10 | Device management |
| comfy-ui | (git dep) | ModelPatcher, model_base, nodes API |

## 🧠 Architecture Notes

- **FLUX models** use `NunchakuFluxTransformer2dModel` wrapped in `ComfyFluxWrapper` → model patcher stores it as `diffusion_model`
- **Qwen-Image models** use custom `NunchakuQwenImageTransformer2DModel` (no wrapper) with `NunchakuModelPatcher`
- **Z-Image models** patch ComfyUI's native `NextDiT` by replacing attention/FF modules with `SVDQW4A4Linear`-based versions, and use `ZImageModelPatcher`
- **LoRA for FLUX** works via `compose_lora()` at forward time (in `ComfyFluxWrapper`)
- **LoRA for Z-Image** works via `concat_lora_weights()` + `apply_lora_to_svdq_linear()` at patch time (in `ZImageModelPatcher`)
- **GPU offload**: FLUX uses nunchaku's built-in offload; Qwen-Image uses `CPUOffloadManager`; Z-Image uses ComfyUI's native `partially_unload`
- **Cache**: Only FLUX has first-block cache (`residual_diff_threshold`). Z-Image and Qwen-Image have no caching.
