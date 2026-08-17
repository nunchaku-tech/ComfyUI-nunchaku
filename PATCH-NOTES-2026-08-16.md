# Nunchaku Patch Notes — 2026-08-16

## Background

ComfyUI (this install) was hanging again: the web server loaded on `:8188`, but
generation would freeze during the first sampling step of a Qwen-Image workflow.

## Root Cause

During sampling, the `ComfyUI-QwenImageLoraLoader` wrapper rebuilds the CPU
offload manager inside `forward()`. `CPUOffloadManager.__init__` created its two
ping-pong GPU scratch blocks with:

```python
self.buffer_blocks = [copy.deepcopy(blocks[0]), copy.deepcopy(blocks[0])]
```

That is a **GPU-tensor deepcopy executed mid-forward-pass**, which deadlocked.
Diagnostic evidence (py-spy dump of hung process):
- `prompt_worker` thread stuck at `torch.nn.parameter.Parameter.__deepcopy__`
  -> `copy.deepcopy` -> `CPUOffloadManager.__init__` (`nunchaku/models/utils.py:115`)
  -> `set_offload` (`ComfyUI-nunchaku/models/qwenimage.py:900`)
- CPU idle (~0 CPU sec over 8s), GPU mostly free (15.9 / 17.1 GB), so it was a
  deadlock, not slow compute.

## Fix Applied (manual, in installed site-packages)

File: `C:\Tavern\ComfyUI\.venv132\Lib\site-packages\nunchaku\models\utils.py`
(around line 115, inside `CPUOffloadManager.__init__`)

```python
# Build the ping-pong GPU scratch blocks. Deepcopy on CPU (not GPU) to
# avoid a GPU-tensor deepcopy deadlock when this manager is constructed
# inside a forward pass. set_device() relocates the blocks afterwards.
first_param = next(blocks[0].parameters(), None)
if first_param is not None and first_param.is_cuda:
    blocks[0].to("cpu")
self.buffer_blocks = [copy.deepcopy(blocks[0]), copy.deepcopy(blocks[0])]
```

Why this is safe: `set_device()` (called right after) moves the buffer blocks and
`blocks[0]` (index 0 < `num_blocks_on_gpu`) back onto the target CUDA device, so
the resulting structure is identical — just no GPU deepcopy during forward.

## IMPORTANT — This Patch Is NOT in Git

`nunchaku` is installed as a wheel in `site-packages`; it is not a git repo and is
not tracked by the `ComfyUI-nunchaku` repository. **Any reinstall of nunchaku will
overwrite this fix.** It must be re-applied manually, or applied to the nunchaku
source fork and rebuilt.

## Other Pending Items

1. **`is_dynamic()` fix (committed, NOT pushed):**
   Commit `1340f9a` on branch `fix/compat-update-2026` changes
   `model_patcher/common.py` so `NunchakuModelPatcher.is_dynamic()` returns `False`
   (prevents ComfyUI `reset_cast_buffers` from creating HostBuffer objects that
   corrupt the heap on Windows). **Push is blocked** — the remote
   `https://github.com/LacklusterOpsec/ComfyUI-nunchaku.git` rejects the local
   `YenLegion` credentials (HTTP 403). Needs a token/fork with push access.

2. **Possible second deadlock suspect:** the first thread dump showed the main
   asyncio thread inside `torch.cuda.memory_stats()` called by the
   `ComfyUI-MemoryVisualization` node (`aimdo_vram_status`). If hangs persist
   after the deepcopy fix, investigate that node's VRAM polling (GIL/CUDA-lock
   deadlock).
