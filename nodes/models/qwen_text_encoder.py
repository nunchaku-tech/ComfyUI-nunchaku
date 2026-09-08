"""
Nunchaku Qwen text encoder nodes for ComfyUI.

Provides quantized text encoder nodes for:
  - FLUX.2 Klein:        Qwen3-4B / Qwen3-8B          (NunchakuQwen3TextEncoderLoader)
  - QwenImage text-only: Qwen2.5-VL text checkpoint    (NunchakuQwenImageTextEncoderLoader)
  - QwenImage edit:      Qwen2.5-VL edit checkpoint    (NunchakuQwenImageEditEncoderLoader)

All three nodes build a standard ComfyUI ``CLIP`` object whose
``encode_token_weights`` is backed by the nunchaku quantised encoder.
"""

from __future__ import annotations

import logging
import numbers
import os
from typing import Any

import torch
import torch.nn as nn

import comfy.model_management
import comfy.model_patcher
import comfy.sd
from comfy.text_encoders.flux import KleinTokenizer, KleinTokenizer8B
from comfy.text_encoders.qwen_image import QwenImageTokenizer
import comfy.text_encoders.qwen_vl as _qwen_vl_utils

from nunchaku.models.text_encoders.qwen_encoder import NunchakuQwenEncoderModel
from nunchaku.utils import is_turing

import folder_paths as _folder_paths
from ..utils import get_filename_list, get_full_path_or_raise

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Qwen2.5-VL special token IDs used by the vision pathway
_VL_IMAGE_PAD = 151655    # <|image_pad|>  – marks vision positions in input_ids
_VL_MERGE_SIZE = 2        # spatial merge factor in Qwen2.5-VL patch merger


def _encoder_byte_size(encoder: Any) -> int:
    """
    Estimate the GPU memory footprint of a nunchaku encoder so ComfyUI's
    ModelPatcher gets a non-zero ``model_size()`` for VBAR reservation.

    Counts registered parameters **and** registered buffers (nunchaku may
    store quantised weight packs as buffers rather than parameters).
    Returns at least 1 so that ``ModelVBAR(model_size() * 10, ...)`` never
    receives 0.
    """
    total = sum(p.numel() * p.element_size() for p in encoder.parameters())
    total += sum(b.numel() * b.element_size() for b in encoder.buffers())
    return max(total, 1)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Shared helpers
# ===========================================================================


def _get_nunchaku_device(encoder: Any) -> torch.device:
    """Return the device the nunchaku encoder's first parameter lives on."""
    try:
        return next(encoder.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _simple_token_pairs_to_ids(
    token_pairs_list: list[list[tuple]],
    pad_token: int = 151643,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Convert ComfyUI ``token_weight_pairs`` batches to flat integer lists.

    Returns ``(input_ids_list, attention_mask_list)`` where each inner list
    corresponds to one batch row (already padded to the same length).

    Only integer token IDs are supported; image-dict tokens are replaced with
    the pad token (image conditioning through the VL encoder is not yet
    implemented in the nunchaku path).
    """
    if not token_pairs_list:
        return [], []

    max_len = max(len(batch) for batch in token_pairs_list)
    input_ids_list: list[list[int]] = []
    attention_mask_list: list[list[int]] = []

    for batch in token_pairs_list:
        ids: list[int] = []
        for pair in batch:
            elem = pair[0]
            if isinstance(elem, numbers.Integral):
                ids.append(int(elem))
            elif torch.is_tensor(elem):
                ids.append(int(elem.item()))
            else:
                # Image dict or other embedded type – not supported in text-only path
                ids.append(pad_token)

        pad_len = max_len - len(ids)
        mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [pad_token] * pad_len
        input_ids_list.append(ids)
        attention_mask_list.append(mask)

    return input_ids_list, attention_mask_list


# ===========================================================================
# FLUX.2 Klein  –  Qwen3 text encoder
# ===========================================================================


def _make_klein_te_class(encoder: Any, model_type: str, layers: list[int]) -> type:
    """
    Return a ``torch.nn.Module`` class whose ``encode_token_weights`` produces
    the stacked-3-layer output expected by ``Flux2TEModel.encode_token_weights``.

    ``encoder``    – pre-loaded ``NunchakuQwen3TextEncoderModel`` instance
    ``model_type`` – ``"qwen3_4b"`` or ``"qwen3_8b"``
    ``layers``     – which hidden-state indices to stack (default: [9, 18, 27])
    """
    _encoder = encoder
    _token_key = model_type
    _layer_indices = layers
    _pad_token = 151643

    class NunchakuKleinTEModel(nn.Module):
        def __init__(self, device="cpu", dtype=None, model_options={}):  # noqa: B006
            super().__init__()
            # Store as a plain Python attribute rather than a registered child
            # nn.Module, so ComfyUI's ModelPatcher does NOT enumerate the
            # nunchaku encoder's parameters as weights it should manage.
            # The nunchaku encoder handles its own CUDA memory; letting ComfyUI
            # walk its ~459 tensors on every VRAM cycle wastes time and risks
            # moving quantized weight formats that nunchaku owns.
            self._encoder_device = _get_nunchaku_device(_encoder)
            object.__setattr__(self, 'nunchaku_encoder', _encoder)
            self.dtypes = {dtype}

        # ---- ComfyUI CLIP interface ----------------------------------------

        def set_clip_options(self, options):
            pass

        def reset_clip_options(self):
            pass

        def load_sd(self, sd):
            # Nunchaku model weights are already loaded; nothing to do.
            return [], []

        # ---- Core encoding -------------------------------------------------

        def encode_token_weights(self, token_weight_pairs: dict) -> tuple:
            """
            Parameters
            ----------
            token_weight_pairs : dict
                Output of ``KleinTokenizer.tokenize_with_weights``.
                Key is ``"qwen3_4b"`` or ``"qwen3_8b"``.

            Returns
            -------
            (out, pooled, extra) where
            ``out``    has shape ``(1, T, D*3)``  – 3 intermediate layers stacked,
            ``pooled`` is ``None``,
            ``extra``  is ``{}`` or ``{"attention_mask": tensor}``.
            """
            tok_pairs_list = token_weight_pairs[_token_key]
            ids_list, masks_list = _simple_token_pairs_to_ids(tok_pairs_list, _pad_token)

            model_device = self._encoder_device
            # Create directly on the target device to avoid a CPU-alloc + copy.
            input_ids = torch.tensor(ids_list, dtype=torch.long, device=model_device)
            attn_mask = torch.tensor(masks_list, dtype=torch.long, device=model_device)

            # Use forward hooks to capture only the 3 required intermediate hidden
            # states (at 1-indexed positions _layer_indices) without forcing the
            # model to allocate and return ALL N intermediate tensors via
            # output_hidden_states=True (saves ~70-180 MB for Qwen3-4B/8B).
            captured: dict[int, torch.Tensor] = {}
            hooks: list = []
            use_fallback = True
            try:
                layers = self.nunchaku_encoder.model.layers
                for k, layer_num in enumerate(_layer_indices):
                    def _hook(m, inp, out, _k=k):
                        captured[_k] = (out[0] if isinstance(out, tuple) else out).detach()
                    hooks.append(layers[layer_num - 1].register_forward_hook(_hook))
                use_fallback = len(hooks) != len(_layer_indices)
            except (AttributeError, IndexError):
                use_fallback = True

            try:
                with torch.inference_mode():
                    outputs = self.nunchaku_encoder(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        output_hidden_states=use_fallback,
                        return_dict=True,
                    )
            finally:
                for h in hooks:
                    h.remove()

            if use_fallback:
                # hidden_states[k] = output after layer k-1 (1-indexed).
                hs = outputs.hidden_states
                h_tensors = [hs[i] for i in _layer_indices]
            else:
                h_tensors = [captured[k] for k in range(len(_layer_indices))]

            # Stack in native dtype (bfloat16), single float32 upcast at end.
            stacked = torch.stack(h_tensors, dim=1)    # (B, 3, T, D)
            out = stacked.movedim(1, 2)                 # (B, T, 3, D)
            out = out.reshape(out.shape[0], out.shape[1], -1).float()  # (B, T, D*3)

            inter_device = comfy.model_management.intermediate_device()
            out = out.to(inter_device)
            attn_mask = attn_mask.to(inter_device)

            extra: dict = {}
            if attn_mask.sum().item() != attn_mask.numel():
                extra["attention_mask"] = attn_mask[:1]

            return out, None, extra

    return NunchakuKleinTEModel


def _build_klein_clip(encoder: Any, folder_paths_embeddings: list[str]) -> comfy.sd.CLIP:
    """
    Build a ``comfy.sd.CLIP`` backed by the pre-loaded nunchaku Qwen3 encoder.

    The model type (4B vs 8B) and corresponding tokenizer are auto-detected
    from the checkpoint metadata.
    """
    meta = encoder.metadata.base_text_config
    num_layers = int(meta.get("num_hidden_layers", 28))

    if num_layers >= 36:
        model_type = "qwen3_8b"
        TokenizerClass = KleinTokenizer8B
    else:
        model_type = "qwen3_4b"
        TokenizerClass = KleinTokenizer

    # Klein uses intermediate layers [9, 18, 27] regardless of model size
    layer_indices = [9, 18, 27]

    TEModelClass = _make_klein_te_class(encoder, model_type, layer_indices)
    enc_bytes = _encoder_byte_size(encoder)

    class _ClipTarget:
        clip = TEModelClass
        tokenizer = TokenizerClass
        params: dict = {}

    embedding_dirs = folder_paths_embeddings
    clip = comfy.sd.CLIP(
        target=_ClipTarget(),
        embedding_directory=embedding_dirs,
        parameters=enc_bytes,
        state_dict=[],
    )
    # `parameters` in CLIP.__init__ only affects initial device selection, not
    # ModelPatcher.size.  Set it explicitly so model_size() returns a non-zero
    # value and ModelVBAR(model_size() * 10, ...) does not fail.
    clip.patcher.size = enc_bytes
    return clip


class NunchakuQwen3TextEncoderLoader:
    """
    Load a nunchaku-quantised Qwen3 text encoder for use with FLUX.2 Klein models.

    Supports both Qwen3-4B and Qwen3-8B checkpoints; the variant is detected
    automatically from the checkpoint metadata.
    """

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_text_encoder"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Qwen3 Text Encoder Loader (FLUX.2 Klein)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    get_filename_list("text_encoders"),
                    {
                        "tooltip": (
                            "Nunchaku-quantised Qwen3 text encoder checkpoint "
                            "(qwen3-4b or qwen3-8b, .safetensors)."
                        )
                    },
                ),
            }
        }

    def load_text_encoder(self, model_path: str) -> tuple:
        path = get_full_path_or_raise("text_encoders", model_path)

        load_device = comfy.model_management.get_torch_device()
        # Turing GPUs (20-series, SM 7.5) do not support bfloat16.
        torch_dtype = torch.float16 if is_turing(load_device) else torch.bfloat16
        logger.info("Loading nunchaku Qwen3 encoder from %s on %s (dtype=%s)", path, load_device, torch_dtype)

        encoder = NunchakuQwenEncoderModel.from_pretrained(
            path,
            device=load_device,
            torch_dtype=torch_dtype,
        )

        emb_dirs = _folder_paths.get_folder_paths("embeddings")
        clip = _build_klein_clip(encoder, emb_dirs)
        logger.info("Nunchaku Qwen3 encoder loaded successfully.")
        return (clip,)


# ===========================================================================
# QwenImage  –  Qwen2.5-VL text encoder  (text-only path)
# ===========================================================================


def _qwenimage_find_template_end(tok_pairs: list[tuple]) -> int:
    """
    Detect where the system + user template prefix ends in a token pair list.

    Replicates the logic from ``QwenImageTEModel.encode_token_weights``:
    finds the 2nd ``<|im_start|>`` (token 151644) and skips the following
    ``user\n`` tokens (IDs 872, 198) to point at the start of user text.

    Returns
    -------
    int
        Index of the first user-text token (0 if detection fails).
    """
    count_im_start = 0
    template_end = -1

    for i, pair in enumerate(tok_pairs):
        elem = pair[0]
        if not torch.is_tensor(elem) and isinstance(elem, numbers.Integral):
            if elem == 151644 and count_im_start < 2:  # <|im_start|>
                template_end = i
                count_im_start += 1

    if template_end >= 0:
        # Skip "<|im_start|>user\n" (3 tokens: 151644, 872, 198)
        if len(tok_pairs) > template_end + 3:
            if (
                tok_pairs[template_end + 1][0] == 872   # "user"
                and tok_pairs[template_end + 2][0] == 198  # "\n"
            ):
                template_end += 3
        return max(template_end, 0)

    return 0  # fallback: keep all tokens


def _make_qwenimage_te_class(encoder: Any) -> type:
    """
    Return a ``torch.nn.Module`` class whose ``encode_token_weights`` produces
    the output expected by the QwenImage diffusion model (last hidden state,
    template-trimmed, with optional attention_mask in extra).

    Works for both text-only and edit checkpoints in text-only mode.
    """
    _encoder = encoder
    _pad_token = 151643

    class NunchakuQwenImageTEModel(nn.Module):
        def __init__(self, device="cpu", dtype=None, model_options={}):  # noqa: B006
            super().__init__()
            # Same rationale as NunchakuKleinTEModel: bypass nn.Module submodule
            # registration so ComfyUI's ModelPatcher ignores the nunchaku params.
            self._encoder_device = _get_nunchaku_device(_encoder)
            object.__setattr__(self, 'nunchaku_encoder', _encoder)
            self.dtypes = {dtype}

        # ---- ComfyUI CLIP interface ----------------------------------------

        def set_clip_options(self, options):
            pass

        def reset_clip_options(self):
            pass

        def load_sd(self, sd):
            return [], []

        # ---- Core encoding -------------------------------------------------

        def encode_token_weights(self, token_weight_pairs: dict) -> tuple:
            """
            Parameters
            ----------
            token_weight_pairs : dict
                Output of ``QwenImageTokenizer.tokenize_with_weights``.
                Key is ``"qwen25_7b"``.

            Returns
            -------
            (out, pooled, extra) where
            ``out``    has shape ``(1, T_trimmed, D)``  – template prefix stripped,
            ``pooled`` is ``None``,
            ``extra``  is ``{}`` or ``{"attention_mask": tensor}``.
            """
            tok_pairs_batch = token_weight_pairs["qwen25_7b"]

            # Detect template end from the first batch item.
            template_end = _qwenimage_find_template_end(tok_pairs_batch[0])

            ids_list, masks_list = _simple_token_pairs_to_ids(tok_pairs_batch, _pad_token)

            model_device = self._encoder_device
            # Create directly on the target device to avoid a CPU-alloc + copy.
            input_ids = torch.tensor(ids_list, dtype=torch.long, device=model_device)
            attn_mask = torch.tensor(masks_list, dtype=torch.long, device=model_device)

            with torch.inference_mode():
                outputs = self.nunchaku_encoder(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

            # Trim the system-prompt template prefix on the source device before
            # transferring to the intermediate device, reducing cross-device bandwidth.
            out = outputs.last_hidden_state[:1, template_end:].float()  # (1, T_trim, D)
            attn_out = attn_mask[:1, template_end:]                      # (1, T_trim)

            inter_device = comfy.model_management.intermediate_device()
            out = out.to(inter_device)
            attn_out = attn_out.to(inter_device)

            extra: dict = {}
            if attn_out.sum().item() != attn_out.numel():
                extra["attention_mask"] = attn_out

            return out, None, extra

    return NunchakuQwenImageTEModel


def _move_cpu_tensors_to_device(root_module: Any, device: "torch.device") -> None:
    """
    Walk every submodule of *root_module* and move any CPU-resident tensors to
    *device*.  Covers two storage locations:

    1. Registered buffers in ``mod._buffers`` (e.g. ``inv_freq`` registered via
       ``register_buffer(..., persistent=False)``).
    2. Plain tensor attributes in ``mod.__dict__`` (e.g. ``original_inv_freq``
       set as a direct assignment and therefore invisible to ``nn.Module``'s
       buffer/parameter machinery).

    Parameters that are already on a CUDA device are left untouched; this is
    safe to call multiple times.
    """
    if device.type != "cuda":
        return
    for mod in root_module.modules():
        # 1. Registered buffers
        for buf_name in list(mod._buffers):
            buf = mod._buffers[buf_name]
            if buf is not None and buf.device.type == "cpu":
                mod._buffers[buf_name] = buf.to(device)
        # 2. Plain tensor attributes in __dict__
        for attr_name, attr_val in list(vars(mod).items()):
            if isinstance(attr_val, torch.Tensor) and attr_val.device.type == "cpu":
                try:
                    # setattr goes through nn.Module.__setattr__; use object
                    # form to bypass potential buffer/param interception.
                    object.__setattr__(mod, attr_name, attr_val.to(device))
                except Exception:
                    pass


def _process_image_tokens(
    tok_pairs_batch: list[list[tuple]],
    pad_token: int,
) -> tuple[list[list[int]], list[list[int]], "torch.Tensor | None", "torch.Tensor | None"]:
    """
    Walk the ComfyUI token_weight_pairs for a QwenImage-edit batch and expand
    any image-dict tokens into the correct number of ``<|image_pad|>`` (151655)
    slots required by Qwen2.5-VL's multimodal forward pass.

    Returns
    -------
    ids_rows      : list[list[int]]  – expanded & length-padded input_ids
    mask_rows     : list[list[int]]  – corresponding attention masks
    pixel_values  : Tensor (total_patches, C*T*P*P) or None
    image_grid_thw: Tensor (n_images, 3)            or None
    """
    all_ids: list[list[int]] = []
    all_pv: list[torch.Tensor] = []
    all_gt: list[torch.Tensor] = []

    for tok_pairs in tok_pairs_batch:
        row_ids: list[int] = []
        for pair in tok_pairs:
            elem = pair[0]
            if isinstance(elem, dict) and elem.get("type") == "image":
                # Extract the raw image tensor – shape (B, H, W, 3) or (H, W, 3).
                img_data: torch.Tensor = elem["data"]
                if img_data.dim() == 3:
                    img_data = img_data.unsqueeze(0)

                # Run through Qwen2.5-VL patch pipeline on CPU.
                flat_patches, grid_thw = _qwen_vl_utils.process_qwen2vl_images(
                    img_data.cpu().float()
                )
                # grid_thw: (1, 3) with values [T, H, W] in patch-grid units.
                t = int(grid_thw[0, 0])
                h = int(grid_thw[0, 1])
                w = int(grid_thw[0, 2])
                # After the spatial merger (merge_size=2) the model produces
                # T * (H//2) * (W//2) visual tokens in the sequence.
                n_merged = t * (h // _VL_MERGE_SIZE) * (w // _VL_MERGE_SIZE)
                row_ids.extend([_VL_IMAGE_PAD] * n_merged)

                all_pv.append(flat_patches)
                all_gt.append(grid_thw[0])     # shape (3,)
            elif isinstance(elem, numbers.Integral):
                row_ids.append(int(elem))
            elif torch.is_tensor(elem):
                row_ids.append(int(elem.item()))
            else:
                row_ids.append(pad_token)
        all_ids.append(row_ids)

    # Pad rows to uniform length.
    max_len = max(len(r) for r in all_ids)
    mask_rows: list[list[int]] = []
    for r in all_ids:
        pad = max_len - len(r)
        mask_rows.append([1] * len(r) + [0] * pad)
        r.extend([pad_token] * pad)

    if all_pv:
        pixel_values = torch.cat(all_pv, dim=0)       # (total_patches, feat_dim)
        image_grid_thw = torch.stack(all_gt, dim=0)   # (n_images, 3)
    else:
        pixel_values = None
        image_grid_thw = None

    return all_ids, mask_rows, pixel_values, image_grid_thw


def _make_qwenimage_edit_te_class(encoder: Any) -> type:
    """
    Return a ``torch.nn.Module`` class that drives a
    ``NunchakuQwen2VLEditEncoderModel`` (the checkpoint that keeps the
    vision tower unquantised).

    When the ComfyUI ``TextEncodeQwenImageEdit`` / ``TextEncodeQwenImageEditPlus``
    node passes image tensors inside the token_weight_pairs (as dicts with
    ``{"type": "image", "data": ...}``), this class:

    1. Expands each image dict into the correct number of ``<|image_pad|>``
       tokens using Qwen2.5-VL's dynamic-resolution patch pipeline.
    2. Calls the nunchaku encoder with ``pixel_values`` + ``image_grid_thw``
       so the vision ViT encodes the image and the LLM attends to the visual
       features.
    3. Strips the system/user template prefix and returns ``(out, None, extra)``
       identical in shape to the text-only path.

    Falls back to text-only mode automatically when no image dicts are present.
    """
    _encoder = encoder
    _pad_token = 151643

    # Fix: Qwen2.5-VL's rotary embeddings are created with register_buffer(
    # "inv_freq", ..., persistent=False) inside __init__, so they live on CPU
    # even when the model is loaded on CUDA.  accelerate's init_empty_weights
    # context does not make non-persistent buffers meta (include_buffers=False
    # default), so materialize_meta_module_tensors never touches them.
    #
    # Additionally, the transformers line
    #   self.original_inv_freq = self.inv_freq
    # stores a plain Python-level tensor in __dict__ (not in _buffers), so a
    # plain buffer loop is insufficient.
    #
    # Calling _move_cpu_tensors_to_device here, before the TE class is
    # instantiated, ensures the correction runs exactly once at load time.
    _enc_device = _get_nunchaku_device(_encoder)
    _move_cpu_tensors_to_device(_encoder.model, _enc_device)

    class NunchakuQwenImageEditTEModel(nn.Module):
        def __init__(self, device="cpu", dtype=None, model_options={}):  # noqa: B006
            super().__init__()
            self._encoder_device = _enc_device
            object.__setattr__(self, 'nunchaku_encoder', _encoder)
            self.dtypes = {dtype}

        # ---- ComfyUI CLIP interface ----------------------------------------

        def set_clip_options(self, options):
            pass

        def reset_clip_options(self):
            pass

        def load_sd(self, sd):
            return [], []

        # ---- Core encoding -------------------------------------------------

        def encode_token_weights(self, token_weight_pairs: dict) -> tuple:
            """
            Parameters
            ----------
            token_weight_pairs : dict
                Output of ``QwenImageTokenizer.tokenize_with_weights``,
                key ``"qwen25_7b"``.  Image dicts may be present when the
                caller used ``clip.tokenize(prompt, images=[img])``.

            Returns
            -------
            (out, pooled, extra) where
            ``out``    – ``(1, T_trimmed, D)`` float32
            ``pooled`` – None
            ``extra``  – ``{}`` or ``{"attention_mask": tensor}``
            """
            tok_pairs_batch = token_weight_pairs["qwen25_7b"]

            # Template_end computed from ORIGINAL pairs (before image expansion).
            # Because the template prefix (system + <|im_start|>user\n) contains
            # only integer tokens – no image dicts – the index is identical in
            # the expanded input_ids sequence.
            template_end = _qwenimage_find_template_end(tok_pairs_batch[0])

            model_device = self._encoder_device

            ids_rows, mask_rows, pixel_values, image_grid_thw = _process_image_tokens(
                tok_pairs_batch, _pad_token
            )
            input_ids = torch.tensor(ids_rows, dtype=torch.long, device=model_device)
            attn_mask = torch.tensor(mask_rows, dtype=torch.long, device=model_device)

            enc_kwargs: dict = dict(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=False,
                return_dict=True,
            )
            if pixel_values is not None:
                # Vision-conditioned forward: pass pixel data to the ViT.
                enc_kwargs["pixel_values"] = pixel_values.to(model_device, dtype=torch.float32)
                enc_kwargs["image_grid_thw"] = image_grid_thw.to(model_device)

            # Safety-net: if anything between load time and the first
            # encode call (e.g. ComfyUI VRAM management) reset rotary buffers
            # back to CPU, fix them again right before the forward pass.
            if not getattr(self, '_inv_freq_fixed', False):
                _move_cpu_tensors_to_device(
                    self.nunchaku_encoder.model, self._encoder_device
                )
                object.__setattr__(self, '_inv_freq_fixed', True)

            with torch.inference_mode():
                outputs = self.nunchaku_encoder(**enc_kwargs)

            # Trim template prefix; trim + upcast on the model device to avoid
            # moving a large bf16 tensor cross-device before truncation.
            out = outputs.last_hidden_state[:1, template_end:].float()  # (1, T_trim, D)
            attn_out = attn_mask[:1, template_end:]                      # (1, T_trim)

            inter_device = comfy.model_management.intermediate_device()
            out = out.to(inter_device)
            attn_out = attn_out.to(inter_device)

            extra: dict = {}
            if attn_out.sum().item() != attn_out.numel():
                extra["attention_mask"] = attn_out

            return out, None, extra

    return NunchakuQwenImageEditTEModel


def _build_qwenimage_clip(encoder: Any, folder_paths_embeddings: list[str]) -> comfy.sd.CLIP:
    """Build a ``comfy.sd.CLIP`` backed by a nunchaku Qwen2.5-VL text encoder (text-only)."""
    TEModelClass = _make_qwenimage_te_class(encoder)
    enc_bytes = _encoder_byte_size(encoder)

    class _ClipTarget:
        clip = TEModelClass
        tokenizer = QwenImageTokenizer
        params: dict = {}

    clip = comfy.sd.CLIP(
        target=_ClipTarget(),
        embedding_directory=folder_paths_embeddings,
        parameters=enc_bytes,
        state_dict=[],
    )
    clip.patcher.size = enc_bytes
    return clip


def _build_qwenimage_edit_clip(encoder: Any, folder_paths_embeddings: list[str]) -> comfy.sd.CLIP:
    """Build a ``comfy.sd.CLIP`` backed by a nunchaku Qwen2.5-VL edit encoder (vision-aware)."""
    TEModelClass = _make_qwenimage_edit_te_class(encoder)
    enc_bytes = _encoder_byte_size(encoder)

    class _ClipTarget:
        clip = TEModelClass
        tokenizer = QwenImageTokenizer
        params: dict = {}

    clip = comfy.sd.CLIP(
        target=_ClipTarget(),
        embedding_directory=folder_paths_embeddings,
        parameters=enc_bytes,
        state_dict=[],
    )
    clip.patcher.size = enc_bytes
    return clip


class NunchakuQwenImageTextEncoderLoader:
    """
    Load a nunchaku-quantised Qwen2.5-VL text encoder for QwenImage models.

    Accepts checkpoints exported for the **text-only** QwenImage path.
    Produces a ComfyUI ``CLIP`` object compatible with QwenImage workflows.
    """

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_text_encoder"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Qwen2.5-VL Text Encoder Loader (QwenImage)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    get_filename_list("text_encoders"),
                    {
                        "tooltip": (
                            "Nunchaku-quantised Qwen2.5-VL text-only encoder checkpoint "
                            "(.safetensors)."
                        )
                    },
                ),
            }
        }

    def load_text_encoder(self, model_path: str) -> tuple:
        path = get_full_path_or_raise("text_encoders", model_path)

        load_device = comfy.model_management.get_torch_device()
        torch_dtype = torch.float16 if is_turing(load_device) else torch.bfloat16
        logger.info("Loading nunchaku Qwen2.5-VL text encoder from %s on %s (dtype=%s)", path, load_device, torch_dtype)

        encoder = NunchakuQwenEncoderModel.from_pretrained(
            path,
            device=load_device,
            torch_dtype=torch_dtype,
        )

        emb_dirs = _folder_paths.get_folder_paths("embeddings")
        clip = _build_qwenimage_clip(encoder, emb_dirs)
        logger.info("Nunchaku Qwen2.5-VL text encoder loaded successfully.")
        return (clip,)


class NunchakuQwenImageEditEncoderLoader:
    """
    Load a nunchaku-quantised Qwen2.5-VL edit encoder for QwenImage edit models.

    Accepts checkpoints exported for the **multimodal edit** QwenImage path.
    Supports both text-only and vision-conditioned (image + text) prompts.
    When the upstream tokenizer inserts image dicts into the token stream, the
    vision ViT is invoked and visual features are fused with the LLM hidden states.

    Produces a ComfyUI ``CLIP`` object compatible with QwenImage edit workflows.
    """

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_text_encoder"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Qwen2.5-VL Edit Encoder Loader (QwenImage Edit)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    get_filename_list("text_encoders"),
                    {
                        "tooltip": (
                            "Nunchaku-quantised Qwen2.5-VL edit/multimodal encoder "
                            "checkpoint (.safetensors)."
                        )
                    },
                ),
            }
        }

    def load_text_encoder(self, model_path: str) -> tuple:
        path = get_full_path_or_raise("text_encoders", model_path)

        load_device = comfy.model_management.get_torch_device()
        torch_dtype = torch.float16 if is_turing(load_device) else torch.bfloat16
        logger.info(
            "Loading nunchaku Qwen2.5-VL edit encoder from %s on %s (dtype=%s)", path, load_device, torch_dtype
        )

        encoder = NunchakuQwenEncoderModel.from_pretrained(
            path,
            device=load_device,
            torch_dtype=torch_dtype,
        )

        emb_dirs = _folder_paths.get_folder_paths("embeddings")
        clip = _build_qwenimage_edit_clip(encoder, emb_dirs)
        logger.info("Nunchaku Qwen2.5-VL edit encoder loaded successfully.")
        return (clip,)
