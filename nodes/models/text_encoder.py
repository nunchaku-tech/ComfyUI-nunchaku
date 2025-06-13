import gc
import logging
import os
import types
from typing import Callable

import comfy
import folder_paths
import torch
from comfy.text_encoders.flux import FluxClipModel
from torch import nn
from transformers import T5EncoderModel

from nunchaku import NunchakuT5EncoderModel

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def nunchaku_t5_forward(
    self: T5EncoderModel,
    input_ids: torch.LongTensor,
    attention_mask,
    embeds=None,
    intermediate_output=None,
    final_layer_norm_intermediate=True,
    dtype: str | torch.dtype = torch.bfloat16,
    **kwargs,
):
    assert attention_mask is None
    assert intermediate_output is None
    assert final_layer_norm_intermediate

    def get_device(tensors: list[torch.Tensor]) -> torch.device:
        for t in tensors:
            if t is not None:
                return t.device
        return torch.device("cpu")

    original_device = None
    if get_device([input_ids, attention_mask, embeds]) != "cuda":
        original_device = get_device([input_ids, attention_mask, embeds])
        logger.warning(
            "Currently, Nunchaku T5 encoder requires CUDA for processing. "
            f"Input tensor is not on {str(original_device)}, moving to CUDA for T5 encoder processing."
        )
        input_ids = input_ids.to(torch.cuda.current_device()) if input_ids is not None else None
        embeds = embeds.to(torch.cuda.current_device()) if embeds is not None else None
        attention_mask = attention_mask.to(torch.cuda.current_device()) if attention_mask is not None else None
        self.encoder = self.encoder.to(torch.cuda.current_device())
    outputs = self.encoder(input_ids=input_ids, inputs_embeds=embeds, attention_mask=attention_mask)

    hidden_states = outputs["last_hidden_state"]
    hidden_states = hidden_states.to(dtype=dtype)
    if original_device is not None:
        hidden_states = hidden_states.to(original_device)
        self.encoder = self.encoder.to(original_device)

        gc.collect()
        torch.cuda.empty_cache()

    return hidden_states, None


class WrappedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input: torch.Tensor, out_dtype: torch.dtype | None = None):
        return self.embedding(input)

    @property
    def weight(self):
        return self.embedding.weight


class NunchakuTextEncoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        prefixes = folder_paths.folder_names_and_paths["text_encoders"][0]
        local_folders = set()
        for prefix in prefixes:
            if os.path.exists(prefix) and os.path.isdir(prefix):
                local_folders_ = os.listdir(prefix)
                local_folders_ = [
                    folder
                    for folder in local_folders_
                    if not folder.startswith(".") and os.path.isdir(os.path.join(prefix, folder))
                ]
                local_folders.update(local_folders_)
        model_paths = ["none"] + sorted(list(local_folders))
        return {
            "required": {
                "model_type": (["flux"],),
                "text_encoder1": (folder_paths.get_filename_list("text_encoders"),),
                "text_encoder2": (folder_paths.get_filename_list("text_encoders"),),
                "t5_min_length": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 1024,
                        "step": 128,
                        "display": "number",
                        "lazy": True,
                    },
                ),
                "use_4bit_t5": (["disable", "enable"],),
                "int4_model": (
                    model_paths,
                    {"tooltip": "The name of the 4-bit T5 model."},
                ),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_text_encoder"

    CATEGORY = "Nunchaku"

    TITLE = "Nunchaku Text Encoder Loader (Deprecated)"

    def load_text_encoder(
        self,
        model_type: str,
        text_encoder1: str,
        text_encoder2: str,
        t5_min_length: int,
        use_4bit_t5: str,
        int4_model: str,
    ):
        logger.warning(
            "Nunchaku Text Encoder Loader will be deprecated in v0.4. "
            "Please use the Nunchaku Text Encoder Loader V2 node instead."
        )
        text_encoder_path1 = folder_paths.get_full_path_or_raise("text_encoders", text_encoder1)
        text_encoder_path2 = folder_paths.get_full_path_or_raise("text_encoders", text_encoder2)
        if model_type == "flux":
            clip_type = comfy.sd.CLIPType.FLUX
        else:
            raise ValueError(f"Unknown type {model_type}")

        clip = comfy.sd.load_clip(
            ckpt_paths=[text_encoder_path1, text_encoder_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
        )

        if model_type == "flux":
            clip.tokenizer.t5xxl.min_length = t5_min_length

        if use_4bit_t5 == "enable":
            assert int4_model != "none", "Please select a 4-bit T5 model."
            transformer = clip.cond_stage_model.t5xxl.transformer
            param = next(transformer.parameters())
            dtype = param.dtype
            device = param.device

            prefixes = folder_paths.folder_names_and_paths["text_encoders"][0]
            model_path = None
            for prefix in prefixes:
                if os.path.exists(os.path.join(prefix, int4_model)):
                    model_path = os.path.join(prefix, int4_model)
                    break
            if model_path is None:
                model_path = int4_model
            transformer = NunchakuT5EncoderModel.from_pretrained(model_path)
            transformer.forward = types.MethodType(nunchaku_t5_forward, transformer)
            transformer.shared = WrappedEmbedding(transformer.shared)

            clip.cond_stage_model.t5xxl.transformer = (
                transformer.to(device=device, dtype=dtype) if device.type == "cuda" else transformer
            )

        return (clip,)


def nunchaku_flux_clip(nunchaku_t5_path: str | os.PathLike[str], dtype_t5=None) -> Callable:
    class NunchakuFluxClipModel(FluxClipModel):

        def __init__(
            self,
            dtype_t5=None,
            device="cpu",
            dtype=None,
            model_options={},
        ):
            super(FluxClipModel, self).__init__()
            dtype_t5 = comfy.model_management.pick_weight_dtype(dtype_t5, dtype, device)
            self.clip_l = comfy.sd1_clip.SDClipModel(
                device=device, dtype=dtype, return_projected_pooled=False, model_options=model_options
            )

            # We are using meta device for T5XXL to avoid loading it into memory and then replacing it
            with torch.device("meta"):
                self.t5xxl = comfy.text_encoders.sd3_clip.T5XXLModel(
                    device=device, dtype=dtype_t5, model_options=model_options
                )

            transformer = NunchakuT5EncoderModel.from_pretrained(nunchaku_t5_path, device=device, torch_dtype=dtype_t5)
            transformer.forward = types.MethodType(nunchaku_t5_forward, transformer)
            transformer.shared = WrappedEmbedding(transformer.shared)
            self.t5xxl.transformer = transformer
            self.t5xxl.logit_scale = nn.Parameter(torch.zeros_like(self.t5xxl.logit_scale, device=device))

            self.dtypes = set([dtype, dtype_t5])

    return NunchakuFluxClipModel


def load_text_encoder_state_dicts(
    paths: list[str | os.PathLike[str]],
    embedding_directory: str | os.PathLike[str] | None = None,
    clip_type=comfy.sd.CLIPType.FLUX,
    model_options: dict = {},
):
    state_dicts, metadata_list = [], []

    for p in paths:
        sd, metadata = comfy.utils.load_torch_file(p, safe_load=True, return_metadata=True)
        state_dicts.append(sd)
        metadata_list.append(metadata)

    class EmptyClass:
        pass

    for i in range(len(state_dicts)):
        if "transformer.resblocks.0.ln_1.weight" in state_dicts[i]:
            state_dicts[i] = comfy.utils.clip_text_transformers_convert(state_dicts[i], "", "")
        else:
            if "text_projection" in state_dicts[i]:
                # old models saved with the CLIPSave node
                state_dicts[i]["text_projection.weight"] = state_dicts[i]["text_projection"].transpose(0, 1)

    tokenizer_data = {}
    clip_target = EmptyClass()
    clip_target.params = {}

    nunchaku_model_id = None
    for i, metadata in enumerate(metadata_list):
        if metadata is not None and metadata.get("model_class", None) == "NunchakuT5EncoderModel":
            nunchaku_model_id = i
            break

    if len(state_dicts) == 2:
        if clip_type == comfy.sd.CLIPType.FLUX:
            if nunchaku_model_id is None:
                clip_target.clip = comfy.text_encoders.flux.flux_clip(**comfy.sd.t5xxl_detect(state_dicts))
            else:
                clip_target.clip = nunchaku_flux_clip(nunchaku_t5_path=paths[nunchaku_model_id], dtype_t5=torch.float16)
            clip_target.tokenizer = comfy.text_encoders.flux.FluxTokenizer
    else:
        raise NotImplementedError(f"Clip type {clip_type} not implemented.")

    parameters = 0
    for c in state_dicts:
        parameters += comfy.utils.calculate_parameters(c)
        tokenizer_data, model_options = comfy.text_encoders.long_clipl.model_options_long_clip(
            c, tokenizer_data, model_options
        )
    clip = comfy.sd.CLIP(
        clip_target,
        embedding_directory=embedding_directory,
        parameters=parameters,
        tokenizer_data=tokenizer_data,
        model_options=model_options,
    )
    for state_dict, metadata in zip(state_dicts, metadata_list):
        if metadata is not None and metadata.get("model_class", None) == "NunchakuT5EncoderModel":
            continue  # Skip Nunchaku T5 model loading here, handled separately above
        m, u = clip.load_sd(state_dict)
        if len(m) > 0:
            logging.warning("clip missing: {}".format(m))

        if len(u) > 0:
            logging.debug("clip unexpected: {}".format(u))

    return clip


class NunchakuTextEncoderLoaderV2:
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_text_encoder"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Text Encoder Loader V2"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["flux.1"],),
                "text_encoder1": (folder_paths.get_filename_list("text_encoders"),),
                "text_encoder2": (folder_paths.get_filename_list("text_encoders"),),
                "t5_min_length": (
                    "INT",
                    {"default": 512, "min": 256, "max": 1024, "step": 128, "display": "number", "lazy": True},
                ),
            }
        }

    def load_text_encoder(self, model_type: str, text_encoder1: str, text_encoder2: str, t5_min_length: int):
        text_encoder_path1 = folder_paths.get_full_path_or_raise("text_encoders", text_encoder1)
        text_encoder_path2 = folder_paths.get_full_path_or_raise("text_encoders", text_encoder2)
        if model_type == "flux.1":
            clip_type = comfy.sd.CLIPType.FLUX
        else:
            raise ValueError(f"Unknown type {model_type}")

        clip = load_text_encoder_state_dicts(
            [text_encoder_path1, text_encoder_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options={},
        )
        clip.tokenizer.t5xxl.min_length = t5_min_length
        return (clip,)
