import logging
import os
import types

import comfy
import folder_paths
import torch
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
    outputs = self.encoder(input_ids=input_ids, inputs_embeds=embeds, attention_mask=attention_mask)
    hidden_states = outputs["last_hidden_state"]
    hidden_states = hidden_states.to(dtype=dtype)
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

    TITLE = "Nunchaku Text Encoder Loader"

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
                clip_target.clip = comfy.text_encoders.flux.flux_clip(dtype_t5=torch.float16)
            clip_target.tokenizer = comfy.text_encoders.flux.FluxTokenizer
    else:
        raise NotImplementedError(f"Clip type {clip_type} not implemented.")

    parameters = 0
    for c in state_dicts:
        parameters += comfy.utils.calculate_parameters(c)
        tokenizer_data, model_options = comfy.text_encoders.long_clipl.model_options_long_clip(
            c, tokenizer_data, model_options
        )
    with torch.device("meta"):
        clip = comfy.sd.CLIP(
            clip_target,
            embedding_directory=embedding_directory,
            parameters=parameters,
            tokenizer_data=tokenizer_data,
            model_options=model_options,
        )

    device = model_options.get("load_device", comfy.model_management.text_encoder_device())
    if nunchaku_model_id is None:
        clip.cond_stage_model.to_empty(
            device=model_options.get("load_device", comfy.model_management.text_encoder_device())
        )
    else:
        for n, m in clip.cond_stage_model.named_children():
            if n != "t5xxl":
                m.to_empty(device=device)
            else:
                transformer = m.transformer
                param = next(transformer.parameters())
                dtype = param.dtype

                transformer = NunchakuT5EncoderModel.from_pretrained(
                    paths[nunchaku_model_id], device=device, torch_dtype=dtype
                )
                transformer.forward = types.MethodType(nunchaku_t5_forward, transformer)
                transformer.shared = WrappedEmbedding(transformer.shared)
                m.transformer = transformer
                m.logit_scale = nn.Parameter(torch.zeros_like(m.logit_scale, device=device))

    for state_dict, metadata in zip(state_dicts, metadata_list):
        if metadata is not None and metadata.get("model_class", None) == "NunchakuT5EncoderModel":
            continue  # Skip Nunchaku T5 model loading here, handled separately above
        m, u = clip.load_sd(state_dict)
        if len(m) > 0:
            logging.warning("clip missing: {}".format(m))

        if len(u) > 0:
            logging.debug("clip unexpected: {}".format(u))

    for n, p in clip.cond_stage_model.named_parameters():
        assert p.device.type != "meta", f"Parameter {n} is still on meta device, expected it to be on {device}."

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
        return (clip,)
