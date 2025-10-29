"""
This module provides the :class:`NunchakuFluxLoraLoader` node
for applying LoRA weights to Nunchaku FLUX models within ComfyUI.
"""

import copy
import logging
import os

import folder_paths

import comfy
from ...models.sdxl import load_lora_for_models

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuSDXLLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to. Make sure it is NunchakuSDXLUnetLoader."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"
    TITLE = "Nunchaku SDXL LoRA Loader"
    CATEGORY = "loaders"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        #load_lora_for_models now takes in a list for lora, strength_model and strength_clip

        model_lora, clip_lora = load_lora_for_models(model, clip, [lora], [strength_model], [strength_clip])
        return (model_lora, clip_lora)

class NunchakuSDXLLoraStack:
    """
    Node for loading and applying multiple LoRAs to a Nunchaku SDXL model with dynamic input.

    This node allows you to configure multiple LoRAs with their respective strengths
    in a single node, providing the same effect as chaining multiple LoRA nodes.

    Attributes
    ----------
    RETURN_TYPES : tuple
        The return type of the node ("MODEL",).
    OUTPUT_TOOLTIPS : tuple
        Tooltip for the output.
    FUNCTION : str
        The function to call ("load_lora_stack").
    TITLE : str
        Node title.
    CATEGORY : str
        Node category.
    DESCRIPTION : str
        Node description.
    """

    def __init__(self):
        self.loaded_loras = {}

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the LoRA stack node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and optional LoRA inputs.
        """
        # Base inputs
        inputs = {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to. Make sure it is NunchakuSDXLUnetLoader."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
            },
            "optional": {},
        }

        # Add fixed number of LoRA inputs (15 slots)
        for i in range(1, 16):  # Support up to 15 LoRAs
            inputs["optional"][f"lora_name_{i}"] = (
                ["None"] + folder_paths.get_filename_list("loras"),
                {"tooltip": f"The file name of LoRA {i}. Select 'None' to skip this slot."},
            )
            inputs["optional"][f"model_strength_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": f"Model strength for LoRA {i}. This value can be negative.",
                },
            )
            inputs["optional"][f"clip_strength_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": f"CLIP strength for LoRA {i}. This value can be negative.",
                },
            )

        return inputs

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora_stack"
    TITLE = "Nunchaku SDXL LoRA Stack"

    CATEGORY = "Nunchaku"
    DESCRIPTION = (
        "Apply multiple LoRAs to a diffusion model in a single node. "
        "Equivalent to chaining multiple LoRA nodes but more convenient for managing many LoRAs. "
        "Supports up to 15 LoRAs simultaneously. Set unused slots to 'None' to skip them."
    )

    def load_lora_stack(self, model, clip, **kwargs):
        """
        Apply multiple LoRAs to a Nunchaku SDXL diffusion model.

        Parameters
        ----------
        model : object
            The diffusion model to modify.
        **kwargs
            Dynamic LoRA name and strength parameters.

        Returns
        -------
        tuple
            A tuple containing the modified diffusion model.
        """
        # Collect LoRA information to apply
        lora_list = []
        strength_model_list = []
        strength_clip_list = []

        for i in range(1, 16):  # Check all 15 LoRA slots
            lora_name = kwargs.get(f"lora_name_{i}")
            model_strength = kwargs.get(f"model_strength_{i}", 1.0)
            clip_strength = kwargs.get(f"clip_strength_{i}", 1.0)

            # Skip unset or None LoRAs
            if lora_name is None or lora_name == "None" or lora_name == "":
                continue

            # Skip LoRAs with zero strength
            if abs(model_strength) < 1e-5 and abs(clip_strength) < 1e-5:
                continue

            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)

            if lora_path not in self.loaded_loras.keys():
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_loras[lora_path] = {"lora": lora, "model_strength": model_strength, "clip_strength": clip_strength}

            lora_list.append(self.loaded_loras.get(lora_path).get("lora"))
            strength_model_list.append(self.loaded_loras.get(lora_path).get("model_strength"))
            strength_clip_list.append(self.loaded_loras.get(lora_path).get("clip_strength"))
                

        # If no LoRAs need to be applied, return the original model
        if lora_list == []:
            return (model, clip)


        model_lora, clip_lora = load_lora_for_models(model, clip, lora_list, strength_model_list, strength_clip_list)
        return (model_lora, clip_lora)
