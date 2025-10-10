"""
This module provides the :class:`NunchakuQwenImageLoraLoader` node
for applying LoRA weights to Nunchaku Qwen Image models within ComfyUI.
"""

import copy
import logging
import os

import folder_paths

from nunchaku.lora.qwenimage import to_diffusers, compose_lora

# from ...wrappers.qwenimage import ComfyQwenImageWrapper  # Not needed - working directly with transformer

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuQwenImageLoraLoader:
    """
    Node for loading and applying a LoRA to a Nunchaku Qwen Image model.

    This implementation follows the Flux LoRA Loader design pattern:
    - LoRAs are stored as metadata (path + strength) without immediate application
    - Actual composition and application happens lazily in the wrapper's forward pass
    - This allows flexible strength adjustment and avoids redundant conversions

    Attributes
    ----------
    RETURN_TYPES : tuple
        The return type of the node ("MODEL",).
    OUTPUT_TOOLTIPS : tuple
        Tooltip for the output.
    FUNCTION : str
        The function to call ("load_lora").
    TITLE : str
        Node title.
    CATEGORY : str
        Node category.
    DESCRIPTION : str
        Node description.
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types and tooltips for the node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and their descriptions for the node interface.
        """
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The diffusion model the LoRA will be applied to. "
                        "Make sure the model is loaded by `Nunchaku Qwen Image DiT Loader`."
                    },
                ),
                "lora_name": (
                    folder_paths.get_filename_list("loras"),
                    {"tooltip": "The file name of the LoRA."},
                ),
                "lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model. This value can be negative.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora"
    TITLE = "Nunchaku Qwen Image LoRA Loader"

    CATEGORY = "Nunchaku"
    DESCRIPTION = (
        "LoRAs are used to modify the diffusion model, "
        "altering the way in which latents are denoised such as applying styles. "
        "You can link multiple LoRA nodes."
    )

    def load_lora(self, model, lora_name: str, lora_strength: float):
        """
        Apply a LoRA to a Nunchaku Qwen Image diffusion model.

        Following Flux's design pattern, this method stores LoRA metadata in the wrapper.
        The actual composition and application happens lazily in the wrapper's forward pass.

        Parameters
        ----------
        model : object
            The diffusion model to modify.
        lora_name : str
            The name of the LoRA to apply.
        lora_strength : float
            The strength with which to apply the LoRA.

        Returns
        -------
        tuple
            A tuple containing the modified diffusion model.
        """
        if abs(lora_strength) < 1e-5:
            return (model,)  # If the strength is too small, return the original model

        model_wrapper = model.model.diffusion_model
        
        # Check if this is a ComfyQwenImageWrapper
        from ...wrappers.qwenimage import ComfyQwenImageWrapper
        if not isinstance(model_wrapper, ComfyQwenImageWrapper):
            logger.error(f"❌ Model type mismatch!")
            logger.error(f"   Expected: ComfyQwenImageWrapper (Nunchaku Qwen Image model)")
            logger.error(f"   Got: {type(model_wrapper).__name__}")
            logger.error(f"   Please make sure you're using 'Nunchaku Qwen Image DiT Loader' to load the model.")
            raise TypeError(
                f"This LoRA loader only works with Nunchaku Qwen Image models. "
                f"Got {type(model_wrapper).__name__} instead. "
                f"Please use 'Nunchaku Qwen Image DiT Loader' to load your model."
            )

        transformer = model_wrapper.model
        
        # Flux-style deepcopy: temporarily remove transformer to avoid copying it
        model_wrapper.model = None
        ret_model = copy.deepcopy(model)  # copy everything except the model
        ret_model_wrapper = ret_model.model.diffusion_model
        
        if not isinstance(ret_model_wrapper, ComfyQwenImageWrapper):
            raise TypeError(f"Model wrapper type changed after deepcopy: {type(ret_model_wrapper).__name__}")

        model_wrapper.model = transformer
        ret_model_wrapper.model = transformer  # Share the same transformer
        
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        ret_model_wrapper.loras.append((lora_path, lora_strength))

        logger.info(f"LoRA added: {lora_name} (strength={lora_strength})")
        logger.debug(f"Total LoRAs: {len(ret_model_wrapper.loras)}")

        return (ret_model,)


class NunchakuQwenImageLoraStack:
    """
    Node for loading and applying multiple LoRAs to a Nunchaku Qwen Image model with dynamic input.

    This node allows you to configure multiple LoRAs with their respective strengths
    in a single node, providing the same effect as chaining multiple LoRA nodes.

    Following Flux's design pattern, this method only stores LoRA metadata.
    The actual composition and application happens lazily in the wrapper's forward pass.

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
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The diffusion model the LoRAs will be applied to. "
                        "Make sure the model is loaded by `Nunchaku Qwen Image DiT Loader`."
                    },
                ),
            },
            "optional": {},
        }

        # Add fixed number of LoRA inputs (15 slots)
        for i in range(1, 16):  # Support up to 15 LoRAs
            inputs["optional"][f"lora_name_{i}"] = (
                ["None"] + folder_paths.get_filename_list("loras"),
                {"tooltip": f"The file name of LoRA {i}. Select 'None' to skip this slot."},
            )
            inputs["optional"][f"lora_strength_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": f"Strength for LoRA {i}. This value can be negative.",
                },
            )

        return inputs

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model with all LoRAs applied.",)
    FUNCTION = "load_lora_stack"
    TITLE = "Nunchaku Qwen Image LoRA Stack"

    CATEGORY = "Nunchaku"
    DESCRIPTION = (
        "Apply multiple LoRAs to a diffusion model in a single node. "
        "Equivalent to chaining multiple LoRA nodes but more convenient for managing many LoRAs. "
        "Supports up to 15 LoRAs simultaneously. Set unused slots to 'None' to skip them."
    )

    def load_lora_stack(self, model, **kwargs):
        """
        Apply multiple LoRAs to a Nunchaku Qwen Image diffusion model.

        Following Flux's design pattern, this method uses shared transformer instances
        to avoid memory overhead and enable efficient LoRA caching.

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
        loras_to_apply = []

        for i in range(1, 16):  # Check all 15 LoRA slots
            lora_name = kwargs.get(f"lora_name_{i}")
            lora_strength = kwargs.get(f"lora_strength_{i}", 1.0)

            # Skip unset or None LoRAs
            if lora_name is None or lora_name == "None" or lora_name == "":
                continue

            # Skip LoRAs with zero strength
            if abs(lora_strength) < 1e-5:
                continue

            loras_to_apply.append((lora_name, lora_strength))

        # If no LoRAs need to be applied, return the original model
        if not loras_to_apply:
            return (model,)

        model_wrapper = model.model.diffusion_model
        
        # Check if this is a ComfyQwenImageWrapper
        from ...wrappers.qwenimage import ComfyQwenImageWrapper
        if not isinstance(model_wrapper, ComfyQwenImageWrapper):
            logger.error(f"❌ Model type mismatch!")
            logger.error(f"   Expected: ComfyQwenImageWrapper (Nunchaku Qwen Image model)")
            logger.error(f"   Got: {type(model_wrapper).__name__}")
            logger.error(f"   Please make sure you're using 'Nunchaku Qwen Image DiT Loader' to load the model.")
            raise TypeError(
                f"This LoRA loader only works with Nunchaku Qwen Image models. "
                f"Got {type(model_wrapper).__name__} instead. "
                f"Please use 'Nunchaku Qwen Image DiT Loader' to load your model."
            )

        transformer = model_wrapper.model
        
        # Flux-style deepcopy: temporarily remove transformer to avoid copying it
        model_wrapper.model = None
        ret_model = copy.deepcopy(model)  # copy everything except the model
        ret_model_wrapper = ret_model.model.diffusion_model
        
        if not isinstance(ret_model_wrapper, ComfyQwenImageWrapper):
            raise TypeError(f"Model wrapper type changed after deepcopy: {type(ret_model_wrapper).__name__}")

        model_wrapper.model = transformer
        ret_model_wrapper.model = transformer  # Share the same transformer

        # Clear existing LoRA list
        ret_model_wrapper.loras = []

        # Add all LoRAs
        for lora_name, lora_strength in loras_to_apply:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            ret_model_wrapper.loras.append((lora_path, lora_strength))
            
            logger.debug(f"LoRA added to stack: {lora_name} (strength={lora_strength})")

        logger.info(f"Total LoRAs in stack: {len(ret_model_wrapper.loras)}")

        return (ret_model,)