"""
This module provides the :class:`NunchakuSDXLUnetLoader` class for loading Nunchaku SDXL models.
Currently CPU Offload and attention implementation selection are not supported.

"""

import gc
import json
import logging
import os
from pathlib import Path

import comfy.model_management
import comfy.model_patcher
import folder_paths
import torch

from comfy.supported_models import SDXL as SDXLSupported #note that this DOES NOT cover the SDXL Refiner
from comfy.model_base import SDXL as SDXLModelBase
from comfy.utils import load_torch_file
#from comfy.sd import load_diffusion_model_state_dict
from comfy import model_management, model_detection
import comfy.ops
ops = comfy.ops.disable_weight_init

#from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel
#from nunchaku.caching.diffusers_adapters.flux import apply_cache_on_transformer

from nunchaku.utils import get_precision

#we need to modify ComfyUI's unet_to_diffusers to support the different layers in the unet
from ...models.sdxl import NunchakuSDXLUNetModel, unet_to_diffusers, convert_sdxl_state_dict

from nunchaku.utils import is_turing



# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_diffusion_model_state_dict(sd, model_options={}):
    """

    modified version of comfy.sd.load_diffusion_model_state_dict

    Loads a UNet diffusion model from a state dictionary, supporting both diffusers and regular formats.

    Args:
        sd (dict): State dictionary containing model weights and configuration
        model_options (dict, optional): Additional options for model loading. Supports:
            - dtype: Override model data type
            - custom_operations: Custom model operations
            - fp8_optimizations: Enable FP8 optimizations

    Returns:
        ModelPatcher: A wrapped model instance that handles device management and weight loading.
        Returns None if the model configuration cannot be detected.

    The function:
    1. Detects and handles different model formats (regular, diffusers, mmdit)
    2. Configures model dtype based on parameters and device capabilities
    3. Handles weight conversion and device placement
    4. Manages model optimization settings
    5. Loads weights and returns a device-managed model instance
    """
    dtype = model_options.get("dtype", None)

    #Allow loading unets from checkpoint files
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    
    if len(temp_sd) > 0:
        sd = temp_sd

    sd = convert_sdxl_state_dict(sd)

    parameters = comfy.utils.calculate_parameters(sd)
    weight_dtype = comfy.utils.weight_dtype(sd)

    load_device = model_management.get_torch_device()
    model_config = model_detection.model_config_from_unet(sd, "")

    SDXL_model_unet_config = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
            'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
            'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
            'use_temporal_attention': False, 'use_temporal_resblock': False, 'dtype': torch.bfloat16}


    if model_config is not None:
        new_sd = sd
    else:

        #right now, ONLY SDXL base (and not refiner) has been converted to Nunchaku. So unet config assumes SDXL base only.

        model_config = SDXLSupported(SDXL_model_unet_config)

        diffusers_keys = unet_to_diffusers(SDXL_model_unet_config)


        new_sd = {}
        for k in diffusers_keys:
            if k in sd:
                new_sd[diffusers_keys[k]] = sd.pop(k)
            else:
                #leave this here to uncomment in case any further errors converting state_dicts from nunchaku format to ComfyUI UNet format
                #logging.warning("{} {}".format(diffusers_keys[k], k))
                pass

    offload_device = model_management.unet_offload_device()
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.scaled_fp8 is not None:
        weight_dtype = None

    if dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)
    else:
        unet_dtype = dtype

    manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)

    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get("custom_operations", model_config.custom_operations)
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(new_sd, "")

    #change the unet_config to add Nunchaku-specific arguments
    SDXL_model_unet_config["attn_precision"] = model_options.get("precision")
    SDXL_model_unet_config["attn_rank"] = model_options.get("rank")
    SDXL_model_unet_config["cache_threshold"] = model_options.get("cache_threshold")
    SDXL_model_unet_config["dtype"] = model_options.get("dtype")

    model.diffusion_model = NunchakuSDXLUNetModel(**SDXL_model_unet_config, device=load_device, operations=ops)
    model = model.to(offload_device)
    
    

    model.load_model_weights(new_sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        logging.info("left over keys in diffusion model: {}".format(left_over))
    return comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)


class NunchakuSDXLUNetLoader:
    """
    Loader for Nunchaku SDXL models.

    This class manages model loading, device selection, attention implementation,
    CPU offload, and caching for efficient inference.

    Attributes
    ----------
    transformer : :class:`~NunchakuSDXLUNetModel` or None
        The loaded transformer model.
    metadata : dict or None
        Metadata associated with the loaded model.
    model_path : str or None
        Path to the loaded model.
    device : torch.device or None
        Device on which the model is loaded.
    cpu_offload : bool or None
        Whether CPU offload is enabled.
    data_type : str or None
        Data type used for inference.
    patcher : object or None
        ComfyUI model patcher instance.
    """

    def __init__(self):
        """
        Initialize the NunchakuSDXLUnetLoader.

        Sets up internal state and selects the default torch device.
        """
        self.model_sd = None
        self.metadata = None
        self.model_path = None
        self.device = None
        self.cpu_offload = None
        self.data_type = None
        self.patcher = None
        self.device = comfy.model_management.get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        """
        Define the input types and tooltips for the node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and their descriptions for the node interface.
        """
        prefixes = folder_paths.folder_names_and_paths["diffusion_models"][0]
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
        model_paths = sorted(list(local_folders))
        safetensor_files = folder_paths.get_filename_list("diffusion_models")

        # exclude the safetensors in the legacy svdquant folders
        new_safetensor_files = []
        for safetensor_file in safetensor_files:
            safetensor_path = folder_paths.get_full_path_or_raise("diffusion_models", safetensor_file)
            safetensor_path = Path(safetensor_path)
            if not (safetensor_path.parent / "config.json").exists():
                new_safetensor_files.append(safetensor_file)
        safetensor_files = new_safetensor_files
        model_paths = model_paths + safetensor_files

        ngpus = torch.cuda.device_count()

        all_turing = True
        for i in range(torch.cuda.device_count()):
            if not is_turing(f"cuda:{i}"):
                all_turing = False

        if all_turing:
            attention_options = ["nunchaku-fp16"]  # turing GPUs do not support flashattn2
            dtype_options = ["float16"]
        else:
            attention_options = ["nunchaku-fp16", "flash-attention2"]
            dtype_options = ["bfloat16", "float16"]

        return {
            "required": {
                "model_path": (
                    model_paths,
                    {"tooltip": "The Nunchaku SDXL model."},
                ),
                "cache_threshold": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1,
                        "step": 0.001,
                        "tooltip": "Adjusts the first-block caching tolerance"
                        "like `residual_diff_threshold` in WaveSpeed. "
                        "Increasing the value enhances speed at the cost of quality. "
                        "A typical setting is 0.12. Setting it to 0 disables the effect.",
                    },
                ),
                "attention": (
                    attention_options,
                    {
                        "default": attention_options[0],
                        "tooltip": "This is not implemented yet for SDXL. Attention implementation. The default implementation is `flash-attention2`. "
                        "`nunchaku-fp16` use FP16 attention, offering ~1.2× speedup. "
                        "Note that 20-series GPUs can only use `nunchaku-fp16`.",
                    },
                ),
                "cpu_offload": (
                    ["auto", "enable", "disable"],
                    {
                        "default": "auto",
                        "tooltip": "This is not implemented yet for SDXL. Whether to enable CPU offload for the transformer model."
                        "auto' will enable it if the GPU memory is less than 14G.",
                    },
                ),
                "device_id": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": ngpus - 1,
                        "step": 1,
                        "display": "number",
                        "lazy": True,
                        "tooltip": "The GPU device ID to use for the model.",
                    },
                ),
                "data_type": (
                    dtype_options,
                    {
                        "default": dtype_options[0],
                        "tooltip": "Specifies the model's data type. Default is `bfloat16`. "
                        "For 20-series GPUs, which do not support `bfloat16`, use `float16` instead.",
                    },
                ),
            },
            "optional": {
                "i2f_mode": (
                    ["enabled", "always"],
                    {
                        "default": "enabled",
                        "tooltip": "The GEMM implementation for 20-series GPUs"
                        "— this option is only applicable to these GPUs.",
                    },
                )
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku SDXL UNet Loader"

    def load_model(
        self,
        model_path: str,
        attention: str,
        cache_threshold: float,
        cpu_offload: str,
        device_id: int,
        data_type: str,
        **kwargs,
    ):
        """
        Load a Nunchaku SDXL model with the specified configuration.

        Parameters
        ----------
        model_path : str
            Path to the model directory or safetensors file.
        attention : str
            Attention implementation to use ("nunchaku-fp16" or "flash-attention2").
        cache_threshold : float
            Caching tolerance for first-block cache. See :ref:`nunchaku:usage-fbcache` for details.
        cpu_offload : str
            Whether to enable CPU offload ("auto", "enable", "disable").
        device_id : int
            GPU device ID to use.
        data_type : str
            Data type for inference ("bfloat16" or "float16").
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tuple
            A tuple containing the loaded and patched model.
        """
        device = torch.device(f"cuda:{device_id}")


        if model_path.endswith((".sft", ".safetensors")):
            model_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", model_path))
        else:
            prefixes = folder_paths.folder_names_and_paths["diffusion_models"][0]
            for prefix in prefixes:
                prefix = Path(prefix)
                if (prefix / model_path).exists() and (prefix / model_path).is_dir():
                    model_path = prefix / model_path
                    break

        # Check if the device_id is valid
        if device_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid device_id: {device_id}. Only {torch.cuda.device_count()} GPUs available.")

        # Get the GPU properties
        gpu_properties = torch.cuda.get_device_properties(device_id)
        gpu_memory = gpu_properties.total_memory / (1024**2)  # Convert to MiB
        gpu_name = gpu_properties.name
        print(f"GPU {device_id} ({gpu_name}) Memory: {gpu_memory} MiB")

        # Check if CPU offload needs to be enabled (this does nothing for now)
        if cpu_offload == "auto":
            if gpu_memory < 14336:  # 14GB threshold
                cpu_offload_enabled = True
                print("VRAM < 14GiB, enabling CPU offload")
            else:
                cpu_offload_enabled = False
                print("VRAM > 14GiB, disabling CPU offload")
        elif cpu_offload == "enable":
            cpu_offload_enabled = True
            print("Enabling CPU offload")
        else:
            cpu_offload_enabled = False
            print("Disabling CPU offload")

        """

        if (
            self.model_path != model_path
            or self.device != device
            or self.cpu_offload != cpu_offload_enabled
            or self.data_type != data_type
        ):
            if self.model_sd is not None:
                model_size = comfy.model_management.module_size(self.model_sd)
                model = self.model_sd
                self.model_sd = None
                model.to("cpu")
                del model
                gc.collect()
                comfy.model_management.cleanup_models_gc()
                comfy.model_management.soft_empty_cache()
                comfy.model_management.free_memory(model_size, device)
        """

        sd, metadata = load_torch_file(str(model_path), return_metadata=True)
        self.model_sd = sd
        self.metadata = metadata
        self.model_path = model_path
        self.device = device
        self.cpu_offload = cpu_offload_enabled
        self.data_type = data_type



        sd = self.model_sd
        metadata = self.metadata



        #non-functional
        '''
        if attention == "nunchaku-fp16":
            transformer.set_attention_impl("nunchaku-fp16")
        else:
            assert attention == "flash-attention2"
            transformer.set_attention_impl("flashattn2")
        '''
        
    
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        rank = quantization_config.get("rank", 32)

        precision = get_precision()
        if precision == "fp4":
            precision = "nvfp4"        

        dtype_inputs = {"bfloat16": torch.bfloat16, "float16": torch.float16}


        model_options = {"precision": precision, "rank": rank, "cache_threshold": cache_threshold, "dtype": dtype_inputs.get(data_type)}
        
        model = load_diffusion_model_state_dict(sd, model_options)



        return (model,)