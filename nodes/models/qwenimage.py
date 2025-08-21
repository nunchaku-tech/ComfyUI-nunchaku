import logging
import os

import comfy.model_management
import folder_paths

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuQwenImageDiTLoader:

    def __init__(self):
        """
        Initialize the NunchakuFluxDiTLoader.

        Sets up internal state and selects the default torch device.
        """
        self.transformer = None
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
        Defines the input types and tooltips for the node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and their descriptions for the node interface.
        """
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "The Nunchaku Qwen-Image model."},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Qwen-Image DiT Loader"

    def load_model(self, model_name: str, **kwargs):
        pass
