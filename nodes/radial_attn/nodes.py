"""
ComfyUI Radial Attention Nodes
Provides radial attention optimization for video generation models like WAN 2.1
"""

from unittest.mock import patch

import comfy
import torch
from comfy.ldm.wan.model import WanModel, sinusoidal_embedding_1d

try:
    from .radial_attn_core import MaskMap, RadialAttention
except ImportError:
    print("Warning: RadialAttention and MaskMap not found. Please install the required dependencies.")
    RadialAttention = None
    MaskMap = None

# Global initialization for original functions
_initialized = False
_original_functions = {}
if not _initialized:
    _original_functions["orig_attention"] = comfy.ldm.modules.attention.optimized_attention
    _initialized = True


def get_radial_attn_func(video_token_num, num_frame, block_size, decay_factor, ra_options):
    if RadialAttention is None or MaskMap is None:
        raise ImportError("RadialAttention or MaskMap not available. Please install required dependencies.")

    mask_map = MaskMap(video_token_num, num_frame)

    @torch.compiler.disable()
    def radial_attn_func(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False):
        assert mask is None and skip_reshape is False and skip_output_reshape is False

        if q.shape != k.shape:
            return _original_functions.get("orig_attention")(q, k, v, heads)

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))

        use_dense = ra_options.get("current_use_dense", False)
        sparsity_type = "dense" if use_dense else "radial"

        out = RadialAttention(
            q,
            k,
            v,
            mask_map=mask_map,
            sparsity_type=sparsity_type,
            block_size=block_size,
            decay_factor=decay_factor,
            model_type="wan",
        )

        result = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return result

    return radial_attn_func


def _WanModel_forward_orig(self, x, t, context, clip_fea=None, freqs=None, transformer_options={}, **kwargs):

    ra_options = transformer_options["radial_attn"]

    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]
    x = x.flatten(2).transpose(1, 2)

    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    context = self.text_embedding(context)

    context_img_len = None
    if clip_fea is not None and self.img_emb is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})

    original_attention = comfy.ldm.wan.model.optimized_attention
    comfy.ldm.wan.model.optimized_attention = ra_options["radial_attn_func"]

    try:
        for i, block in enumerate(self.blocks):
            use_dense = i in ra_options["dense_layers"] or ra_options["use_dense_timestep"]

            # Store the dense flag for the radial attention function to use
            ra_options["current_use_dense"] = use_dense

            if ("double_block", i) in blocks_replace:

                def block_wrap(args):
                    return {
                        "img": block(
                            args["img"],
                            context=args["txt"],
                            e=args["vec"],
                            freqs=args["pe"],
                            context_img_len=context_img_len,
                        )
                    }

                out = blocks_replace[("double_block", i)](
                    {"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap}
                )
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

        x = self.head(x, e)
        result = self.unpatchify(x, grid_sizes)
        return result
    finally:
        comfy.ldm.wan.model.optimized_attention = original_attention


class PatchRadialAttn:
    """
    ComfyUI node for applying radial attention optimization to WAN models.

    This node patches WAN diffusion models to use radial sparse attention patterns
    during generation. It provides significant memory reduction and speedup for long
    video sequences while maintaining visual quality.

    The node allows fine-grained control over which layers and timesteps use
    radial attention, enabling optimal performance/quality tradeoffs.

    Features:
        - Selective layer application (specify which transformer layers to optimize)
        - Timestep-aware switching (dense early steps, sparse later steps)
        - Configurable sparsity patterns (block size, decay factor)
        - Automatic fallback to dense attention for cross-attention layers
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "dense_layers": (
                    "STRING",
                    {
                        "default": "1,2",
                        "tooltip": "Comma-separated layer indices to apply radial attention (e.g., '1,2,3')",
                    },
                ),
                "dense_timesteps": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Apply radial attn from which timestep.",
                    },
                ),
                "block_size": ([64, 128], {"default": 128}),
                "decay_factor": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "Lower is faster, higher is more accurate.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_radial_attn"
    CATEGORY = "RadialAttn"
    DESCRIPTION = "Apply radial attention to WAN models with proper forward patching"

    def patch_radial_attn(self, model, dense_layers, dense_timesteps, block_size, decay_factor):
        """
        Apply radial attention patching to a WAN model.

        Args:
            model (MODEL): ComfyUI model object containing WAN diffusion model
            dense_layers (str): Comma-separated layer indices (e.g., "1,2,3")
            dense_timesteps (int): Apply radial attention after this timestep
            block_size (int): Attention block size (64 or 128)
            decay_factor (float): Attention decay factor (0.0-1.0)

        Returns:
            tuple: (patched_model,) - Model with radial attention applied

        Raises:
            AssertionError: If model is not a WAN model
            ValueError: If dense_layers format is invalid
        """
        model = model.clone()

        diffusion_model = model.get_model_object("diffusion_model")
        assert type(diffusion_model) is WanModel, f"Expected WanModel, got {type(diffusion_model)}"

        # Parse dense_layers string into list of integers
        try:
            dense_layers_list = [int(x.strip()) for x in dense_layers.split(",") if x.strip()]
        except ValueError:
            raise ValueError(
                f"Invalid dense_layers format: {dense_layers}. Expected comma-separated integers like '1,2,3'"
            )

        if "transformer_options" not in model.model_options:
            model.model_options["transformer_options"] = {}
        if "radial_attn" not in model.model_options["transformer_options"]:
            model.model_options["transformer_options"]["radial_attn"] = {}
        ra_options = model.model_options["transformer_options"]["radial_attn"]

        ra_options["patch_size"] = diffusion_model.patch_size
        ra_options["dense_layers"] = dense_layers_list
        ra_options["dense_timesteps"] = dense_timesteps
        ra_options["block_size"] = block_size
        ra_options["decay_factor"] = decay_factor

        # Patch the WAN model forward function
        context = patch.multiple(
            diffusion_model, forward_orig=_WanModel_forward_orig.__get__(diffusion_model, diffusion_model.__class__)
        )

        def unet_wrapper_function(model_function, kwargs):
            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            sigmas = c["transformer_options"]["sample_sigmas"]

            current_step_index = 0
            for i in range(len(sigmas) - 1):
                if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                    current_step_index = i
                    break

            ra_options = c["transformer_options"]["radial_attn"]
            patch_size = ra_options["patch_size"]
            num_frame = (input.shape[2] - 1) // patch_size[0] + 1
            frame_size = (input.shape[3] // patch_size[1]) * (input.shape[4] // patch_size[2])
            video_token_num = frame_size * num_frame

            ra_options["use_dense_timestep"] = current_step_index < ra_options["dense_timesteps"]
            # use_dense = ra_options["use_dense_timestep"]
            ra_options["radial_attn_func"] = get_radial_attn_func(
                video_token_num, num_frame, ra_options["block_size"], ra_options["decay_factor"], ra_options
            )
            with context:
                result = model_function(input, timestep, **c)
            return result

        model.set_model_unet_function_wrapper(unet_wrapper_function)

        print("✅ RADIAL ATTENTION PATCHED:")
        print(f"   🔧 Dense layers: {dense_layers_list}")
        print(f"   ⏰ Dense timesteps: {dense_timesteps}")
        print(f"   🧱 Block size: {block_size}")
        print(f"   📉 Decay factor: {decay_factor}")

        return (model,)


class RadialAttentionInfo:
    """
    ComfyUI node for displaying radial attention configuration information.

    This utility node shows the current radial attention settings and provides
    useful information about token dimensions, layer configuration, and
    dependency status. It helps users understand how their settings will
    affect the generation process.

    The node calculates video token dimensions based on WAN model patch sizes
    and displays configuration in a human-readable format.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_length": ("INT", {"default": 81, "min": 1, "max": 1024}),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
                "num_frames": ("INT", {"default": 16, "min": 1, "max": 256}),
                "block_size": ([64, 128], {"default": 128}),
                "decay_factor": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1}),
                "dense_layers": ("STRING", {"default": "1,2"}),
                "dense_timesteps": ("INT", {"default": 2, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_info"
    CATEGORY = "RadialAttn"
    OUTPUT_NODE = True

    def get_info(
        self, video_length, width, height, num_frames, block_size, decay_factor, dense_layers, dense_timesteps
    ):
        """
        Generate radial attention configuration information.

        Args:
            video_length (int): Video length in tokens
            width (int): Video width in pixels
            height (int): Video height in pixels
            num_frames (int): Number of video frames
            block_size (int): Attention block size
            decay_factor (float): Attention decay factor
            dense_layers (str): Comma-separated layer indices
            dense_timesteps (int): Dense timestep threshold

        Returns:
            tuple: (info_string,) - Formatted configuration information
        """
        # Parse dense_layers
        try:
            dense_layers_list = [int(x.strip()) for x in dense_layers.split(",") if x.strip()]
        except ValueError:
            dense_layers_list = [1, 2]

        # Calculate token dimensions (WAN model patch size is 16x16 for spatial, 4 for temporal)
        patch_size_spatial = 16  # WAN model spatial patch size
        patch_size_temporal = 4  # WAN model temporal patch size

        height_tokens = height // patch_size_spatial
        width_tokens = width // patch_size_spatial
        temporal_tokens = ((video_length - 1) // patch_size_temporal) + 1
        video_token_num = height_tokens * width_tokens * temporal_tokens

        # Calculate per-frame tokens
        tokens_per_frame = video_token_num // num_frames

        info = f"""Radial Attention Configuration:

Video Settings:
  Resolution: {width}x{height}x{video_length}
  Frames: {num_frames}
  Tokens per frame: {tokens_per_frame}
  Total video tokens: {video_token_num}
  Token dimensions: {height_tokens}×{width_tokens}×{temporal_tokens}

Radial Attention Settings:
  Dense layers: {dense_layers_list}
  Dense timesteps: {dense_timesteps} (radial attn applied after this step)
  Block size: {block_size}
  Decay factor: {decay_factor}

Dependencies:
  RadialAttention: {'✓ Available' if RadialAttention is not None else '✗ Not found'}
  MaskMap: {'✓ Available' if MaskMap is not None else '✗ Not found'}
"""

        return (info,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "PatchRadialAttention": PatchRadialAttn,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PatchRadialAttention": "Patch Radial Attention",
}
