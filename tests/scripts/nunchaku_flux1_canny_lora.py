import os
import sys
from typing import Any, Mapping, Sequence, Union

import torch

from nunchaku.utils import get_precision


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print("Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead.")
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio

    import execution
    import server

    from nodes import init_extra_nodes

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def main(precision: str):
    import_custom_nodes()
    with torch.inference_mode():
        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_34 = dualcliploader.load_clip(
            clip_name1="clip_l.safetensors",
            clip_name2="t5xxl_fp16.safetensors",
            type="flux",
            device="default",
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_7 = cliptextencode.encode(text="", clip=get_value_at_index(dualcliploader_34, 0))

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_17 = loadimage.load_image(image="robot.png")

        cliptextencode_23 = cliptextencode.encode(
            text="A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts, yarn art style",
            clip=get_value_at_index(dualcliploader_34, 0),
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_32 = vaeloader.load_vae(vae_name="ae.safetensors")

        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        fluxguidance_26 = fluxguidance.append(guidance=30, conditioning=get_value_at_index(cliptextencode_23, 0))

        imagescale = NODE_CLASS_MAPPINGS["ImageScale"]()
        imagescale_38 = imagescale.upscale(
            upscale_method="nearest-exact",
            width=1024,
            height=1024,
            crop="center",
            image=get_value_at_index(loadimage_17, 0),
        )

        canny = NODE_CLASS_MAPPINGS["Canny"]()
        canny_18 = canny.detect_edge(
            low_threshold=0.15,
            high_threshold=0.3,
            image=get_value_at_index(imagescale_38, 0),
        )

        instructpixtopixconditioning = NODE_CLASS_MAPPINGS["InstructPixToPixConditioning"]()
        instructpixtopixconditioning_35 = instructpixtopixconditioning.encode(
            positive=get_value_at_index(fluxguidance_26, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            vae=get_value_at_index(vaeloader_32, 0),
            pixels=get_value_at_index(canny_18, 0),
        )

        nunchakufluxditloader = NODE_CLASS_MAPPINGS["NunchakuFluxDiTLoader"]()
        nunchakufluxditloader_42 = nunchakufluxditloader.load_model(
            model_path=f"svdq-{precision}_r32-flux.1-dev.safetensors",
            cache_threshold=0,
            attention="nunchaku-fp16",
            cpu_offload="auto",
            device_id=0,
            data_type="bfloat16",
            i2f_mode="enabled",
        )

        nunchakufluxloraloader = NODE_CLASS_MAPPINGS["NunchakuFluxLoraLoader"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            nunchakufluxloraloader_43 = nunchakufluxloraloader.load_lora(
                lora_name="flux1-canny-dev-lora.safetensors",
                lora_strength=0.8500000000000002,
                model=get_value_at_index(nunchakufluxditloader_42, 0),
            )

            ksampler_3 = ksampler.sample(
                seed=2367983337527674677,
                steps=20,
                cfg=1,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                model=get_value_at_index(nunchakufluxloraloader_43, 0),
                positive=get_value_at_index(instructpixtopixconditioning_35, 0),
                negative=get_value_at_index(instructpixtopixconditioning_35, 1),
                latent_image=get_value_at_index(instructpixtopixconditioning_35, 2),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(vaeloader_32, 0),
            )

            saveimage_9 = saveimage.save_images(filename_prefix="ComfyUI", images=get_value_at_index(vaedecode_8, 0))
        filename = saveimage_9["ui"]["images"][0]["filename"]
        path = os.path.join("output", filename)
        with open("image_path.txt", "w") as f:
            f.write(path)
        print(path)
        return path


if __name__ == "__main__":
    main(get_precision())
