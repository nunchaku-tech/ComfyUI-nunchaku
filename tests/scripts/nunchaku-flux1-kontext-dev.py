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
        dualcliploader_38 = dualcliploader.load_clip(
            clip_name1="clip_l.safetensors",
            clip_name2="t5xxl_fp8_e4m3fn_scaled.safetensors",
            type="flux",
            device="default",
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text="Make the creature hold a sign that says 'Nunchaku is awesome', yarn art style, detailed, vibrant colors. Do not change other features.",
            clip=get_value_at_index(dualcliploader_38, 0),
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_39 = vaeloader.load_vae(vae_name="ae.safetensors")

        loadimageoutput = NODE_CLASS_MAPPINGS["LoadImageOutput"]()
        loadimageoutput_142 = loadimageoutput.load_image(image="yarn-art-pikachu.png [output]")

        imagestitch = NODE_CLASS_MAPPINGS["ImageStitch"]()
        imagestitch_146 = imagestitch.stitch(
            direction="right",
            match_image_size=True,
            spacing_width=0,
            spacing_color="white",
            image1=get_value_at_index(loadimageoutput_142, 0),
        )

        fluxkontextimagescale = NODE_CLASS_MAPPINGS["FluxKontextImageScale"]()
        fluxkontextimagescale_42 = fluxkontextimagescale.scale(image=get_value_at_index(imagestitch_146, 0))

        vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
        vaeencode_124 = vaeencode.encode(
            pixels=get_value_at_index(fluxkontextimagescale_42, 0),
            vae=get_value_at_index(vaeloader_39, 0),
        )

        nunchakufluxditloader = NODE_CLASS_MAPPINGS["NunchakuFluxDiTLoader"]()
        nunchakufluxditloader_189 = nunchakufluxditloader.load_model(
            model_path=f"svdq-{precision}_r32-flux.1-kontext-dev.safetensors",
            cache_threshold=0,
            attention="nunchaku-fp16",
            cpu_offload="auto",
            device_id=0,
            data_type="bfloat16",
            i2f_mode="enabled",
        )

        referencelatent = NODE_CLASS_MAPPINGS["ReferenceLatent"]()
        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            referencelatent_177 = referencelatent.append(
                conditioning=get_value_at_index(cliptextencode_6, 0),
                latent=get_value_at_index(vaeencode_124, 0),
            )

            fluxguidance_35 = fluxguidance.append(guidance=2.5, conditioning=get_value_at_index(referencelatent_177, 0))

            conditioningzeroout_135 = conditioningzeroout.zero_out(conditioning=get_value_at_index(cliptextencode_6, 0))

            ksampler_31 = ksampler.sample(
                seed=3569396000607825004,
                steps=20,
                cfg=1,
                sampler_name="euler",
                scheduler="simple",
                denoise=1,
                model=get_value_at_index(nunchakufluxditloader_189, 0),
                positive=get_value_at_index(fluxguidance_35, 0),
                negative=get_value_at_index(conditioningzeroout_135, 0),
                latent_image=get_value_at_index(vaeencode_124, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_31, 0),
                vae=get_value_at_index(vaeloader_39, 0),
            )

            saveimage_136 = saveimage.save_images(filename_prefix="ComfyUI", images=get_value_at_index(vaedecode_8, 0))

            filename = saveimage_136["ui"]["images"][0]["filename"]
            path = os.path.join("output", filename)
            with open("image_path.txt", "w") as f:
                f.write(path)
            print(path)
            return path


if __name__ == "__main__":
    main(get_precision())
