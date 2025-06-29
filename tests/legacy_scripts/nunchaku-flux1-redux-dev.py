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
        dualcliploader_11 = dualcliploader.load_clip(
            clip_name1="t5xxl_fp16.safetensors",
            clip_name2="clip_l.safetensors",
            type="flux",
            device="default",
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text="cute anime girl with massive fluffy fennec ears",
            clip=get_value_at_index(dualcliploader_11, 0),
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_10 = vaeloader.load_vae(vae_name="ae.safetensors")

        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        ksamplerselect_16 = ksamplerselect.get_sampler(sampler_name="euler")

        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        randomnoise_25 = randomnoise.get_noise(noise_seed=68098388376617147)

        emptysd3latentimage = NODE_CLASS_MAPPINGS["EmptySD3LatentImage"]()
        emptysd3latentimage_27 = emptysd3latentimage.generate(width=1024, height=1024, batch_size=1)

        clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
        clipvisionloader_38 = clipvisionloader.load_clip(clip_name="sigclip_vision_patch14_384.safetensors")

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_40 = loadimage.load_image(image="robot.png")

        clipvisionencode = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()
        clipvisionencode_39 = clipvisionencode.encode(
            crop="center",
            clip_vision=get_value_at_index(clipvisionloader_38, 0),
            image=get_value_at_index(loadimage_40, 0),
        )

        stylemodelloader = NODE_CLASS_MAPPINGS["StyleModelLoader"]()
        stylemodelloader_42 = stylemodelloader.load_style_model(style_model_name="flux1-redux-dev.safetensors")

        nunchakufluxditloader = NODE_CLASS_MAPPINGS["NunchakuFluxDiTLoader"]()
        nunchakufluxditloader_45 = nunchakufluxditloader.load_model(
            model_path=f"svdq-{precision}-flux.1-dev",
            cache_threshold=0,
            attention="nunchaku-fp16",
            cpu_offload="auto",
            device_id=0,
            data_type="bfloat16",
            i2f_mode="enabled",
        )

        modelsamplingflux = NODE_CLASS_MAPPINGS["ModelSamplingFlux"]()
        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        stylemodelapply = NODE_CLASS_MAPPINGS["StyleModelApply"]()
        basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
        basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            modelsamplingflux_30 = modelsamplingflux.patch(
                max_shift=1.15,
                base_shift=0.5,
                width=1024,
                height=1024,
                model=get_value_at_index(nunchakufluxditloader_45, 0),
            )

            fluxguidance_26 = fluxguidance.append(guidance=3.5, conditioning=get_value_at_index(cliptextencode_6, 0))

            stylemodelapply_41 = stylemodelapply.apply_stylemodel(
                strength=1,
                strength_type="multiply",
                conditioning=get_value_at_index(fluxguidance_26, 0),
                style_model=get_value_at_index(stylemodelloader_42, 0),
                clip_vision_output=get_value_at_index(clipvisionencode_39, 0),
            )

            basicguider_22 = basicguider.get_guider(
                model=get_value_at_index(modelsamplingflux_30, 0),
                conditioning=get_value_at_index(stylemodelapply_41, 0),
            )

            basicscheduler_17 = basicscheduler.get_sigmas(
                scheduler="simple",
                steps=20,
                denoise=1,
                model=get_value_at_index(modelsamplingflux_30, 0),
            )

            samplercustomadvanced_13 = samplercustomadvanced.sample(
                noise=get_value_at_index(randomnoise_25, 0),
                guider=get_value_at_index(basicguider_22, 0),
                sampler=get_value_at_index(ksamplerselect_16, 0),
                sigmas=get_value_at_index(basicscheduler_17, 0),
                latent_image=get_value_at_index(emptysd3latentimage_27, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(samplercustomadvanced_13, 0),
                vae=get_value_at_index(vaeloader_10, 0),
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
