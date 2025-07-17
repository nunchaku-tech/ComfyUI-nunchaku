FLUX ControlNets
================

.. _nunchaku-flux.1-dev-controlnet-union-pro2-json:

nunchaku-flux.1-dev-controlnet-union-pro2.json
----------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-dev-controlnet-union-pro2.png
    :alt: nunchaku-flux.1-dev-controlnet-union-pro2.json
    :target: https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-dev-controlnet-union-pro2.json

Workflow for advanced image generation and control using the FLUX.1-ControlNet-Union-Pro-2.0 model with Nunchaku FLUX.1-dev.

**Links:**

- Workflow: :download:`nunchaku-flux.1-dev-controlnet-union-pro2.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-dev-controlnet-union-pro2.json>`
- Nunchaku FLUX.1-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-dev>`
  or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-dev>`
  (Place in ``models/diffusion_models``)
- FLUX.1-ControlNet-Union-Pro-2.0: :download:`Hugging Face <https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0/blob/main/diffusion_pytorch_model.safetensors>`
  (Place in ``models/controlnet``)
- Example input image: :download:`mushroom_depth.webp <https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/ComfyUI-nunchaku/inputs/mushroom_depth.webp>`

.. seealso::
    See node :ref:`nunchaku-flux-dit-loader`.

.. _nunchaku-flux.1-dev-controlnet-upscaler-json:

nunchaku-flux.1-dev-controlnet-upscaler.json
--------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-dev-controlnet-upscaler.png
    :alt: nunchaku-flux.1-dev-controlnet-upscaler.json
    :target: https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-dev-controlnet-upscaler.json

Workflow for upscaling images with fine control using the FLUX.1-ControlNet-Upscaler model and Nunchaku FLUX.1-dev.

**Links:**

- Workflow: :download:`nunchaku-flux.1-controlnet-upscaler.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-dev-controlnet-upscaler.json>`
- Nunchaku FLUX.1-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-dev>`
  (Place in ``models/diffusion_models``)
- FLUX.1-ControlNet-Upscaler: :download:`Hugging Face <https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/blob/main/diffusion_pytorch_model.safetensors>`
  (Place in ``models/controlnet`` and rename to ``controlnet-upscaler.safetensors``)
- Example input image: :download:`robot.png <https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/ComfyUI-nunchaku/inputs/robot.png>`

.. seealso::
    See node :ref:`nunchaku-flux-dit-loader`.
