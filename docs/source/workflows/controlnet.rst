FLUX ControlNets
================


.. _nunchaku-flux.1-controlnet-union-pro2-json:

`nunchaku-flux.1-controlnet-union-pro2.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-controlnet-union-pro2.json>`__
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-controlnet-union-pro2.png
    :alt: nunchaku-flux.1-controlnet-union-pro2.json

A workflow demonstrating the use of the **FLUX.1-ControlNet-Union-Pro-2.0** model with Nunchaku FLUX.1-dev for advanced image generation and control.

**Links:**

- Nunchaku FLUX.1-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-dev>`
  (Place in ``models/diffusion_models``)
- ControlNet-Union-Pro-2.0: :download:`Hugging Face <https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0/blob/main/diffusion_pytorch_model.safetensors>` (Place in ``models/controlnet``)
- Example input image: :download:`mushroom_depth.webp <https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/ComfyUI-nunchaku/test_data/mushroom_depth.webp>`

.. seealso::
    See node :ref:`nunchaku-flux-dit-loader`.

----

.. _nunchaku-flux.1-controlnet-upscaler-json:

`nunchaku-flux.1-controlnet-upscaler.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-dev-controlnet-upscaler.json>`__
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-controlnet-upscaler.png
    :alt: nunchaku-flux.1-controlnet-upscaler.json

A workflow for using the **FLUX.1-ControlNet-Upscaler** model with Nunchaku to upscale images with fine control.

**Links:**

- Nunchaku FLUX.1-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-dev>`
  (Place in ``models/diffusion_models``)
- ControlNet-Upscaler: :download:`Hugging Face <https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/blob/main/diffusion_pytorch_model.safetensors>` (Place in ``models/controlnet`` and rename to ``controlnet-upscaler.safetensors``)
- Example input image: :download:`robot.png <https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/ComfyUI-nunchaku/test_data/robot.png>`

.. seealso::
    See node :ref:`nunchaku-flux-dit-loader`.
