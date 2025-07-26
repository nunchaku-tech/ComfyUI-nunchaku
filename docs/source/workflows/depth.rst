FLUX Depth
==========

.. _nunchaku-flux.1-depth-json:

nunchaku-flux.1-depth.json
--------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-depth.png
    :alt: nunchaku-flux.1-depth.json
    :target: https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-depth.json

Image-to-image workflow for style transfer using depth detection with the Nunchaku FLUX.1-Depth-dev model.

**Links:**

- Workflow: :download:`nunchaku-flux.1-depth.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-depth.json>`
- Nunchaku FLUX.1-Depth-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-depth-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-depth-dev>`
  (Place in ``models/diffusion_models``)
- Example input image: :download:`logo.png <https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/ComfyUI-nunchaku/inputs/logo.png>`

.. note::

   You need to install `comfyui_controlnet_aux <https://github.com/Fannovel16/comfyui_controlnet_aux>`_ to use this workflow.

.. seealso::
    See node :ref:`nunchaku-flux-dit-loader`.

.. _nunchaku-flux.1-depth-lora-json:

nunchaku-flux.1-depth-lora.json
-------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-depth-lora.png
    :alt: nunchaku-flux.1-depth-lora.json
    :target: https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-depth-lora.json

Image-to-image workflow for style transfer using depth detection with the Nunchaku FLUX.1-dev model and FLUX.1-Depth-dev LoRA.

**Links:**

- Workflow: :download:`nunchaku-flux.1-depth-lora.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-depth-lora.json>`
- Nunchaku FLUX.1-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-dev>`
  (Place in ``models/diffusion_models``)
- FLUX.1-Depth-dev LoRA: :download:`Hugging Face <https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora>`
  (Place in ``models/loras``)
- Example input image: :download:`logo.png <https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/ComfyUI-nunchaku/inputs/logo.png>`

.. note::

   You need to install `comfyui_controlnet_aux <https://github.com/Fannovel16/comfyui_controlnet_aux>`_ to use this workflow.

.. seealso::
    See nodes :ref:`nunchaku-flux-dit-loader`, :ref:`nunchaku-flux-lora-loader`.
