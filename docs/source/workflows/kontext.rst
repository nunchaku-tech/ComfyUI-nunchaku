FLUX Kontext
============

.. _nunchaku-flux.1-kontext-dev-json:

nunchaku-flux.1-dev-kontext.json
--------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-kontext-dev.png
    :alt: nunchaku-flux.1-dev-kontext.json
    :target: https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-kontext-dev.json

Image editing workflow using the Nunchaku FLUX.1-Kontext-dev model.

**Links:**

- Workflow: :download:`nunchaku-flux.1-kontext-dev.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-kontext-dev.json>`
- Nunchaku FLUX.1-Kontext-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-kontext-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-kontext-dev>`
  (Place in ``models/diffusion_models``)
- Example input image: :download:`yarn-art-pikachu.png <https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png>`

.. seealso::
    See node :ref:`nunchaku-flux-dit-loader`.

.. _nunchaku-flux.1-kontext-dev-turbo_lora-json:

nunchaku-flux.1-kontext-dev-turbo_lora.json
-------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-kontext-dev-turbo_lora.png
    :alt: nunchaku-flux.1-kontext-dev-turbo_lora.json
    :target: https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-kontext-dev-turbo_lora.json

Image editing workflow using the Nunchaku FLUX.1-Kontext-dev model with FLUX.1-Turbo-Alpha LoRA acceleration.

**Links:**

- Workflow: :download:`nunchaku-flux.1-kontext-dev-turbo_lora.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-kontext-dev-turbo_lora.json>`
- Nunchaku FLUX.1-Kontext-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-kontext-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-kontext-dev>`
  (Place in ``models/diffusion_models``)
- FLUX.1-Turbo-Alpha LoRA: :download:`Hugging Face <https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/blob/main/diffusion_pytorch_model.safetensors>`
  (Place in ``models/loras``)
- Example input image: :download:`yarn-art-pikachu.png <https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png>`

.. note::

   If you disable the FLUX.1-Turbo-Alpha LoRA, increase inference steps to at least 20.

.. seealso::
    See nodes :ref:`nunchaku-flux-dit-loader`, :ref:`nunchaku-flux-lora-loader`.
