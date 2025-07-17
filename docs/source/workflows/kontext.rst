FLUX Kontext
============

.. _nunchaku-flux.1-dev-kontext-json:

`nunchaku-flux.1-dev-kontext.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-dev-kontext.json>`__
---------------------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-dev-kontext.png
    :alt: nunchaku-flux.1-dev-kontext.json

Image editing workflow using the Nunchaku FLUX.1-Kontext-dev model.

**Links:**

- Nunchaku FLUX.1-Kontext-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-kontext-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-kontext-dev>`
  (Place in ``models/diffusion_models``)
- Example input image: :download:`yarn-art-pikachu.png <https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png>`

.. seealso::
    See node :ref:`nunchaku-flux-dit-loader`.

.. _nunchaku-flux.1-kontext-dev-turbo_lora-json:

`nunchaku-flux.1-kontext-dev-turbo_lora.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-kontext-dev-turbo_lora.json>`__
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-kontext-dev-turbo_lora.png
    :alt: nunchaku-flux.1-kontext-dev-turbo_lora.json

Image editing workflow using the Nunchaku FLUX.1-Kontext-dev model with FLUX.1-Turbo-Alpha LoRA acceleration.

**Links:**

- Nunchaku FLUX.1-Kontext-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-kontext-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-kontext-dev>`
  (Place in ``models/diffusion_models``)
- FLUX.1-Turbo-Alpha LoRA: :download:`Hugging Face <https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/blob/main/diffusion_pytorch_model.safetensors>`
  (Place in ``models/loras``)
- Example input image: :download:`yarn-art-pikachu.png <https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png>`

.. note::
    If you disable the FLUX.1-Turbo-Alpha LoRA, increase inference steps to at least 20.

.. seealso::
    See nodes :ref:`nunchaku-flux-dit-loader`, :ref:`nunchaku-flux-lora-loader`.
