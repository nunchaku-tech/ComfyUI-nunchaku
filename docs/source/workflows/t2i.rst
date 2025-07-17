FLUX Text-to-Image
==================

.. _nunchaku-flux.1-dev-json:

`nunchaku-flux.1-dev.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-dev.json>`__
-----------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-dev.png
    :alt: nunchaku-flux.1-dev.json

Text-to-image workflow using the Nunchaku FLUX.1-dev model with multiple LoRAs.

**Links:**

- Nunchaku FLUX.1-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-dev>`
  (Place in ``models/diffusion_models``)
- Example LoRAs (Place in ``models/loras``):

  - :download:`FLUX.1-Turbo-Alpha <https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/blob/main/diffusion_pytorch_model.safetensors>`
  - :download:`Ghibsky Illustration <https://huggingface.co/aleksa-codes/flux-ghibsky-illustration/blob/main/lora.safetensors>`

.. note::
    If you disable the FLUX.1-Turbo-Alpha LoRA, increase inference steps to at least 20.

.. seealso::
    See nodes :ref:`nunchaku-flux-dit-loader`, :ref:`nunchaku-flux-lora-loader`.

.. _nunchaku-flux.1-schnell-json:

`nunchaku-flux.1-schnell.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-schnell.json>`__
-------------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-schnell.png
    :alt: nunchaku-flux.1-schnell.json

Text-to-image workflow using the Nunchaku FLUX.1-schnell model.

**Links:**

- Nunchaku FLUX.1-schnell: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-schnell>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-schnell>`
  (Place in ``models/diffusion_models``)

.. seealso::

    See node :ref:`nunchaku-flux-dit-loader`.

.. _nunchaku-flux.1-dev-qencoder-json:

`nunchaku-flux.1-dev-qencoder.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-dev-qencoder.json>`__
-----------------------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-dev-qencoder.png
    :alt: nunchaku-flux.1-dev-qencoder.json

Text-to-image workflow using FLUX.1-dev with a 4-bit T5 text encoder.

**Links:**

- Nunchaku FLUX.1-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-dev>`
  (Place in ``models/diffusion_models``)
- 4-bit T5 encoder: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-t5>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-t5>`
  (Place in ``models/text_encoders``)

.. seealso::
    See nodes :ref:`nunchaku-flux-dit-loader`, :ref:`nunchaku-text-encoder-loader-v2`
