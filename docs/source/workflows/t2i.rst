FLUX Text-to-Image
==================

.. _nunchaku-flux.1-dev-json:

`nunchaku-flux.1-dev.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-dev.json>`__
-----------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-dev.png
    :alt: nunchaku-flux.1-dev.json

This workflow demonstrates how to use the Nunchaku FLUX.1-dev model with multiple LoRAs to generate images from a given prompt.

You can download the Nunchaku FLUX.1-dev models from `Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-dev>`__ or `ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-dev>`__ and place them in ``models/diffusion_models``.

Example LoRA weights are available at:

- `flux1-turbo.safetensors <https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/blob/main/diffusion_pytorch_model.safetensors>`__
- `diffusers-ghibsky.safetensors <https://huggingface.co/aleksa-codes/flux-ghibsky-illustration/blob/main/lora.safetensors>`__

.. note::
    If you disable the `FLUX.1-Turbo-Alpha LoRA <https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/blob/main/diffusion_pytorch_model.safetensors>`__, please increase the number of inference steps to at least 20.

.. seealso::
    See nodes :ref:`nunchaku-flux-dit-loader`, :ref:`nunchaku-flux-lora-loader`.

.. _nunchaku-flux.1-schnell-json:

`nunchaku-flux.1-schnell.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-schnell.json>`__
-------------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-schnell.png
    :alt: nunchaku-flux.1-schnell.json

This workflow demonstrates how to use the Nunchaku FLUX.1-schnell model to generate images from a given prompt.

You can download the Nunchaku FLUX.1-schnell models from `Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-schnell>`__ or `ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-schnell>`__ and place them in ``models/diffusion_models``.

.. seealso::

    See node :ref:`nunchaku-flux-dit-loader`.

.. _nunchaku-flux.1-dev-qencoder-json:

`nunchaku-flux.1-dev-qencoder.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-dev-qencoder.json>`__
-----------------------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-dev-qencoder.png
    :alt: nunchaku-flux.1-dev-qencoder.json

This workflow demonstrates how to use the Nunchaku FLUX.1-dev model with a 4-bit T5 text encoder for text-to-image generation.

Model Links:

- Nunchaku FLUX.1-dev: `Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-dev>`__, `ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-dev>`__. Place the model in ``models/diffusion_models``.
- 4-bit T5 encoder: `Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-t5>`__, `ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-t5>`__. Place the model in ``models/text_encoders``.

.. seealso::

    See nodes :ref:`nunchaku-flux-dit-loader`, :ref:`nunchaku-text-encoder-loader-v2`.
