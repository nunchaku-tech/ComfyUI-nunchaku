FLUX Depth
==========

.. _nunchaku-flux.1-depth:

`nunchaku-flux.1-depth.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-depth.json>`__
---------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-depth.png
    :alt: nunchaku-flux.1-depth.json

A workflow of converting the image to another style given the prompt using Depth detection with Nunchaku FLUX.1-Depth-dev model.

You can download the Nunchaku FLUX.1-Depth-dev models from `Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-depth-dev>`__ or `ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-depth-dev>`__ and place them in ``models/diffusion_models``.

Example input image: :download:`logo.png <https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/ComfyUI-nunchaku/test_data/logo.png>`

.. seealso::

    See node :ref:`nunchaku-flux-dit-loader`.

.. _nunchaku-flux.1-depth-lora-json:

`nunchaku-flux.1-depth-lora.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-depth-lora.json>`__
-------------------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-depth-lora.png
    :alt: nunchaku-flux.1-depth-lora.json

A workflow of converting the image to another style given the prompt using Depth detection with Nunchaku FLUX.1-dev model and FLUX.1-Depth-dev LoRA.

You can download the Nunchaku FLUX.1-dev models from `Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-dev>`__ or `ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-dev>`__ and place them in ``models/diffusion_models``.

The FLUX.1-Depth-dev LoRA is available at `Hugging Face <https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora>`__.

Example input image: :download:`logo.png <https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/ComfyUI-nunchaku/test_data/logo.png>`

.. seealso::

    See node :ref:`nunchaku-flux-dit-loader` and :ref:`nunchaku-flux-lora-loader`.
