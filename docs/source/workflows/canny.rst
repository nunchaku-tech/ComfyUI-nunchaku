FLUX Canny
==========

.. _nunchaku-flux.1-canny:

`nunchaku-flux.1-canny.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-canny.json>`__
---------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-canny.png
    :alt: nunchaku-flux.1-canny.json

A workflow of converting the image to another style given the prompt using Canny edge detection with Nunchaku FLUX.1-Canny-dev model.

You can download the Nunchaku FLUX.1-Canny-dev models from `Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-canny-dev>`__ or `ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-canny-dev>`__ and place them in ``models/diffusion_models``.

Example input image: :download:`robot.png <https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/ComfyUI-nunchaku/test_data/robot.png>`

.. seealso::

    See node :ref:`nunchaku-flux-dit-loader`.


.. _nunchaku-flux.1-canny-lora-json:

`nunchaku-flux.1-canny-lora.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-canny-lora.json>`__
-------------------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-canny-lora.png
    :alt: nunchaku-flux.1-canny-lora.json

A workflow of converting the image to another style given the prompt using Canny edge detection with Nunchaku FLUX.1-dev model and FLUX.1-Canny-dev LoRA.

You can download the Nunchaku FLUX.1-dev models from `Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-dev>`__ or `ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-dev>`__ and place them in ``models/diffusion_models``.

The FLUX.1-Canny-dev LoRA is available at `Hugging Face <https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora>`__.

Example input image: :download:`robot.png <https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/ComfyUI-nunchaku/test_data/robot.png>`

.. seealso::

    See node :ref:`nunchaku-flux-dit-loader` and :ref:`nunchaku-flux-lora-loader`.
