FLUX Kontext Workflows
======================

.. _nunchaku-flux.1-dev-kontext-json:

`nunchaku-flux.1-dev-kontext.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-dev-kontext.json>`__
---------------------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-dev-kontext.png
    :alt: nunchaku-flux.1-dev-kontext.json

This workflow demonstrates how to use the Nunchaku FLUX.1-Kontext-dev model to edit images with a given prompt.

You can download the Nunchaku FLUX.1-Kontext-dev models from `Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-kontext-dev>`__ or `ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-kontext-dev>`__.

Example input image: :download:`yarn-art-pikachu.png <https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png>`

.. seealso::

    See node :ref:`nunchaku-flux-dit-loader`.

.. _nunchaku-flux.1-kontext-dev-turbo_lora-json:

`nunchaku-flux.1-kontext-dev-turbo_lora.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-kontext-dev-turbo_lora.json>`__
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-kontext-dev-turbo_lora.png
    :alt: nunchaku-flux.1-kontext-dev-turbo_lora.json

This workflow demonstrates how to use the Nunchaku FLUX.1-Kontext-dev model with the `FLUX.1-Turbo-Alpha LoRA <turbo_lora_>`_ acceleration to edit images with a given prompt.

You can download the Nunchaku FLUX.1-Kontext-dev models from `Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-kontext-dev>`__ or `ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-kontext-dev>`__ and place them in ``models/diffusion_models``.

Example input image: :download:`yarn-art-pikachu.png <https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png>`


.. note::
    If you disable the `FLUX.1-Turbo-Alpha LoRA <turbo_lora_>`_, please increase the number of inference steps to at least 20.

.. seealso::

    See node :ref:`nunchaku-flux-dit-loader` and :ref:`nunchaku-flux-lora-loader`.