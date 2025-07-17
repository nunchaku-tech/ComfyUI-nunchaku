FLUX Fill
=========

.. _nunchaku-flux.1-fill-json:

`nunchaku-flux.1-fill.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-fill.json>`__
---------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-fill.png
    :alt: nunchaku-flux.1-fill.json

A workflow of inpainting the image with the prompt using Nunchaku FLUX.1-Fill-dev model.

**Links:**

- Nunchaku FLUX.1-Fill-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-fill-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-fill-dev>`
  (Place in ``models/diffusion_models``)
- Example input image: :download:`strawberry.png <https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/ComfyUI-nunchaku/test_data/strawberry.png>`

.. note::
    You need to install https://github.com/CY-CHENYUE/ComfyUI-InpaintEasy to use this workflow.

.. seealso::
    See node :ref:`nunchaku-flux-dit-loader`.

.. _nunchaku-flux.1-fill-removalV2-json:

`nunchaku-flux.1-fill-removalV2.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-fill-removalV2.json>`__
---------------------------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-fill-removalV2.png
    :alt: nunchaku-flux.1-fill-removalV2.json

A workflow of removing an object from the image using Nunchaku FLUX.1-Fill-dev model with a removal LoRA.

**Links:**

- Nunchaku FLUX.1-Fill-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-fill-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-fill-dev>`
  (Place in ``models/diffusion_models``)
- Removal LoRA: :download:`Hugging Face <https://huggingface.co/lrzjason/ObjectRemovalFluxFill/blob/main/removal_timestep_alpha-2-1740.safetensors>`
  (Place in ``models/loras``)
- Example input image: :download:`removal.png <https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/ComfyUI-nunchaku/test_data/removal.png>`

.. note::
    You need to install https://github.com/CY-CHENYUE/ComfyUI-InpaintEasy to use this workflow.

.. seealso::
    See node :ref:`nunchaku-flux-dit-loader`.