FLUX Fill
=========

.. _nunchaku-flux.1-fill-json:

nunchaku-flux.1-fill.json
-------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-fill.png
    :alt: nunchaku-flux.1-fill.json
    :target: https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-fill.json

Workflow for inpainting an image using a text prompt with the Nunchaku FLUX.1-Fill-dev model.

**Links:**

- Workflow: :download:`nunchaku-flux.1-fill.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-fill.json>`
- Nunchaku FLUX.1-Fill-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-fill-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-fill-dev>`
  (Place in ``models/diffusion_models``)
- Example input image: :download:`strawberry.png <https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/ComfyUI-nunchaku/inputs/strawberry.png>`

.. note::

   You need to install https://github.com/CY-CHENYUE/ComfyUI-InpaintEasy to use this workflow.

.. seealso::
    See node :ref:`nunchaku-flux-dit-loader`.

.. _nunchaku-flux.1-fill-removalV2-json:

nunchaku-flux.1-fill-removalV2.json
-----------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-fill-removalV2.png
    :alt: nunchaku-flux.1-fill-removalV2.json
    :target: https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-fill-removalV2.json

Workflow for removing an object from an image using the Nunchaku FLUX.1-Fill-dev model with a removal LoRA.

**Links:**

- Workflow: :download:`nunchaku-flux.1-fill-removalV2.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-fill-removalV2.json>`
- Nunchaku FLUX.1-Fill-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-fill-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-fill-dev>`
  (Place in ``models/diffusion_models``)
- Removal LoRA: :download:`Hugging Face <https://huggingface.co/lrzjason/ObjectRemovalFluxFill/blob/main/removal_timestep_alpha-2-1740.safetensors>`
  (Place in ``models/loras``)
- Example input image: :download:`removal.png <https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/ComfyUI-nunchaku/inputs/removal.png>`

.. note::

   You need to install https://github.com/CY-CHENYUE/ComfyUI-InpaintEasy to use this workflow.

.. seealso::
    See node :ref:`nunchaku-flux-dit-loader`.
