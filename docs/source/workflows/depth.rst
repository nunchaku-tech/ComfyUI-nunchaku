FLUX Depth
==========

.. _nunchaku-flux.1-depth-json:

`nunchaku-flux.1-depth.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-depth.json>`__
---------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-depth.png
    :alt: nunchaku-flux.1-depth.json

Image-to-image workflow for style transfer using depth detection with the Nunchaku FLUX.1-Depth-dev model.

**Links:**

- Nunchaku FLUX.1-Depth-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-depth-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-depth-dev>`
  (Place in ``models/diffusion_models``)
- Example input image: :download:`logo.png <https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/ComfyUI-nunchaku/test_data/logo.png>`

.. seealso::
    See node :ref:`nunchaku-flux-dit-loader`.

.. _nunchaku-flux.1-depth-lora-json:

`nunchaku-flux.1-depth-lora.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-depth-lora.json>`__
-------------------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-depth-lora.png
    :alt: nunchaku-flux.1-depth-lora.json

Image-to-image workflow for style transfer using depth detection with the Nunchaku FLUX.1-dev model and FLUX.1-Depth-dev LoRA.

**Links:**

- Nunchaku FLUX.1-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-dev>`
  (Place in ``models/diffusion_models``)
- FLUX.1-Depth-dev LoRA: :download:`Hugging Face <https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora>`
  (Place in ``models/loras``)
- Example input image: :download:`logo.png <https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/ComfyUI-nunchaku/test_data/logo.png>`

.. seealso::
    See nodes :ref:`nunchaku-flux-dit-loader`, :ref:`nunchaku-flux-lora-loader`.
