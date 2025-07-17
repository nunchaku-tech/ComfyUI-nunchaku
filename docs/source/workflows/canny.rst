FLUX Canny
==========

.. _nunchaku-flux.1-canny-json:

nunchaku-flux.1-canny.json
--------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-canny.png
    :alt: nunchaku-flux.1-canny.json
    :target: https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-canny.json

Image-to-image workflow for style transfer using Canny edge detection with the Nunchaku FLUX.1-Canny-dev model.

**Links:**

- Workflow: :download:`nunchaku-flux.1-canny.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-canny.json>`
- Nunchaku FLUX.1-Canny-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-canny-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-canny-dev>`
  (Place in ``models/diffusion_models``)
- Example input image: :download:`robot.png <https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/ComfyUI-nunchaku/inputs/robot.png>`

.. seealso::
    See node :ref:`nunchaku-flux-dit-loader`.

.. _nunchaku-flux.1-canny-lora-json:

nunchaku-flux.1-canny-lora.json
-------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-canny-lora.png
    :alt: nunchaku-flux.1-canny-lora.json
    :target: https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-canny-lora.json

Image-to-image workflow for style transfer using Canny edge detection with the Nunchaku FLUX.1-dev model and FLUX.1-Canny-dev LoRA.

**Links:**

- Workflow: :download:`nunchaku-flux.1-canny-lora.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-canny-lora.json>`
- Nunchaku FLUX.1-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-dev>`
  (Place in ``models/diffusion_models``)
- FLUX.1-Canny-dev LoRA: :download:`Hugging Face <https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora>`
  (Place in ``models/loras``)
- Example input image: :download:`robot.png <https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/ComfyUI-nunchaku/inputs/robot.png>`

.. seealso::
    See nodes :ref:`nunchaku-flux-dit-loader`, :ref:`nunchaku-flux-lora-loader`.
