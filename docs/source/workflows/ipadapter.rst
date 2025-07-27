FLUX IP-Adapter
===============

.. _nunchaku-flux.1-ip-adapter-json:

nunchaku-flux.1-ip-adapter.json
-------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-ip-adapter.png
    :alt: nunchaku-flux.1-ip-adapter.json
    :target: https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-ip-adapter.json

Image-based prompting workflow using `IP-Adapter <https://huggingface.co/XLabs-AI/flux-ip-adapter-v2>`__ and the Nunchaku FLUX.1-dev model.

**Links:**

- Workflow: :download:`nunchaku-flux.1-ip-adapter.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-ip-adapter.json>`
- Example input image: :download:`monalisa.jpg <https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/ComfyUI-nunchaku/inputs/monalisa.jpg>`

.. warning::
   This workflow is experimental and currently requires a large amount of VRAM.
   It will automatically download the IP-Adapter and its associated CLIP models
   from `Hugging Face <https://huggingface.co/XLabs-AI/flux-ip-adapter-v2>`__ to the default cache directory.
   At this time, specifying custom model paths is not supported.

.. seealso::
   See nodes :ref:`nunchaku-flux-ip-adapter-loader` and :ref:`nunchaku-flux-ip-adapter-apply`.
