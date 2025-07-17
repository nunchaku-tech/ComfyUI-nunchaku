FLUX PuLID
==========

.. _nunchaku-flux.1-dev-pulid-json:

nunchaku-flux.1-dev-pulid.json
------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-dev-pulid.png
    :alt: nunchaku-flux.1-dev-pulid.json
    :target: https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-dev-pulid.json

Identity-preserving image generation workflow using `PuLID <paper_pulid_>`_ and the Nunchaku FLUX.1-dev model.

**Links:**

- Workflow: :download:`nunchaku-flux.1-dev-pulid.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-dev-pulid.json>`
- Nunchaku FLUX.1-dev: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-dev>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-dev>`
  (Place in ``models/diffusion_models``)
- PuLID weights: :download:`Hugging Face <https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors>`
  (Place in ``models/pulid``)
- EVA-CLIP weights: :download:`Hugging Face <https://huggingface.co/QuanSun/EVA-CLIP/blob/main/EVA02_CLIP_L_336_psz14_s6B.pt>`
  (Place in ``models/clip``; autodownload supported)
- AntelopeV2 ONNX models: :download:`Hugging Face <https://huggingface.co/MonsterMMORPG/tools/tree/main>`
  (Place in ``models/insightface/models/antelopev2``; autodownload supported)
- FaceXLib models (autodownload supported; place in ``models/facexlib``):

  - :download:`parsing_bisenet <https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth>`
  - :download:`parsing_parsenet <https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth>`
  - :download:`Resnet50 <https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth>`

- Example input image: :download:`lecun.jpg <https://github.com/ToTheBeginning/PuLID/blob/main/example_inputs/lecun.jpg?raw=true>`

.. note::

   If you use autodownload, required models will be downloaded automatically on first run.

.. seealso::

   Workflow adapted from https://github.com/lldacing/ComfyUI_PuLID_Flux_ll.

   See nodes: :ref:`nunchaku-flux-dit-loader`, :ref:`nunchaku-flux-pulid-apply-v2`, :ref:`nunchaku-pulid-loader-v2`.
