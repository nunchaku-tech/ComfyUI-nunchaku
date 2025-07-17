FLUX PuLID
==========

.. _nunchaku-flux.1-dev-pulid-json:

`nunchaku-flux.1-dev-pulid.json <https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-flux.1-dev-pulid.json>`_
-----------------------------------------------------------------------------------------------------------------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-flux.1-dev-pulid.png
    :alt: nunchaku-flux.1-dev-pulid.json

This workflow demonstrates identity-preserving image generation using `PuLID <pulid_paper_>`_.

Used models:

- Nunchaku FLUX.1-dev: `Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-flux.1-dev>`__, `ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-dev>`__ and place them in ``models/diffusion_models``.
- PuLID weights: Download from `Hugging Face <https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors>`__ and place in ``models/pulid``.
- EVA-CLIP weights: Download from `Hugging Face <https://huggingface.co/QuanSun/EVA-CLIP/blob/main/EVA02_CLIP_L_336_psz14_s6B.pt>`__ and place in ``models/clip`` (autodownload supported).
- AntelopeV2 ONNX models: Download from `Hugging Face <https://huggingface.co/MonsterMMORPG/tools/tree/main>`__ and place in ``models/insightface/models/antelopev2`` (autodownload supported).
- FaceXLib models: Place in ``models/facexlib`` (autodownload supported):

  - `parsing_bisenet <https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth>`__
  - `parsing_parsenet <https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth>`__
  - `Resnet50 <https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth>`__

Example input image: :download:`lecun.jpg <https://github.com/ToTheBeginning/PuLID/blob/main/example_inputs/lecun.jpg?raw=true>`

.. seealso::

    Workflow adapted from https://github.com/lldacing/ComfyUI_PuLID_Flux_ll.

    Related nodes: :ref:`nunchaku-flux-dit-loader`, :ref:`nunchaku-flux-pulid-apply-v2`, :ref:`nunchaku-pulid-loader-v2`.
