PuLID Nodes
===========

.. _nunchaku-flux-pulid-apply-v2:

Nunchaku FLUX PuLID Apply V2
----------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/nodes/NunchakuFluxPuLIDApplyV2.png
    :alt: NunchakuFluxPuLIDApplyV2

A node for applying PuLID identity customization to a Nunchaku FLUX model according to a reference image.

**Inputs:**

- **model**: The Nunchaku FLUX model to modify (must be loaded by Nunchaku FLUX DiT Loader).
- **pulid_pipline**: The PuLID pipeline instance (from :ref:`nunchaku-pulid-loader-v2`).
- **image**: The input image for identity embedding extraction.
- **weight**: How strongly to apply the PuLID effect. Range: -1.0 to 5.0, default: 1.0.
- **start_at**: When to start applying PuLID during the denoising process. Range: 0.0 to 1.0, default: 0.0.
- **end_at**: When to stop applying PuLID during the denoising process. Range: 0.0 to 1.0, default: 1.0.
- **attn_mask** (optional): Attention mask for selective application. Currently not supported.
- **options** (optional): Additional options for PuLID processing.

**Outputs:**

- **model**: The modified diffusion model with PuLID applied.

.. seealso::

    API reference: :class:`~comfyui_nunchaku.nodes.models.pulid.NunchakuFluxPuLIDApplyV2`.

    Example workflow: :ref:`nunchaku-flux.1-dev-pulid-json`.

.. _nunchaku-pulid-loader-v2:

Nunchaku PuLID Loader V2
------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/nodes/NunchakuPuLIDLoaderV2.png
    :alt: NunchakuPuLIDLoaderV2

A node for loading the PuLID pipeline required for identity-preserving image generation.

**Inputs:**

- **model**: The base Nunchaku FLUX model to apply PuLID to (must be loaded by :ref:`nunchaku-flux-dit-loader`).
- **pulid_file**: PuLID model weights file. You can download this file from `Hugging Face <https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors>`__ and place it under ``models/pulid``.
- **eva_clip_file**: EVA-CLIP model weights file. Download it from `Hugging Face <https://huggingface.co/QuanSun/EVA-CLIP/blob/main/EVA02_CLIP_L_336_psz14_s6B.pt>`__ and place it in the ``models/clip`` directory. Autodownload is also supported.
- **insight_face_provider**: ONNX provider for InsightFace. Choose ``gpu`` for CUDA or ``cpu`` for CPU inference.

**Outputs:**

- **model**: PuLID injected Nunchaku FLUX model.
- **pulid_pipeline**: The loaded PuLID pipeline, ready for use with PuLID Apply nodes.


.. note::

    PuLID requires the `AntelopeV2 <https://huggingface.co/MonsterMMORPG/tools/tree/main>`__ ONNX models in ``models/insightface/models/antelopev2`` (autodownload supported).

    The following FaceXLib models are also required in ``models/facexlib`` (autodownload supported):

    - `parsing_bisenet <https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth>`__
    - `parsing_parsenet <https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth>`__
    - `Resnet50 <https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth>`__

    See the https://github.com/lldacing/ComfyUI_PuLID_Flux_ll?tab=readme-ov-file#pulid-models for details.

.. seealso::

    API reference: :class:`~comfyui_nunchaku.nodes.models.pulid.NunchakuPuLIDLoaderV2`.

    Example workflow: :ref:`nunchaku-flux.1-dev-pulid-json`.


.. _nunchaku-pulid-loader:

Nunchaku PuLID Loader (Deprecated)
----------------------------------

.. warning::
    This node is **deprecated** and will be removed in December 2025.
    Please use :ref:`nunchaku-pulid-loader-v2` instead.

A legacy node for loading the PuLID pipeline for a Nunchaku FLUX model. This node loads the PuLID model and required face libraries, returning both the original model and a ready-to-use PuLID pipeline.

**Inputs:**

- **model**: The base Nunchaku FLUX model to apply PuLID to (must be loaded by :ref:`nunchaku-flux-dit-loader`).

**Outputs:**

- **model**: The input Nunchaku FLUX model (unchanged).
- **pulid**: The loaded PuLID pipeline.

.. seealso::

    API reference: :class:`~comfyui_nunchaku.nodes.models.pulid.NunchakuPulidLoader`.

----

.. _nunchaku-pulid-apply:

Nunchaku PuLID Apply (Deprecated)
---------------------------------

.. warning::
    This node is **deprecated** and will be removed in December 2025.
    Please use :ref:`nunchaku-flux-pulid-apply-v2` instead.

A legacy node for applying PuLID identity embeddings to a Nunchaku FLUX model.

**Inputs:**

- **pulid**: The PuLID pipeline instance (from :ref:`nunchaku-pulid-loader`).
- **image**: The image to encode for identity.
- **model**: The Nunchaku FLUX model to modify.
- **ip_weight**: The weight for the identity embedding (default: 1.0, range: 0.0–2.0).

**Outputs:**

- **model**: The updated model with PuLID applied.

.. seealso::

    API reference: :class:`~comfyui_nunchaku.nodes.models.pulid.NunchakuPulidApply`.
