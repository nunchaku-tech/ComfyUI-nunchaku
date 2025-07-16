Model Nodes
===========

.. _nunchaku-flux-dit-loader:

Nunchaku Flux DiT Loader
------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/nodes/NunchakuFluxDiTLoader.png
    :alt: NunchakuFluxDiTLoader

A node for loading SVDQuant-quantized FLUX.1 models with `nunchaku <nunchaku_repo_>`_ acceleration.
This node manages model loading, device selection, attention implementation, CPU offload, and caching for efficient inference.

**Inputs:**

- **model_path**: The path to the Nunchaku FLUX model folder or `.safetensors` file. You must manually download the model from our `HuggingFace collection <https://huggingface.co/collections/mit-han-lab/nunchaku-6837e7498f680552f7bbb5ad>`_ or `ModelScope collection <https://modelscope.cn/collections/Nunchaku-519fed7f9de94e>`_.

- **cache_threshold**: Adjusts the first-block caching tolerance like `residual_diff_threshold` in WaveSpeed. Increasing the value enhances speed at the cost of quality. A typical setting is 0.12. Setting it to 0 disables the effect.

- **attention**: Attention implementation. Options include `flash-attention2` and `nunchaku-fp16`. The `nunchaku-fp16` uses FP16 attention, offering ~1.2× speedup. Note that 20-series GPUs can only use `nunchaku-fp16`.

- **cpu_offload**: Whether to enable CPU offload for the transformer model. Options include:
  
  - `auto`: Will enable it if the GPU memory is less than 14GiB
  - `enable`: Force enable CPU offload
  - `disable`: Disable CPU offload

- **device_id**: The GPU device ID to use for the model.

- **data_type**: Specifies the model's data type. Default is `bfloat16`. For 20-series GPUs, which do not support `bfloat16`, use `float16` instead.

- **i2f_mode**: For Turing (20-series) GPUs, controls the GEMM implementation mode. Options are `enabled` and `always`. This option is ignored on other GPU architectures.

**Outputs:**

- **MODEL**: The loaded diffusion model.

.. seealso::

    See example workflows :ref:`nunchaku-flux.1-dev-json`, :ref:`nunchaku-flux.1-schnell-json`.

.. _nunchaku-text-encoder-loader-v2:

Nunchaku Text Encoder Loader V2
-------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/nodes/NunchakuTextEncoderLoaderV2.png
    :alt: NunchakuTextEncoderLoaderV2

A node for loading Nunchaku text encoders, supporting both standard and 4-bit quantized models.

**Inputs:**

- **model_type**: The type of model to load (currently only `flux.1` is supported).

- **text_encoder1**: The first text encoder checkpoint (T5 encoder). You can use standard T5 models or our enhanced `4-bit T5XXL model <https://huggingface.co/mit-han-lab/nunchaku-t5/resolve/main/awq-int4-flux.1-t5xxl.safetensors>`_ for saving GPU memory.

- **text_encoder2**: The second text encoder checkpoint (CLIP encoder). Typically `clip_l.safetensors`.

- **t5_min_length**: Minimum sequence length for the T5 encoder. The default value is 512 for better image quality (compared to 256 in DualCLIPLoader).

**Outputs:**

- **CLIP**: The loaded text encoder model.

.. seealso::

    See example workflows :ref:`nunchaku-flux.1-dev-json`, :ref:`nunchaku-flux.1-dev-qencoder-json`.

.. _nunchaku-flux-pulid-apply-v2:

Nunchaku FLUX PuLID Apply V2
----------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/nodes/NunchakuFluxPuLIDApplyV2.png
    :alt: NunchakuFluxPuLIDApplyV2

A node for applying PuLID identity customization to a Nunchaku FLUX model according to a reference image.

**Inputs:**

- **model**: The Nunchaku FLUX model to modify (must be loaded by Nunchaku FLUX DiT Loader).

- **pulid_pipline**: The PuLID pipeline instance (from Nunchaku PuLID Loader V2).

- **image**: The input image for identity embedding extraction.

- **weight**: How strongly to apply the PuLID effect. Range: -1.0 to 5.0, default: 1.0.

- **start_at**: When to start applying PuLID during the denoising process. Range: 0.0 to 1.0, default: 0.0.

- **end_at**: When to stop applying PuLID during the denoising process. Range: 0.0 to 1.0, default: 1.0.

- **attn_mask** (optional): Attention mask for selective application.

- **options** (optional): Additional options for PuLID processing.

**Outputs:**

- **MODEL**: The modified diffusion model with PuLID applied.

.. seealso::

    See example workflow :ref:`nunchaku-flux.1-dev-pulid-json`.

.. _nunchaku-pulid-loader-v2:

Nunchaku PuLID Loader V2
------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/nodes/NunchakuPuLIDLoaderV2.png
    :alt: NunchakuPuLIDLoaderV2

A node for loading the PuLID pipeline required for identity-preserving image generation.

**Inputs:**

- **model**: The base Nunchaku FLUX model to apply PuLID to.

- **pulid_path**: Path to the PuLID model checkpoint. Download from the official repositories.

- **pulid_mode**: The PuLID processing mode. Options include different quality/speed trade-offs.

- **device_id**: The GPU device ID to use for PuLID processing.

**Outputs:**

- **PULID_PIPELINE**: The loaded PuLID pipeline for use with PuLID Apply nodes.

.. seealso::

    See example workflow :ref:`nunchaku-flux.1-dev-pulid-json`.

.. _nunchaku-text-encoder-loader:

Nunchaku Text Encoder Loader (Deprecated)
-----------------------------------------

.. warning::
    This node is deprecated and will be removed in v0.4. Please use :ref:`nunchaku-text-encoder-loader-v2` instead.

A legacy node for loading Nunchaku text encoders with 4-bit T5 support.

**Inputs:**

- **text_encoder1**: The first text encoder checkpoint (T5).
- **text_encoder2**: The second text encoder checkpoint (CLIP).
- **t5_min_length**: Minimum sequence length for T5 embeddings.
- **use_4bit_t5**: Whether to use quantized 4-bit T5 encoder.
- **int4_model**: The INT4 T5 model folder name (when use_4bit_t5 is enabled).

**Outputs:**

- **CLIP**: The loaded text encoder model. 