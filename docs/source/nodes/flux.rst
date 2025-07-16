Model Nodes
===========

.. _nunchaku-flux-dit-loader:

Nunchaku FLUX DiT Loader
------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/nodes/NunchakuFluxDiTLoader.png
    :alt: NunchakuFluxDiTLoader

A node for loading Nunchaku FLUX models. It manages model loading, device selection, attention implementation, CPU offload, and caching for efficient inference.

**Inputs:**

- **model_path**: The path to the Nunchaku FLUX model folder (legacy format) or ``.safetensors`` file. You can download the model from `HuggingFace <nunchaku_huggingface_>`_ or `ModelScope <nunchaku_modelscope_>`_.

- **cache_threshold**: Adjusts the first-block caching tolerance like ``residual_diff_threshold`` in WaveSpeed. Increasing the value enhances speed at the cost of quality. A typical setting is 0.12. Setting it to 0 disables the effect. See :ref:`nunchaku:usage-fbcache` for more details.

- **attention**: Attention implementation. Options include ``flash-attention2`` and ``nunchaku-fp16``. The ``nunchaku-fp16`` uses FP16 attention, offering ~1.2× speedup. Note that 20-series GPUs can only use ``nunchaku-fp16``.

- **cpu_offload**: Whether to enable CPU offload for the transformer model. Options include:

  - ``auto``: Will enable it if the GPU memory is less than 14GiB
  - ``enable``: Force enable CPU offload
  - ``disable``: Disable CPU offload

- **device_id**: The GPU device ID to use for the model.

- **data_type**: Specifies the model's data type. Default is ``bfloat16``. For 20-series GPUs, which do not support ``bfloat16``, use ``float16`` instead.

- **i2f_mode**: For Turing (20-series) GPUs, controls the GEMM implementation mode. Options are `enabled` and `always`. This option is ignored on other GPU architectures.

**Outputs:**

- **model**: The loaded diffusion model.

.. seealso::

    See API reference: :class:`~comfyui_nunchaku.nodes.models.flux.NunchakuFluxDiTLoader`.

.. _nunchaku-text-encoder-loader-v2:

Nunchaku Text Encoder Loader V2
-------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/nodes/NunchakuTextEncoderLoaderV2.png
    :alt: NunchakuTextEncoderLoaderV2

A node for loading Nunchaku text encoders.

.. tip::
    You can also use this node to load the 16-bit or FP8 text encoders.

.. note::
    When loading our 4-bit T5, a 16-bit T5 is first initialized on a meta device,
    then replaced by the Nunchaku T5.

.. warning::
    Our 4-bit T5 currently requires a CUDA device.
    If not on CUDA, the model will be moved automatically, which may cause out-of-memory errors.
    Turing GPUs (20-series) are not supported for now.

**Inputs:**

- **model_type**: The type of model to load (currently only `flux.1` is supported).

- **text_encoder1**: The first text encoder checkpoint.

- **text_encoder2**: The second text encoder checkpoint.

- **t5_min_length**: Minimum sequence length for the T5 encoder. The default value is 512 to better align our quantization settings.

**Outputs:**

- **clip**: The loaded text encoder model.

.. seealso::

    API reference: :class:`~comfyui_nunchaku.nodes.models.text_encoder.NunchakuTextEncoderLoaderV2`.

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
