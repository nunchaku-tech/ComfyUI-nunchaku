Text Encoder Nodes
==================

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
    This node is deprecated and will be removed in December 2025. Please use :ref:`nunchaku-text-encoder-loader-v2` instead.

A legacy node for loading Nunchaku text encoders with 4-bit T5 support.

**Inputs:**

- **text_encoder1**: The first text encoder checkpoint (T5).
- **text_encoder2**: The second text encoder checkpoint (CLIP).
- **t5_min_length**: Minimum sequence length for T5 embeddings.
- **use_4bit_t5**: Whether to use quantized 4-bit T5 encoder.
- **int4_model**: The INT4 T5 model folder name (when use_4bit_t5 is enabled).

**Outputs:**

- **CLIP**: The loaded text encoder model.
