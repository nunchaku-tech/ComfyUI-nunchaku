Preprocessor Nodes
==================

.. _flux-depth-preprocessor:

FLUX Depth Preprocessor (Deprecated)
------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/nodes/FluxDepthPreprocessor.png
    :alt: FluxDepthPreprocessor

.. warning::
    This node is deprecated and will be removed in October 2025. Please use the **Depth Anything** node in `comfyui_controlnet_aux <https://github.com/Fannovel16/comfyui_controlnet_aux>`_ instead.

A legacy node for depth preprocessing using `Depth Anything <https://huggingface.co/LiheYoung/depth-anything-large-hf>`_.
This node applies a depth estimation model to an input image to produce a corresponding depth map.

**Inputs:**

- **model_path**: Path to the depth estimation model checkpoint. You can manually download the model repository from `Hugging Face <https://huggingface.co/LiheYoung/depth-anything-large-hf>`__ and place it under the `models/checkpoints` directory.

- **image**: The input image to process for depth estimation.

**Outputs:**

- **IMAGE**: The generated depth map as a grayscale image.

.. tip::

    You can use the following command to download the model from Hugging Face:

    .. code-block:: bash

        huggingface-cli download LiheYoung/depth-anything-large-hf --local-dir models/checkpoints/depth-anything-large-hf

.. seealso::

    API reference: :class:`~comfyui_nunchaku.nodes.preprocessors.depth.FluxDepthPreprocessor`.
