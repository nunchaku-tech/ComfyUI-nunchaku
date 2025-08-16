Preprocessor Nodes
==================

.. _flux-depth-preprocessor:

FLUX Depth Preprocessor (Deprecated)
------------------------------------

.. warning::
    This node is deprecated and will be removed in October 2025.
    Please use the **Depth Anything** node in `comfyui_controlnet_aux <github_comfyui_controlnet_aux_>`_ instead.

A legacy node for depth preprocessing using `Depth Anything <hf_depth_anything_>`_.
This node applies a depth estimation model to an input image to produce a corresponding depth map.

**Inputs:**

- **model_path**: Path to the depth estimation model checkpoint. You can manually download the model repository from `Hugging Face <hf_depth_anything_>`_ and place it under the `models/checkpoints` directory.

- **image**: The input image to process for depth estimation.

**Outputs:**

- **IMAGE**: The generated depth map as a grayscale image.

.. tip::

    You can use the following command to download the model from Hugging Face:

    .. code-block:: bash

        hf download LiheYoung/depth-anything-large-hf --local-dir models/checkpoints/depth-anything-large-hf

.. seealso::

    API reference: :class:`~comfyui_nunchaku.nodes.preprocessors.depth.FluxDepthPreprocessor`.
