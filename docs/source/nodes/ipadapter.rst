IP-Adapter Nodes
================

.. _nunchaku-flux-ip-adapter-loader:

Nunchaku IP-Adapter Loader
--------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/nodes/NunchakuIPAdapterLoader.png

A node for loading IP-Adapter weights to Nunchaku FLUX models within ComfyUI.

**Inputs:**

- **model**: The Nunchaku FLUX model to inject IP-Adapter to. Make sure the model is loaded by :ref:`nunchaku-flux-dit-loader`.

**Outputs:**

- **model**: IP-Adapter injected Nunchaku FLUX model.
- **ipadapter_pipeline**: The loaded IP-Adapter pipeline, ready for use with IP-Adapter Apply nodes.

.. warning::
   This node will automatically download the IP-Adapter and associated CLIP models from Hugging Face.
   Custom model paths are not supported for now.

.. seealso::
    API reference: :class:`~comfyui_nunchaku.nodes.models.ipadapter.NunchakuIPAdapterLoader`.

    Example workflow: :ref:`nunchaku-flux.1-ip-adapter-json`.

.. _nunchaku-flux-ip-adapter-apply:

Nunchaku IP-Adapter Apply
-------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/nodes/NunchakuFluxIPAdapterApply.png
    :alt: NunchakuFluxIPAdapterApply

A node for applying IP-Adapter to a Nunchaku model using a given image and weight.

**Inputs:**

- **model**: The Nunchaku FLUX model to apply IP-Adapter to. Make sure the model is loaded by :ref:`nunchaku-flux-ip-adapter-loader`.
- **ipadapter_pipeline**: The IP-Adapter pipeline to apply.
- **image**: The image to apply IP-Adapter to.
- **weight**: The weight of the IP-Adapter.

**Outputs:**

- **model**: The Nunchaku FLUX model with IP-Adapter applied.

.. seealso::
    API reference: :class:`~comfyui_nunchaku.nodes.models.ipadapter.NunchakuFluxIPAdapterApply`.

    Example workflow: :ref:`nunchaku-flux.1-ip-adapter-json`.
