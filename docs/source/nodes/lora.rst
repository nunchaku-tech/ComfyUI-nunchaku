LoRA Nodes
==========

.. _nunchaku-flux-lora-loader:

Nunchaku FLUX LoRA Loader
-------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/nodes/NunchakuFluxLoraLoader.png
    :alt: NunchakuFluxLoraLoader

A node for loading and applying LoRA weights to Nunchaku FLUX models within ComfyUI.

**Inputs:**

- **model**: The diffusion model the LoRA will be applied to. Make sure the model is loaded by :ref:`nunchaku-flux-dit-loader`.
- **lora_name**: The file name of the LoRA checkpoint. Place your LoRA files in the ``models/loras`` directory, and they will appear as selectable options.
- **lora_strength**: How strongly to modify the diffusion model. This value can be negative. Range: -100.0 to 100.0, default: 1.0.

**Outputs:**

- **model**: The modified diffusion model with LoRA applied.

.. tip::

    Multiple LoRA modules can be chained together by connecting the output of one LoRA node to the input of another.

.. seealso::

    API reference: :class:`~comfyui_nunchaku.nodes.lora.flux.NunchakuFluxLoraLoader`.
