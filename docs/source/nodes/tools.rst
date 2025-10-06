Tool Nodes
==========

.. _nunchaku-wheel-installer:

Nunchaku Wheel Installer
------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/NunchakuWheelInstaller-v1.0.1.png
    :alt: NunchakuWheelInstaller

This node installs or uninstalls the Nunchaku Python package directly from the ComfyUI interface.
It supports both official and development versions, and can update its version list from online sources.

The node operates in offline mode using a local cache file (``nunchaku_versions.json``).
It features separate dropdowns for official and development versions.

For official versions, installation attempts sources in order: ModelScope, Hugging Face, then GitHub Releases.
For development versions (Linux only), only GitHub Releases are used.

**Important:** After installation or uninstallation, you must **completely restart ComfyUI** for changes to take effect.

**Inputs:**

- **version**: Official Nunchaku version to install. Use "update node" mode to get the latest list. If dev_version is also selected, it will take priority.
- **dev_version**: Development Nunchaku version to install. Use "update node" mode to get the latest list. This option has priority over the official version.
- **mode**: "install" to install Nunchaku, "uninstall" to remove it, or "update node" to refresh the version list from the CDN.

**Outputs:**

- **status**: Displays the result of the installation or uninstallation process.

.. seealso::

    API reference: :class:`~comfyui_nunchaku.nodes.tools.installers.NunchakuWheelInstaller`.

    Example workflow: :ref:`install-wheel-json`.

.. _nunchaku-model-merger:

Nunchaku Model Merger
---------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/nodes/NunchakuModelMerger.png
    :alt: NunchakuModelMerger

A utility node that merges a legacy SVDQuant FLUX.1 model folder into a single ``.safetensors`` file.

**Inputs:**

- **model_folder**: Select the model folder to merge (from your ``models/diffusion_models`` directory).
- **save_name**: Set the output filename for the merged model. The merged model will be saved in the same directory as the original folder.

**Outputs:**

- **status**: The status of the merge.

.. seealso::

    API reference: :class:`~comfyui_nunchaku.nodes.tools.merge_safetensors.NunchakuModelMerger`.

    Example workflow: :ref:`merge-safetensors-json`.
