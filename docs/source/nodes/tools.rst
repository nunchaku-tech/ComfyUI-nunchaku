Tool Nodes
==========

.. _nunchaku-wheel-installer:

Nunchaku Wheel Installer
------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/NunchakuWheelInstaller-v1.0.1.png
    :alt: NunchakuWheelInstaller

This node automates the installation of Nunchaku wheels from `GitHub Releases <github_nunchaku_releases_>`_, `Hugging Face <hf_nunchaku_wheels_>`_, or `ModelScope <ms_nunchaku_wheels_>`_.
For official releases, it attempts installation in the following order: ModelScope, then Hugging Face, and finally GitHub Releases.
For development versions, only GitHub Releases are used.

If you select "latest" for either version, the node will fetch the most up-to-date version list from the internet and update `nunchaku_versions.json` before proceeding with installation.

**Important:** After installation, you must **restart ComfyUI** for changes to take effect.

If both ``version`` and ``dev_version`` are specified, the development version takes precedence.

**Inputs:**

- **version**: Choose the official Nunchaku version to install. Select "latest" to refresh and use the newest available version.
- **dev_version**: Choose a development version to install. Select "latest" to refresh and use the newest available development version. If both version and dev_version are set, dev_version will be prioritized.
- **mode**: Select "install" to install Nunchaku or "uninstall" to remove it.

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
