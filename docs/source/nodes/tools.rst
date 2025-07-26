Tool Nodes
==========

.. _nunchaku-wheel-installer:

Nunchaku Wheel Installer
------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/nodes/NunchakuWheelInstaller.png
    :alt: NunchakuWheelInstaller

A utility node for automatically installing the correct version of the `nunchaku <github_nunchaku_>`_ wheel in ComfyUI.
After installation, please **restart ComfyUI** to apply the changes.

**Inputs:**

- **source**: Select the wheel source. Options include:

  - `GitHub Releases <github_nunchaku_releases_>`_
  - `HuggingFace <hf_nunchaku_wheels_>`_
  - `ModelScope <ms_nunchaku_wheels_>`_

- **version**: Choose the compatible `nunchaku <github_nunchaku_>`_ version to install.

**Outputs:**

- **status**: The status of the installation.

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
