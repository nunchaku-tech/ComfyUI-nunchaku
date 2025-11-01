Tool Workflows
==============

.. _install-wheel-json:

install_wheel.json
------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/install_wheel-v1.0.1.png
    :alt: install_wheel.json
    :target: https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/install_wheel.json

Example workflow demonstrating the Nunchaku Wheel Installer node for package management.

Overview
~~~~~~~~

This workflow provides a single-node setup for installing, uninstalling, or updating Nunchaku packages.
It demonstrates three main operations:

- **install**: Download and install a specific Nunchaku version from multiple sources.
- **uninstall**: Remove the currently installed Nunchaku package.
- **update node**: Refresh the version list from the CDN.

Workflow Components
~~~~~~~~~~~~~~~~~~~

The workflow consists of a single :ref:`nunchaku-wheel-installer` node with configurable inputs:

1. **version**: Select from available official releases or "none".
2. **dev_version**: Select from development versions or "none".
3. **mode**: Choose the operation type (install/uninstall/update node).

Usage Instructions
~~~~~~~~~~~~~~~~~~

1. **First-time setup**: Set ``mode`` to "update node" and queue the prompt to fetch the latest version list.
2. **Installing a version**: Select your desired version from the ``version`` dropdown, set ``mode`` to "install", and queue the prompt.
3. **Development versions**: Use ``dev_version`` dropdown for development builds.
4. **After installation**: Completely restart ComfyUI for changes to take effect.

The ``status`` output shows installation logs and instructions.

**Links:**

- Workflow file: :download:`install_wheel.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/install_wheel.json>`

.. seealso::
    Node documentation: :ref:`nunchaku-wheel-installer`

.. _merge-safetensors-json:

merge_safetensors.json
----------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/merge_safetensors.png
    :alt: merge_safetensors.json
    :target: https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/merge_safetensors.json

Workflow for merging a legacy SVDQuant model folder into a single ``.safetensors`` file.

**Links:**

- Workflow: :download:`merge_safetensors.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/merge_safetensors.json>`

.. seealso::
    See node :ref:`nunchaku-model-merger`.
