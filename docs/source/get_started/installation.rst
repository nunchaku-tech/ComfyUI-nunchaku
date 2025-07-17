Installation
============

We provide tutorial videos to help you install and use Nunchaku on Windows,
available in both `English <nunchaku_windows_tutorial_en_>`_ and `Chinese <nunchaku_windows_tutorial_zh_>`_.
If you run into issues, these resources are a good place to start.

Step 1: Install the ComfyUI-nunchaku Plugin
-------------------------------------------

You can use the following ways to install the `ComfyUI-nunchaku <comfyui_nunchaku_repo_>`_ plugin.

Comfy-CLI
~~~~~~~~~

You can easily use `comfy-cli <comfy_cli_repo_>`_ to run ComfyUI with Nunchaku:

.. code-block:: shell

   pip install comfy-cli  # Install ComfyUI CLI
   comfy install          # Install ComfyUI
   comfy node registry-install ComfyUI-nunchaku  # Install Nunchaku

ComfyUI-Manager
~~~~~~~~~~~~~~~

1. Install `ComfyUI <comfyui_repo_>`_ with

   .. code-block:: shell

      git clone https://github.com/comfyanonymous/ComfyUI.git
      cd ComfyUI
      pip install -r requirements.txt

2. Install `ComfyUI-Manager <comfyui_manager_repo_>`_ with the following commands:

   .. code-block:: shell

      cd custom_nodes
      git clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager

3. Launch ComfyUI

   .. code-block:: shell

      cd ..  # Return to the ComfyUI root directory
      python main.py

4. Open the Manager, search ``ComfyUI-nunchaku`` in the Custom Nodes Manager and then install it.

Manual Installation
~~~~~~~~~~~~~~~~~~~

1. Set up `ComfyUI <comfyui_repo_>`_ with the following commands:

   .. code-block:: shell

      git clone https://github.com/comfyanonymous/ComfyUI.git
      cd ComfyUI
      pip install -r requirements.txt

2. Clone this repository into the ``custom_nodes`` directory inside ComfyUI:

   .. code-block:: shell

      cd custom_nodes
      git clone https://github.com/mit-han-lab/ComfyUI-nunchaku nunchaku_nodes

Step 2: Install the Nunchaku Backend
------------------------------------

Starting from **ComfyUI-nunchaku v0.3.2**,
you can easily install or update the `Nunchaku <nunchaku_repo_>`_ wheel using this
`workflow file <comfyui_nunchaku_wheel_installation_workflow_>`_, once all dependencies are installed.

Alternatively, you can follow the manual installation instructions in the :ref:`nunchaku:installation-installation`.
