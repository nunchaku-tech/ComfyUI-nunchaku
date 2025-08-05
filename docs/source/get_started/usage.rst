Usage
=====

Step 1: Set Up ComfyUI Workflows
--------------------------------

- Nunchaku workflows can be found at `example_workflows <github_comfyui-nunchaku_example_workflows_>`_. To use them, copy the files to ``user/default/workflows`` in the ComfyUI root directory:

  .. code-block:: shell

     cd ComfyUI

     # Create the example_workflows directory if it doesn't exist
     mkdir -p user/default/example_workflows

     # Copy workflows
     cp custom_nodes/nunchaku_nodes/example_workflows/* user/default/example_workflows/

- Install any missing nodes (e.g., ``comfyui-inpainteasy``) by following `this tutorial <github_comfyui-manager_missing-nodes-installation_>`_.

Step 2: Download Models
-----------------------

First, follow `this tutorial <comfyui_examples_flux_>`_
to download the necessary FLUX models into the appropriate directories.
Alternatively, use the following commands:

.. code-block:: shell

   hf download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir models/text_encoders
   hf download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir models/text_encoders
   hf download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir models/vae

Then, download the nunchaku models from `Hugging Face <hf_nunchaku_>`_ or `ModelScope <ms_nunchaku_>`_.
For example, to download the Nunchaku Flux.1 dev models from Hugging Face, run:

.. code-block:: shell

   hf download nunchaku-tech/nunchaku-flux.1-dev svdq-int4_r32-flux.1-dev.safetensors --local-dir models/unet/

.. note::

   - For **Blackwell GPUs** (such as the RTX 50-series), please use our **FP4** models for hardware compatibility.
   - For all other GPUs, please use our **INT4** models.

Step 3: Run ComfyUI
-------------------

To start ComfyUI, navigate to its root directory and run ``python main.py``.
If you are using `comfy-cli <github_comfy-cli_>`_, simply run ``comfy launch``.

Step 4: Select the Nunchaku Workflow
------------------------------------

Choose one of the Nunchaku workflows to get started. See :doc:`../workflows/toc` for more details.
