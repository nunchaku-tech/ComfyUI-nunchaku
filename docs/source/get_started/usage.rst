Usage
=====

Step 1: Set Up ComfyUI Workflows
--------------------------------

- Nunchaku workflows can be found at `example_workflows <comfyui_nunchaku_example_workflows_>`_. To use them, copy the files to ``user/default/workflows`` in the ComfyUI root directory:

  .. code-block:: shell

     cd ComfyUI

     # Create the example_workflows directory if it doesn't exist
     mkdir -p user/default/example_workflows

     # Copy workflows
     cp custom_nodes/nunchaku_nodes/example_workflows/* user/default/example_workflows/

- Install any missing nodes (e.g., ``comfyui-inpainteasy``) by following `this tutorial <https://github.com/ltdrdata/ComfyUI-Manager?tab=readme-ov-file#support-of-missing-nodes-installation>`__.

Step 2: Download Models
-----------------------

First, follow `this tutorial <https://comfyanonymous.github.io/ComfyUI_examples/flux/>`__
to download the necessary FLUX models into the appropriate directories. Alternatively, use the following commands:

.. code-block:: shell

   huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir models/text_encoders
   huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir models/text_encoders
   huggingface-cli download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir models/vae

Then, download the nunchaku models from our `HuggingFace <nunchaku_huggingface_>`_ or `ModelScope <nunchaku_modelscope_>`_ collection.

.. note::

   - For **Blackwell GPUs** (such as the RTX 50-series), please use our **FP4** models for hardware compatibility.
   - For all other GPUs, please use our **INT4** models.

Step 3: Run ComfyUI
-------------------

To start ComfyUI, navigate to its root directory and run ``python main.py``.
If you are using ``comfy-cli``, simply run ``comfy launch``.

Step 4: Select the Nunchaku Workflow
------------------------------------

Choose one of the Nunchaku workflows (workflows that start with ``nunchaku-``) to get started.
