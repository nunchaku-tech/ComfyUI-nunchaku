Usage
=====

1. **Set Up ComfyUI and Nunchaku**

   - Nunchaku workflows can be found at :doc:`example_workflows <../../example_workflows>`. To use them, copy the files to ``user/default/workflows`` in the ComfyUI root directory:

     .. code-block:: shell

        cd ComfyUI

        # Create the example_workflows directory if it doesn't exist
        mkdir -p user/default/example_workflows

        # Copy workflow configurations
        cp custom_nodes/nunchaku_nodes/example_workflows/* user/default/example_workflows/

   - Install any missing nodes (e.g., ``comfyui-inpainteasy``) by following `this tutorial <https://github.com/ltdrdata/ComfyUI-Manager?tab=readme-ov-file#support-of-missing-nodes-installation>`_.

2. **Download Required Models**

   Follow `this tutorial <https://comfyanonymous.github.io/ComfyUI_examples/flux/>`_ to download the necessary models into the appropriate directories. Alternatively, use the following commands:

   .. code-block:: shell

      huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir models/text_encoders
      huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir models/text_encoders
      huggingface-cli download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir models/vae

3. **Run ComfyUI**

   To start ComfyUI, navigate to its root directory and run ``python main.py``. If you are using ``comfy-cli``, simply run ``comfy launch``.

4. **Select the Nunchaku Workflow**

   Choose one of the Nunchaku workflows (workflows that start with ``nunchaku-``) to get started. For the ``flux.1-fill`` workflow, you can use the built-in **MaskEditor** tool to apply a mask over an image.

5. All the 4-bit models are available at our `HuggingFace <https://huggingface.co/collections/mit-han-lab/svdquant-67493c2c2e62a1fc6e93f45c>`_ or `ModelScope <https://modelscope.cn/collections/svdquant-468e8f780c2641>`_ collection. Except :doc:`svdq-flux.1-t5 <https://huggingface.co/mit-han-lab/svdq-flux.1-t5>`, please download the **entire model folder** to ``models/diffusion_models``.
