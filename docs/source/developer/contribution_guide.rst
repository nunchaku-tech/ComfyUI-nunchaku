Contribution Guide
==================

Welcome to **ComfyUI-nunchaku**! We appreciate your interest in contributing.
This guide outlines how to set up your environment, run tests, and submit a Pull Request (PR).
Whether you're fixing a minor bug or implementing a major feature, we encourage you to
follow these steps for a smooth and efficient contribution process.

🚀 Setting Up & Building from Source
------------------------------------

1. Fork and Clone the Repository

   .. note::

      As a new contributor, you won't have write access to the repository.
      Please fork the repository to your own GitHub account, then clone your fork locally:

   .. code-block:: shell

      git clone https://github.com/<your_username>/ComfyUI-nunchaku.git

2. Install Dependencies & Build

   To install dependencies and build the project, follow the instructions in :doc:`Installation <../get_started/installation>`.
   For the test environment, you also need to install the following plugins:

   - https://github.com/Fannovel16/comfyui_controlnet_aux
   - https://github.com/CY-CHENYUE/ComfyUI-InpaintEasy

   If you use `comfy-cli <github_comfy-cli_>`_, you can install them by running the following command:
   
   .. code-block:: shell
   
      comfy node install comfyui_controlnet_aux
      comfy node install comfyui-inpainteasy

3. Download models and data for testing
   
   .. code-block:: shell
   
      HF_TOKEN=$YOUR_HF_TOKEN python custom_nodes/ComfyUI-nunchaku/scripts/download_models.py
      HF_TOKEN=$YOUR_HF_TOKEN python custom_nodes/ComfyUI-nunchaku/scripts/download_test_data.py

🧹 Code Formatting with Pre-Commit
----------------------------------

We use `pre-commit <https://pre-commit.com/>`__ hooks to ensure code style consistency. Please install and run it before submitting your changes:

.. code-block:: shell

   pip install pre-commit
   pre-commit install
   pre-commit run --all-files

- ``pre-commit run --all-files`` manually triggers all checks and automatically fixes issues where possible. If it fails initially, re-run until all checks pass.

- ✅ **Ensure your code passes all checks before opening a PR.**

- 🚫 **Do not commit directly to the** ``main`` **branch.**
- Always create a feature branch (e.g., ``feat/my-new-feature``),
- commit your changes there, and open a PR from that branch.

🧪 Running Unit Tests & Integrating with CI
-------------------------------------------

Nunchaku uses ``pytest`` for unit testing. If you're adding a new feature,
please include corresponding test cases in the ``tests`` directory.
**Please avoid modifying existing tests.**

Running the Tests
~~~~~~~~~~~~~~~~~

.. code-block:: shell

    cd ComfyUI # Go to the ComfyUI directory, make sure you put the `ComfyUI-nunchaku` directory in the `custom_nodes` directory
    ln -s custom_nodes/ComfyUI-nunchaku/tests nunchaku_tests
    HF_TOKEN=$YOUR_HF_TOKEN pytest -v nunchaku_tests/

.. note::

   ``$YOUR_HF_TOKEN`` refers to your Hugging Face access token, required to download models and datasets.
   You can create one at https://huggingface.co/settings/tokens.
   If you've already logged in using ``huggingface-cli login``,
   you can skip setting this environment variable.

Writing Tests
~~~~~~~~~~~~~

When adding a new feature or fixing a bug, please include corresponding test cases in the ``tests`` directory. **Please avoid modifying existing tests.**

Here we provide a guidance on how to add a test case.

Step 1: Install https://github.com/pydn/ComfyUI-to-Python-Extension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use https://github.com/pydn/ComfyUI-to-Python-Extension
to convert the ComfyUI workflow to Python code and then test the workflow.
To install it, run the following command:

.. code-block:: shell

   pip install black
   cd custom_nodes
   git clone https://github.com/pydn/ComfyUI-to-Python-Extension

Step 2: Convert the ComfyUI workflow to Python Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After installing the extension,
you can convert the workflow in the ComfyUI by clicking
``workflow -> Save as Script`` in top left corner like the following figure:

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/save_script.png
   :alt: Save as Script
   :align: center

Put the converted Python Script in the `tests/scripts <https://github.com/nunchaku-tech/ComfyUI-nunchaku/tree/main/tests/scripts>`__ directory
and the corresponding workflow in the `tests/workflows <https://github.com/nunchaku-tech/ComfyUI-nunchaku/tree/main/tests/workflows>`__ directory.
Typically, the workflow name is the same as the Python Script name.

Step 3: Modify the Script
^^^^^^^^^^^^^^^^^^^^^^^^^

You need make some minor modifications to the script to make it work with the test environment. Here we show an example of how to modify the script.

.. literalinclude:: ../../../tests/scripts/nunchaku-flux1-schnell.py
    :language: python
    :caption: Test script for :ref:`nunchaku-flux.1-schnell-json`
    :linenos:
    :emphasize-lines: 7, 120, 149, 153, 195-200, 204

The major changes are:

- Pass the precision to the main function (line 7, 120, 153, 204). The precision can be got from :func:`~nunchaku:nunchaku.utils.get_precision`.
- Fix the random seed (line 149).
- Get the name of the output image to save it in the ``image_path.txt`` file (line 195-200).

Step 4: Run Your Script
^^^^^^^^^^^^^^^^^^^^^^^

You can run your script by running the following command. Suppose the script is named ``nunchaku-flux1-schnell.py``

.. code-block:: shell

   python nunchaku_tests/nunchaku-flux1-schnell.py

Step 5: Register your Test script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can register your test script in the `tests/test_workflows.py <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/c3d3450cdcf392ed1c7f7f1e5c213c71eea6579b/tests/test_workflows.py#L37>`__ file.

Simply add a new line to the ``test_workflows.py`` file.

.. code-block:: python

   ("nunchaku-flux1-schnell.py", 0.9, 0.29, 19.3),

The parameters are:

- ``script_name``: The name of the script.
- ``expected_clip_iqa``: The expected CLIP IQA score.
- ``expected_lpips``: The expected LPIPS score.
- ``expected_psnr``: The expected PSNR score.

CLIP IQA is the image quality of your generated image. You can run your script multiple times (e.g., 5 times) and set the lowest CLIP IQA score as the expected CLIP IQA score.

LPIPS and PSNR are the similarity scores between your generated image and the reference image.
You can also run the scripts multiple times (e.g., 5) and set the first image as the reference images.
The use the highest LPIPS and PSNR score as the expected scores.

You then need to upload the reference images to our `Hugging Face dataset <https://huggingface.co/datasets/nunchaku-tech/test-data/tree/main/ComfyUI-nunchaku/ref_images>`__. Upload the images to both the ``int4`` and ``fp4`` directories.

Step 5: Add the Links to Download the Test Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You need to add the links to download the test data in the `test_data/images.yaml <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/test_data/images.yaml>`__ file.

Also, you may need update `scripts/download_models.py <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/scripts/download_models.py>`__ and `test_data/models.yaml <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/test_data/models.yaml>`__ to add the links to download the models if you need to download the models.