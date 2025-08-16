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

   Follow the steps in :doc:`Installation <../get_started/installation>` to set up your environment.
   For testing, also install the following plugins:

   - https://github.com/Fannovel16/comfyui_controlnet_aux
   - https://github.com/CY-CHENYUE/ComfyUI-InpaintEasy

   If you use `comfy-cli <github_comfy-cli_>`_, you can install them by running the following command:

   .. code-block:: shell

      comfy node install comfyui_controlnet_aux
      comfy node install comfyui-inpainteasy

3. Download Models and Test Data

   .. code-block:: shell

      HF_TOKEN=$YOUR_HF_TOKEN python custom_nodes/ComfyUI-nunchaku/scripts/download_models.py
      HF_TOKEN=$YOUR_HF_TOKEN python custom_nodes/ComfyUI-nunchaku/scripts/download_test_data.py

🧹 Code Formatting with Pre-Commit
----------------------------------

We enforce code style using `pre-commit <https://pre-commit.com/>`__ hooks. Please install and run them before submitting changes:

.. code-block:: shell

   pip install pre-commit
   pre-commit install
   pre-commit run --all-files

- Run ``pre-commit run --all-files`` until all checks pass.
- Ensure your code passes all checks before opening a pull request.
- Do **not** commit directly to the ``main`` branch. Always use a feature branch (e.g., ``feat/my-feature``) and open a PR from that branch.

🧪 Running Unit Tests & Integrating with CI
-------------------------------------------

We use ``pytest`` for unit testing. When adding features or bug fixes, include new test cases in the ``tests`` directory. **Do not modify existing tests.**

Running Tests
~~~~~~~~~~~~~

.. code-block:: shell

   cd ComfyUI  # Ensure `ComfyUI-nunchaku` is in the `custom_nodes` directory
   ln -s custom_nodes/ComfyUI-nunchaku/tests nunchaku_tests
   HF_TOKEN=$YOUR_HF_TOKEN pytest -v nunchaku_tests/

.. note::

   ``$YOUR_HF_TOKEN`` is your Hugging Face access token, required for downloading models and datasets. Create one at https://huggingface.co/settings/tokens. If you have already logged in with ``hf auth login``, you may omit this variable.

Writing Tests
~~~~~~~~~~~~~

When contributing new features or bug fixes, add corresponding tests in the ``tests`` directory. **Do not alter existing tests.**

To add a test case:

1. **Install the ComfyUI-to-Python Extension**

   We use https://github.com/pydn/ComfyUI-to-Python-Extension to convert ComfyUI workflows to Python scripts for testing.

   .. code-block:: shell

      pip install black
      cd custom_nodes
      git clone https://github.com/pydn/ComfyUI-to-Python-Extension

2. **Convert the Workflow to a Python Script**

   In ComfyUI, use ``workflow -> Save as Script`` to export your workflow.

   .. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/save_script.png
      :alt: Save as Script
      :align: center

   Place the generated script in `tests/scripts <https://github.com/nunchaku-tech/ComfyUI-nunchaku/tree/main/tests/scripts>`__ and the corresponding workflow in `tests/workflows <https://github.com/nunchaku-tech/ComfyUI-nunchaku/tree/main/tests/workflows>`__.

3. **Modify the Script for Testing**

   Make minor adjustments to ensure compatibility with the test environment. For example:

   .. literalinclude:: ../../../tests/scripts/nunchaku-flux1-schnell.py
      :language: python
      :caption: Test script for :ref:`nunchaku-flux.1-schnell-json`
      :linenos:
      :emphasize-lines: 7, 120, 149, 153, 195-200, 204

   Key changes from the generated script:

   - Pass precision to the main function (see lines 7, 120, 153, 204; use :func:`~nunchaku:nunchaku.utils.get_precision`).
   - Set a fixed random seed (line 149).
   - Save the output image path to ``image_path.txt`` (lines 195-200).

4. **Run the Script**

   .. code-block:: shell

      python nunchaku_tests/nunchaku-flux1-schnell.py

5. **Register the Test Script**

   Register your new test script in `tests/test_workflows.py <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/tests/test_workflows.py>`__ by adding an entry to the parameterized test list. For example:

   .. code-block:: python

      ("nunchaku-flux1-schnell.py", 0.9, 0.29, 19.3),

   Each tuple should contain:

   - ``script_name``: The filename of your test script.
   - ``expected_clip_iqa``: The minimum acceptable CLIP IQA score for your generated image.
   - ``expected_lpips``: The maximum acceptable LPIPS score (measuring perceptual similarity to the reference image).
   - ``expected_psnr``: The minimum acceptable PSNR score (measuring pixel-level similarity to the reference image).

   **How to determine the expected values:**

   - **CLIP IQA**: Run your script several times (e.g., 5 runs) and use the lowest CLIP IQA score as the expected value.
   - **LPIPS and PSNR**: These metrics compare your generated image to a reference image.
     You can run your script multiple times (e.g., 5 runs), use the first generated image as the reference,
     and then set the highest observed LPIPS and PSNR values as the expected thresholds.

   After determining your reference image, upload it to our `Hugging Face dataset <https://huggingface.co/datasets/nunchaku-tech/test-data/tree/main/ComfyUI-nunchaku/ref_images>`__.
   Be sure to upload the image to both the ``int4`` and ``fp4`` directories to support all test configurations.

6. **Add Test Data Links**

   Add download links for test data in `test_data/images.yaml <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/test_data/images.yaml>`__.

   If new models are required, update `scripts/download_models.py <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/scripts/download_models.py>`__ and `test_data/models.yaml <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/test_data/models.yaml>`__ accordingly.
