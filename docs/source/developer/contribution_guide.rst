Contribution Guide
==================

Welcome to **ComfyUI-nunchaku**! We appreciate your interest in contributing.
This guide outlines how to set up your environment, run tests, and submit a Pull Request (PR).
Whether you're fixing a minor bug or implementing a major feature, we encourage you to
follow these steps for a smooth and efficient contribution process.

🚀 Setting Up Test Environment
------------------------------

1. Fork and Clone the Repository

   New contributors should fork the repository to their GitHub account and clone locally:

   .. code-block:: shell

      git clone https://github.com/<your_username>/ComfyUI-nunchaku.git

2. Install Dependencies

   Install the ``uv`` package manager and project dependencies:

   .. code-block:: shell

      cd ComfyUI-nunchaku
      pip install uv
      uv venv # Create a new virtual environment if you don't already have one

   Choose an installation method based on your development needs:

**Option A: Fresh Installation with Published Wheels**

.. code-block:: shell

   TORCH_VERSION=$(uv pip freeze | sed -n 's/^torch==\([0-9]\+\)\.\([0-9]\+\).*/torch\1\2/p')
   uv pip install --torch-backend=auto -e ".[${TORCH_VERSION},dev]"

**Option B: Build from Source (For Development)**

See :ref:`Build from Source <nunchaku:build-from-source>` for building ``nunchaku`` from source. Then install the environment:

.. code-block:: shell

   uv pip install --torch-backend=auto -e ".[dev]"

**3. Create Test Workspace**

Set up an isolated testing workspace outside the repository:

.. code-block:: shell

   cd ..
   mkdir -p test-workspace && cd test-workspace
   ln -s ../ComfyUI-nunchaku/tests tests
   ln -s ../ComfyUI-nunchaku/test_data test_data
   python ../ComfyUI-nunchaku/scripts/setup_custom_nodes.py

This creates a clean environment with symlinks to test files and installs required ComfyUI custom nodes.

🧹 Code Formatting with Pre-Commit
----------------------------------

We enforce code style using `pre-commit <https://pre-commit.com/>`__ hooks. Please install and run them before submitting changes:

.. code-block:: shell

   uv pip install pre-commit
   pre-commit install
   pre-commit run --all-files

- Run ``pre-commit run --all-files`` until all checks pass.
- Ensure your code passes all checks before opening a pull request.
- Do **not** commit directly to the ``main`` branch. Always use a feature branch (e.g., ``feat/my-feature``) and open a PR from that branch.

🧪 Running Unit Tests & Integrating with CI
-------------------------------------------

We use ``pytest`` for unit testing. When adding features or bug fixes, include new test cases in the ``tests`` directory. **Do not modify existing tests.**

.. _running-tests:

Running Tests
~~~~~~~~~~~~~

.. code-block:: shell

   cd test-workspace
   HF_TOKEN=$YOUR_HF_TOKEN pytest -v tests/ -x -vv

To run only your newly added test, use the ``-k`` flag with your workflow folder name:

.. code-block:: shell

   HF_TOKEN=$YOUR_HF_TOKEN pytest -v tests/ -x -vv -k "nunchaku-flux.1-schnell"

.. note::

   ``$YOUR_HF_TOKEN`` is your Hugging Face access token, required for downloading models and datasets. Create one at https://huggingface.co/settings/tokens. If you have already logged in with ``hf auth login``, you may omit this variable.

Writing Tests
~~~~~~~~~~~~~

When contributing new features or bug fixes, you must register a new test in the ``tests/workflows`` directory. **Do not alter existing tests.**

To add a test case:

1. Create a Workflow Folder

   Create a new folder in ``tests/workflows/`` with a descriptive name (e.g., ``nunchaku-flux.1-schnell``). This folder must contain four JSON files:

   - ``ref.json``: The reference workflow using BF16/FP8 models (for benchmarking)
   - ``workflow.json``: The corresponding Nunchaku version of the workflow
   - ``api.json``: API version of ``workflow.json`` (exported via ComfyUI's ``Export (API)`` option)
   - ``test_cases.json``: Test configurations with different parameters

   .. note::

      Both ``ref.json`` and ``workflow.json`` are for backup purposes, making it easier for future maintenance, development, testing, and debugging.

2. Create the API Workflow

   In ComfyUI, after designing your workflow, export it using ``Export (API)`` and save it as ``api.json`` (see example below).

   .. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/export_api.png
      :alt: ComfyUI Export API Example
      :align: center

3. Configure Test Cases

   Create ``test_cases.json`` to define test parameters. You can override variables in ``api.json`` using the ``inputs`` field. Here's an example:

   .. code-block:: json

      [
        {
          "ref_image_url": "https://github.com/user/repo/issues/123#issuecomment-456789",
          "expected_clip_iqa": {
            "int4-bf16": 0.98,
            "fp4-bf16": 0.99
          },
          "expected_lpips": {
            "int4-bf16": 0.23,
            "fp4-bf16": 0.22
          },
          "expected_psnr": {
            "int4-bf16": 19,
            "fp4-bf16": 19
          },
          "inputs": {
            "30,inputs,model_path": "svdq-{precision}_r32-flux.1-schnell.safetensors",
            "25,inputs,noise_seed": 778459239
          }
        }
      ]

   Each test case should include:

   - ``ref_image_url``: URL to the reference image generated by your BF16/FP8 workflow with the same parameters (remember to fix the seed). Upload the image to a GitHub PR comment to get a public URL.
   - ``expected_clip_iqa``, ``expected_lpips``, ``expected_psnr``: Image quality metrics. These keys use the format ``{precision}-{torch_dtype}``:

     - ``int4``/``fp4``: Nunchaku model precision
     - ``bf16``/``fp16``: Activation torch dtype (fp16 is typically used on RTX 20-series GPUs; others use bf16)

   - ``inputs``: Override parameters in ``api.json`` for testing different configurations

   **How to determine the expected values:**

   Run your test locally first (see :ref:`running-tests`). Use the local results as reference values. If you can only test one precision type (int4 or fp4), you can use the same reference values for both.

4. Add Additional Test Data (if needed)

   If your test requires additional input images or models:

   - Upload input images to a GitHub PR comment to get a public URL
   - Register the URLs in `test_data/inputs.yaml <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/test_data/inputs.yaml>`__
   - If new models are required, update `scripts/download_models.py <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/scripts/download_models.py>`__ and `test_data/models.yaml <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/test_data/models.yaml>`__

5. Register Additional Custom Nodes (if needed)

   If your test requires additional ComfyUI custom nodes, register them in `test_data/custom_nodes.yaml <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/test_data/custom_nodes.yaml>`__:

   .. code-block:: yaml

      - name: ComfyUI-CustomNode
        url: https://github.com/username/ComfyUI-CustomNode
        branch: abcdefg # commit hash or branch name

   The ``scripts/setup_custom_nodes.py`` script (which runs automatically during test workspace setup in Step 3) processes this configuration to:

   - Clone custom node repositories into the test workspace
   - Install Python dependencies from each node's ``requirements.txt`` (if present)
   - Set up the ComfyUI environment with all required nodes

   **Dependency Management:**

   By default, dependencies are installed from the custom node's ``requirements.txt``. To override this, create a file at ``test_data/dependencies/{node_name}.txt`` with your custom requirements. This is useful when the default requirements conflict with the test environment or need version pinning.
