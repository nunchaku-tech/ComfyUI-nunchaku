Building the Documentation
==========================

Follow this guide to build the ComfyUI-nunchaku documentation locally using Sphinx.

Step 1: Set Up the Environment
------------------------------

Install the required dependencies for building the documentation.

**Option A: Fresh Installation**

Install ``ComfyUI-nunchaku`` in editable mode with pre-compiled nunchaku wheels:

.. code-block:: shell

    cd ComfyUI-nunchaku
    pip install uv
    TORCH_VERSION=$(uv pip freeze | sed -n 's/^torch==\([0-9]\+\)\.\([0-9]\+\).*/torch\1\2/p')
    uv pip install --torch-backend=auto -e ".[${TORCH_VERSION},dev,docs]"

This command installs the ``uv`` package manager, detects your PyTorch version, and installs all development and documentation dependencies.

**Option B: Existing Installation**

If ``nunchaku`` is already installed, install only the documentation dependencies:

.. code-block:: shell

    cd ComfyUI-nunchaku
    pip install uv
    uv pip install --torch-backend=auto -e ".[dev,docs]"

Step 2: Build the Documentation
-------------------------------

Navigate to the ``docs`` directory and build the HTML documentation:

.. code-block:: shell

    cd docs
    make clean
    make html

This will generate the HTML files in the ``docs/build/html`` directory.

Step 3: Preview the Documentation
---------------------------------

To view the generated documentation locally, start a simple HTTP server:

.. code-block:: shell

    cd build/html
    python -m http.server 2333

Then open your browser and go to ``http://localhost:2333`` to browse the documentation.
Feel free to change the port to any other port you prefer.

.. tip::

   If you make changes to the documentation source files, simply rerun ``make html`` to update the output.
