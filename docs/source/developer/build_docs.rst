Building the Documentation
==========================

Follow this guide to build the ComfyUI-nunchaku documentation locally using Sphinx.

Step 1: Set Up the Environment
------------------------------

First, ensure your environment is prepared. This process is similar to :doc:`Installation <../get_started/installation>`.
Make sure you have installed all the necessary development and documentation dependencies of Nunchaku with the ``dev,docs`` extras:

.. code-block:: shell

    pip install -e ".[dev,docs]"

Alternatively, you can install the documentation dependencies manually:

.. code-block:: shell

    pip install sphinx sphinx-tabs myst-parser sphinx-copybutton breathe sphinxcontrib-mermaid nbsphinx jupyter ipykernel graphviz sphinxext-rediraffe
    pip install furo sphinxawesome-theme sphinx-book-theme sphinx-rtd-theme

Step 2: Set Up Directory Structure and Install ComfyUI
-------------------------------------------------------

The documentation build process requires a specific directory structure where ``ComfyUI-nunchaku``, ``ComfyUI``, and ``comfyui_nunchaku`` are at the same level.

1. Navigate to the parent directory and create a symbolic link:

   .. code-block:: shell

       cd ..
       ln -s ComfyUI-nunchaku comfyui_nunchaku

   This creates the following directory structure:

   .. code-block:: text

       parent-directory/
       ├── ComfyUI-nunchaku/    # Your clone of this repository
       ├── comfyui_nunchaku     # Symbolic link to ComfyUI-nunchaku
       └── ComfyUI/             # Will be created in the next step

   .. note::

      The ``comfyui_nunchaku`` symbolic link is required because the autodoc system uses this as the package name for the nodes.
      The link allows the documentation build process to import and document the nodes correctly.

2. Install ComfyUI using comfy-cli:

   .. code-block:: shell

       pip install comfy-cli
       yes | comfy --here install --nvidia --skip-torch-or-directml --version 0.3.60

   .. note::

      ComfyUI installation is required because the autodoc process needs to import ComfyUI packages such as ``folder_paths`` to successfully build the API documentation.

Step 3: Build the Documentation
-------------------------------

Navigate to the ``docs`` directory and build the HTML documentation:

.. code-block:: shell

    cd docs
    make clean
    make html

This will generate the HTML files in the ``docs/build/html`` directory.

Step 4: Preview the Documentation
---------------------------------

To view the generated documentation locally, start a simple HTTP server:

.. code-block:: shell

    cd build/html
    python -m http.server 2333

Then open your browser and go to ``http://localhost:2333`` to browse the documentation.
Feel free to change the port to any other port you prefer.

.. tip::

   If you make changes to the documentation source files, simply rerun ``make html`` to update the output.
