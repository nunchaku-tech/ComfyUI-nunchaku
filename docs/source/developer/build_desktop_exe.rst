Building ComfyUI Desktop EXE
============================

This document describes how to build the ComfyUI Desktop executable (EXE) on Windows systems.

System Requirements
===================

Hardware and Operating System
------------------------------

- Windows 10/11 (64-bit)
- Administrative privileges (required for some package installations)
- Git (for cloning repositories)
- Stable internet connection (for downloading dependencies)

Visual Studio C++ Development Environment
------------------------------------------

Visual Studio 2019 or later with the Desktop C++ workload is **required** for ``node-gyp`` compilation of native Node.js modules.

Required Components
~~~~~~~~~~~~~~~~~~~

1. **Visual Studio Community 2022** (17.12.1 or later)
2. **Desktop development with C++ workload**
3. **MSVC v143 x64 Spectre-mitigated libraries** (v14.42-17.12 or latest)

Installing Spectre-mitigated Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Spectre-mitigated libraries are **critical dependencies** and must be installed separately:

1. Open the Visual Studio Installer
2. Click "Modify" on your Visual Studio 2022 Community installation
3. Go to the "Individual Components" tab
4. Search for "Spectre"
5. Check the boxes for: **"MSVC v143 - VS 2022 C++ x64/x86 Spectre-mitigated libs"**
6. Install the selected components

.. warning::
   Without the Spectre-mitigated libraries, the build process will fail during native module compilation.

Build Script Overview
=====================

The ``scripts/build_comfyui_desktop.cmd`` script automates the entire build process:

1. Installing Python 3.12 via winget
2. Installing the ``uv`` package manager
3. Installing Node Version Manager (NVM) for Windows
4. Setting up Node.js 20.18.0
5. Installing Yarn 4.5.0
6. Cloning the ComfyUI Desktop repository
7. Installing dependencies and building the application
8. Configuring PyTorch with CUDA support and Nunchaku integration

Script Configuration
====================

The script uses the following default versions (configurable at the top of the script):

.. code-block:: batch

   set PYTHON_VERSION=3.12
   set TORCH_VERSION=2.7
   set NODE_VERSION=20.18.0
   set YARN_VERSION=4.5.0
   set NUNCHAKU_VERSION=1.0.0
   set CUDA_PIP_INDEX=cu128

Path Configuration
------------------

- **Python Path**: The script assumes Python installs to ``%LocalAppData%\Programs\Python\Python312\python.exe``
- **NVM Path**: Expects NVM to be installed at ``%LocalAppData%\nvm``
- **CUDA Support**: Configured for CUDA 12.8 (cu128 index) by default

If your installation paths differ, modify the ``PYTHON_EXE`` and ``NVM_HOME`` variables in the script accordingly.

Usage
=====

1. Open Windows Command Prompt (CMD) as **Administrator**

   .. warning::
      Use CMD, not PowerShell.

2. Navigate to the project's ``scripts`` directory:

   .. code-block:: batch

      cd path\to\ComfyUI-nunchaku\scripts

3. Run the build script:

   .. code-block:: batch

      build_comfyui_desktop.cmd

4. Wait for the build to complete. The entire process may take 30 minutes to 1 hour, depending on network speed and machine performance.

Build Output
============

Upon successful completion, the script will:

- Create a fully configured ComfyUI Desktop environment
- Generate NVIDIA-optimized builds with CUDA support
- Include Nunchaku acceleration support
- Produce ready-to-use application binaries

Build artifacts are typically located in the ``desktop/out`` directory.

Troubleshooting
===============

Common Issues
-------------

1. **"Module was compiled against a different Node.js version" error**

   Run in the project directory:

   .. code-block:: batch

      npx electron-rebuild

2. **Missing Spectre-mitigated libraries error**

   Ensure you've installed the Spectre-mitigated libraries following the steps above.

3. **Permission denied errors**

   Run Windows CMD as Administrator. Do not use PowerShell.

4. **Python installation fails**

   Manually install Python 3.12 from `python.org <https://www.python.org/>`_ or Microsoft Store.
   You can also try using Anaconda or Miniconda.

5. **NVM installation fails**

   Manually download and install from `NVM for Windows <https://github.com/coreybutler/nvm-windows>`_.

6. **Network connection issues**

   If in mainland China, some dependency downloads may be slow or fail. Consider:

   - Configuring mirror sources (e.g., Tsinghua, Alibaba Cloud mirrors)
   - Using a VPN or proxy

Maintenance Notes
=================

Version Updates
---------------

When updating dependency versions, modify the version variables at the top of the script:

.. code-block:: batch

   set PYTHON_VERSION=3.12
   set TORCH_VERSION=2.7
   set NODE_VERSION=20.18.0
   set YARN_VERSION=4.5.0
   set NUNCHAKU_VERSION=1.0.0

Customizing CUDA Version
-------------------------

To use a different CUDA version, modify the ``CUDA_PIP_INDEX`` variable:

.. code-block:: batch

   REM For CUDA 11.8
   set CUDA_PIP_INDEX=cu118

   REM For CUDA 12.1
   set CUDA_PIP_INDEX=cu121

References
==========

- `ComfyUI Desktop Official Repository <https://github.com/Comfy-Org/desktop>`_
- `Node Version Manager for Windows <https://github.com/coreybutler/nvm-windows>`_
- `Python Official Website <https://www.python.org/>`_
- `Visual Studio Downloads <https://visualstudio.microsoft.com/>`_
