# ComfyUI Desktop Windows Build Setup

This README provides instructions for setting up and running the Windows build script for ComfyUI Desktop.

## Prerequisites

### Visual Studio with C++ Support

Visual Studio 2019 or later with the Desktop C++ workload is **required** for `node-gyp` compilation. This is essential for building native Node.js modules.

#### Required Components:

1. **Visual Studio Community 2022** (17.12.1 or later)
2. **Desktop development with C++ workload**
3. **MSVC v143 x64 Spectre-mitigated libraries** (v14.42-17.12 or latest)

#### Installing Spectre-mitigated Libraries:

The Spectre-mitigated libraries are **critical** and must be installed separately:

1. Open the Visual Studio Installer
2. Click "Modify" on your Visual Studio 2022 Community installation
3. Go to the "Individual Components" tab
4. Search for "Spectre"
5. Check the boxes for: **"MSVC v143 - VS 2022 C++ x64/x86 Spectre-mitigated libs"**
6. Install the selected components

> **Note**: Without the Spectre-mitigated libraries, the build process will fail during native module compilation.

### System Requirements

- **Windows 10/11** (64-bit)
- **Administrative privileges** (required for some package installations)
- **Git** (for cloning repositories)
- **Internet connection** (for downloading dependencies)

## Build Script Overview

The `build_comfyui_desktop.cmd` script automates the entire build process by:

1. Installing Python 3.12 via winget
2. Installing the `uv` package manager
3. Installing Node Version Manager (NVM) for Windows
4. Setting up Node.js 20.18.0
5. Installing Yarn 4.5.0
6. Cloning the ComfyUI Desktop repository
7. Installing dependencies and building the application
8. Configuring PyTorch with CUDA support and Nunchaku integration

## Script Configuration

The script uses the following default versions (configurable at the top of the script):

```batch
set PYTHON_VERSION=3.12
set TORCH_VERSION=2.7
set NODE_VERSION=20.18.0
set YARN_VERSION=4.5.0
set NUNCHAKU_VERSION=1.0.0
set CUDA_PIP_INDEX=cu128
```

## Important Notes

- **Administrative Rights**: The script requires admin privileges for installing global packages
- **Python Path**: The script assumes Python installs to `%LocalAppData%\Programs\Python\Python312\python.exe`
- **NVM Path**: Expects NVM to be installed at `%LocalAppData%\nvm`
- **CUDA Support**: Configured for CUDA 12.8 by default (cu128 index)

## Build Output

Upon successful completion, the script will:

- Create a fully configured ComfyUI Desktop environment
- Generate NVIDIA-optimized builds with CUDA support
- Include Nunchaku acceleration support
- Produce ready-to-use application binaries

## Troubleshooting

### Common Issues:

1. **"Module was compiled against a different Node.js version"**

   - Run: `npx electron-rebuild` in the project directory

2. **Missing Spectre-mitigated libraries error**

   - Ensure you've installed the Spectre-mitigated libraries as described above

3. **Permission denied errors**

   - Run Windows CMD as Administrator. Do not use PowerShell.

4. **Python installation fails**
   - Manually install Python 3.12 from python.org or Microsoft Store. You can also try Anaconda or Miniconda.

## Support

For detailed development information and additional troubleshooting, refer to the main ComfyUI Desktop README.md in the desktop project repository.
