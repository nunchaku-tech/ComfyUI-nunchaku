@echo off
setlocal

REM ==============================
REM  ComfyUI Desktop Build Script
REM ==============================

REM Use environment variables if set, otherwise use defaults
if not defined TORCH_VERSION set TORCH_VERSION=2.7
if not defined NUNCHAKU_VERSION set NUNCHAKU_VERSION=1.0.0

echo Using TORCH_VERSION=%TORCH_VERSION%
echo Using NUNCHAKU_VERSION=%NUNCHAKU_VERSION%

REM Set version-specific defaults, which is the desktop Python version
set PYTHON_VERSION=3.12
REM Node.js version must be complete version string instead of 20 or 20.19
set NODE_VERSION=20.18.0
set YARN_VERSION=4.5.0
set CUDA_PIP_INDEX=cu128

REM Set torchaudio and torchvision versions based on TORCH_VERSION
if "%TORCH_VERSION%"=="2.7" (
    set TORCHAUDIO_VERSION=2.7
    set TORCHVISION_VERSION=0.22
) else if "%TORCH_VERSION%"=="2.8" (
    set TORCHAUDIO_VERSION=2.8
    set TORCHVISION_VERSION=0.23
) else if "%TORCH_VERSION%"=="2.9" (
    set TORCHAUDIO_VERSION=2.9
    set TORCHVISION_VERSION=0.24
) else (
    echo Warning: Unknown TORCH_VERSION=%TORCH_VERSION%, using default versions
    set TORCHAUDIO_VERSION=%TORCH_VERSION%
    set TORCHVISION_VERSION=0.22
)

set PYTHON_VERSION_STR=%PYTHON_VERSION:.=%

REM 1. Install uv package
echo Installing uv package...
python -m pip install --upgrade pip || (
    echo ERROR: Failed to upgrade pip
    exit /b 1
)
python -m pip install uv || (
    echo ERROR: Failed to install uv
    exit /b 1
)

REM 2. Clone ComfyUI desktop repo
echo Cloning ComfyUI Desktop...
git clone https://github.com/nunchaku-tech/desktop.git || (
    echo ERROR: Failed to clone desktop repository
    exit /b 1
)
cd desktop || (
    echo ERROR: Failed to enter desktop directory
    exit /b 1
)
git checkout ComfyUI-nunchaku-1.0.2 || (
    echo ERROR: Failed to checkout dev branch
    exit /b 1
)
git log -1 --oneline

REM 3. Install Yarn using corepack
echo Installing yarn...
call corepack enable || (
    echo ERROR: Failed to enable corepack
    exit /b 1
)
call corepack prepare yarn@%YARN_VERSION% --activate || (
    echo ERROR: Failed to prepare yarn
    exit /b 1
)

REM 4. Install node modules and rebuild electron
echo Rebuilding native modules...
call yarn install || (
    echo ERROR: Failed to run yarn install
    exit /b 1
)
call npx --yes electron-rebuild || (
    echo ERROR: Failed to rebuild electron
    exit /b 1
)
call yarn make:assets || (
    echo ERROR: Failed to make assets
    exit /b 1
)

REM 5. Overwrite override.txt with torch version + custom nunchaku wheel
echo Writing override.txt...

xcopy /E /I /Y /H ..\ComfyUI-nunchaku assets\ComfyUI\custom_nodes\ComfyUI-nunchaku || (
    echo ERROR: Failed to copy ComfyUI-nunchaku to assets
    exit /b 1
)

set NUNCHAKU_URL=https://github.com/nunchaku-tech/nunchaku/releases/download/v%NUNCHAKU_VERSION%/nunchaku-%NUNCHAKU_VERSION%+torch%TORCH_VERSION%-cp%PYTHON_VERSION_STR%-cp%PYTHON_VERSION_STR%-win_amd64.whl

(
echo torch==%TORCH_VERSION%+%CUDA_PIP_INDEX%
echo torchaudio==%TORCHAUDIO_VERSION%+%CUDA_PIP_INDEX%
echo torchvision==%TORCHVISION_VERSION%+%CUDA_PIP_INDEX%
echo nunchaku @ %NUNCHAKU_URL%
) > assets\override.txt
echo nunchaku >> assets\ComfyUI\requirements.txt

REM 6. Build compiled requirements with uv
echo Rebuilding requirements (windows_nvidia.compiled)...
assets\uv\win\uv.exe pip compile assets\ComfyUI\requirements.txt ^
assets\ComfyUI\custom_nodes\ComfyUI-Manager\requirements.txt ^
assets\ComfyUI\custom_nodes\ComfyUI-nunchaku\requirements.txt ^
--emit-index-annotation --emit-index-url --index-strategy unsafe-best-match ^
-o assets\requirements\windows_nvidia.compiled ^
--override assets\override.txt ^
--index-url https://pypi.org/simple ^
--extra-index-url https://download.pytorch.org/whl/%CUDA_PIP_INDEX% || (
    echo ERROR: Failed to compile requirements with uv
    exit /b 1
)

REM 7. Build for NVIDIA users on Windows
echo Building ComfyUI for NVIDIA...
call yarn make:nvidia || (
    echo ERROR: Failed to build ComfyUI for NVIDIA
    exit /b 1
)

echo ========================================
echo ✅ Build process completed successfully!
echo ========================================

endlocal
