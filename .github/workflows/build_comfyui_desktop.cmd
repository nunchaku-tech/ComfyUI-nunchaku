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
) else (
    echo Warning: Unknown TORCH_VERSION=%TORCH_VERSION%, using default versions
    set TORCHAUDIO_VERSION=%TORCH_VERSION%
    set TORCHVISION_VERSION=0.22
)

set PYTHON_VERSION_STR=%PYTHON_VERSION:.=%

REM 1. Install uv package
echo Installing uv package...
python -m pip install --upgrade pip
python -m pip install uv

REM 2. Clone ComfyUI desktop repo
echo Cloning ComfyUI Desktop...
git clone https://github.com/nunchaku-tech/desktop.git
cd desktop
git checkout ed6400a

REM 3. Install Yarn using corepack
echo Installing yarn...
call corepack enable
call corepack prepare yarn@%YARN_VERSION% --activate

REM 4. Install node modules and rebuild electron
echo Rebuilding native modules...
call yarn install
call npx --yes electron-rebuild
call yarn make:assets

REM 5. Overwrite override.txt with torch version + custom nunchaku wheel
echo Writing override.txt...

xcopy /E /I /Y /H ..\ComfyUI-nunchaku assets\ComfyUI\custom_nodes\ComfyUI-nunchaku

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
--extra-index-url https://download.pytorch.org/whl/%CUDA_PIP_INDEX%

REM 7. Build for NVIDIA users on Windows
echo Building ComfyUI for NVIDIA...
call yarn make:nvidia

echo ========================================
echo ✅ Build process completed successfully!
echo ========================================

endlocal
