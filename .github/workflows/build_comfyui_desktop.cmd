@echo off
setlocal

REM ==============================
REM  ComfyUI Desktop Build Script
REM ==============================

set PYTHON_VERSION=3.12
set TORCH_VERSION=2.7
REM Node.js version must be complete version string instead of 20 or 20.19
set NODE_VERSION=20.18.0
set YARN_VERSION=4.5.0
set NUNCHAKU_VERSION=1.0.0
set TORCHAUDIO_VERSION=2.7
set TORCHVISION_VERSION=0.22
set CUDA_PIP_INDEX=cu128

REM path to node version manager. Change it if you installed it somewhere else.
set NVM_HOME=%LocalAppData%\nvm

set PYTHON_VERSION_STR=%PYTHON_VERSION:.=%

set WORK_DIR=%cd%

REM Assume Python 3.12 installs here. This is the default location for winget installations.
REM Adjust if your installation path is different.
set PYTHON_EXE="%LocalAppData%\Programs\Python\Python%PYTHON_VERSION_STR%\python.exe"

REM 1. Install Python 3.12 silently with winget, skip if PYTHON_EXE already exists
if exist %PYTHON_EXE% (
    echo Python %PYTHON_VERSION% is already installed. Skip downloading..
) else (
    echo Installing Python %PYTHON_VERSION%...
    winget install -e --id Python.Python.%PYTHON_VERSION% --accept-source-agreements --accept-package-agreements -h
    if %errorlevel% neq 0 (
        echo Failed to install Python %PYTHON_VERSION%
        exit /b 1
    )
)


REM 2. Install uv package
echo Installing uv package...
%PYTHON_EXE% -m pip install --upgrade pip
%PYTHON_EXE% -m pip install uv

REM 3. Install Node.js 20 via NVM
echo Installing Node.js %NODE_VERSION% with NVM...
cd %NVM_HOME%
nvm install %NODE_VERSION%
nvm use %NODE_VERSION%

REM 4. Clone ComfyUI desktop repo
echo Cloning ComfyUI Desktop...
cd %WORK_DIR%
git clone https://github.com/nunchaku-tech/desktop.git
cd desktop

REM 5. Install Yarn using npm
REM Note: this step needs admin permission
echo Installing yarn...
call npm install -g yarn
echo corepack enable
call corepack enable
echo corepack prepare yarn@4.5.0 --activate
call corepack prepare yarn@4.5.0 --activate
@REM yarn use %YARN_VERSION%

REM 6. Install node modules and rebuild electron
echo Rebuilding native modules...
call yarn install
call npx --yes electron-rebuild
call yarn make:assets

REM 7. Overwrite override.txt with torch 2.7 + custom nunchaku wheel
echo Writing override.txt...

git clone https://github.com/nunchaku-tech/ComfyUI-nunchaku.git assets/ComfyUI/custom_nodes/ComfyUI-nunchaku

set NUNCHAKU_URL=https://github.com/nunchaku-tech/nunchaku/releases/download/v%NUNCHAKU_VERSION%/nunchaku-%NUNCHAKU_VERSION%+torch%TORCH_VERSION%-cp%PYTHON_VERSION_STR%-cp%PYTHON_VERSION_STR%-win_amd64.whl

(
echo torch==%TORCH_VERSION%+%CUDA_PIP_INDEX%
echo torchaudio==%TORCHAUDIO_VERSION%+%CUDA_PIP_INDEX%
echo torchvision==%TORCHVISION_VERSION%+%CUDA_PIP_INDEX%
echo nunchaku @ %NUNCHAKU_URL%
) > assets\override.txt
echo nunchaku >> assets\ComfyUI\requirements.txt

REM 8. Build compiled requirements with uv
echo Rebuilding requirements (windows_nvidia.compiled)...
assets\uv\win\uv.exe pip compile assets\ComfyUI\requirements.txt ^
assets\ComfyUI\custom_nodes\ComfyUI-Manager\requirements.txt ^
assets\ComfyUI\custom_nodes\ComfyUI-nunchaku\requirements.txt ^
--emit-index-annotation --emit-index-url --index-strategy unsafe-best-match ^
-o assets\requirements\windows_nvidia.compiled ^
--override assets\override.txt ^
--index-url https://pypi.org/simple ^
--extra-index-url https://download.pytorch.org/whl/%CUDA_PIP_INDEX%

REM 9. Build for NVIDIA users on Windows
echo Building ComfyUI for NVIDIA...
call yarn make:nvidia

echo ========================================
echo ✅ Build process completed successfully!
echo ========================================

endlocal
