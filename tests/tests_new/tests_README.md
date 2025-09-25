
# Nunchaku Workflow Tests

This test suite validates Nunchaku workflows using [ComfyUI LTS](https://github.com/hiddenswitch/ComfyUI). The tests automatically discover and execute all workflow files stored in the `workflows_api/` directory.

## Overview

The test suite performs the following:
- **Automatic Discovery**: Finds all `.json` workflow files in the `workflows_api/` directory
- **Workflow Validation**: Validates workflow structure and executes them
- **Output Verification**: Checks that workflows produce expected outputs (images, audio, or text)
- **Model Dependency Handling**: Gracefully skips tests when required models are missing
- **Image Input Management**: Automatically copies test images for workflows that require them

## Prerequisites

### System Requirements
- **Python 3.12** (required)
- **GPU** (required for running the workflows)
- **Virtual environment** (recommended)

### Dependencies
- **pytest** - Testing framework
- **pytest-asyncio** - Async testing support

### Model Requirements
You must manually download and install:
- **Nunchaku backend**
- **ComfyUI-Nunchaku custom node**
- **Diffusion models** (int4 or fp4 format)
- **Text encoders**
- **VAE models**

## Installation

### 1. Set up Python environment
```bash
#recommend a virtual python environment e.g. conda
```

### 2. Install ComfyUI LTS and Nunchaku backend
```bash
pip install "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
# for Nunchaku backend, use the one match the PyTorch version
pip install nunchaku-1.0.0+torch2.8-cp312-cp312-linux_x86_64.whl
```

### 3. Install test dependencies
```bash
pip install pytest pytest-asyncio
```
### 4. Download Required Models

The test suite requires several Nunchaku models to run the workflows. Follow these steps:

cd to the tests_new directory. 

1. **Start ComfyUI**: Run `comfyui` from the `tests_new` directory to launch the ComfyUI interface. Then control + c to exit. 
2. **Directories creation (automatically)**: step 1 ComfyUI will automatically create a `models` directory structure and `custom_node` directory in the current folder (tests_new). 
3. **Download Required Models**: Download and place these models in their corresponding folders:
   - **Diffusion models** (int4 or fp4 format)
   - **Text encoders**
   - **VAE models**
4. **Install ComfyUI-Nunchaku custom node** install it to the `custom_node` directory
```bash
cd custom_nodes
#git clone and install the node
git clone https://github.com/nunchaku-tech/ComfyUI-nunchaku.git
cd ComfyUI-nunchaku
pip install -r requirements.txt 
```

**Note**: The specific model files required depend on the workflows you're testing. Check the workflow documentation or error messages for exact model requirements.


## Test Data Setup

### Workflow Files
Place your ComfyUI workflow files in the `workflows_api/` directory:
- Files must be in **API format** (export workflows as API from ComfyUI)
- Only `.json` files are automatically discovered
- Current workflows:
  - `flux.1-dev-nunchaku-api.json` - Flux.1 Dev workflow
  - `nunchaku-qwen-image-api.json` - Qwen image generation
  - `nunchaku-qwen-image-edit-api.json` - Qwen image editing

### Test Images
For workflows that use `LoadImage` nodes:
- Place test images in the `images_input/` directory
- Supported formats: PNG, JPG
- The test suite will automatically copy these to ComfyUI's input directory
- Current test image: `input.png`

## Running Tests

### Basic test execution
```bash
pytest test_nunchaku_workflows.py -v
```

### Available test options
```bash
# Run with verbose output
pytest test_nunchaku_workflows.py -v

# Run specific workflow test
pytest test_nunchaku_workflows.py::test_custom_workflow -v -k "flux"

pytest test_nunchaku_workflows.py::test_custom_workflow -v -k "qwen-image-edit"



# Run with custom pytest options
pytest test_nunchaku_workflows.py --tb=short --maxfail=5
```

## Expected Results

When all tests pass successfully, you should see output similar to this:

```bash
test_nunchaku_workflows.py::test_custom_workflow[nunchaku-qwen-image-api.json-workflow_file0] PASSED [ 33%]
test_nunchaku_workflows.py::test_custom_workflow[nunchaku-qwen-image-edit-api.json-workflow_file1] PASSED [ 66%]
test_nunchaku_workflows.py::test_custom_workflow[flux.1-dev-nunchaku-api.json-workflow_file2] PASSED [100%]

========================= 3 passed in 45.23s =========================
```






