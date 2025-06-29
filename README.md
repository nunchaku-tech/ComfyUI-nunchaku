<div align="center" id="nunchaku_logo">
  <img src="https://raw.githubusercontent.com/mit-han-lab/nunchaku/96615bd93a1f0d2cf98039fddecfec43ce34cc96/assets/nunchaku.svg" alt="logo" width="220"></img>
</div>
<h3 align="center">
<a href="http://arxiv.org/abs/2411.05007"><b>Paper</b></a> | <a href="https://hanlab.mit.edu/projects/svdquant"><b>Website</b></a> | <a href="https://hanlab.mit.edu/blog/svdquant"><b>Blog</b></a> | <a href="https://svdquant.mit.edu"><b>Demo</b></a> | <a href="https://huggingface.co/collections/mit-han-lab/svdquant-67493c2c2e62a1fc6e93f45c"><b>HuggingFace</b></a> | <a href="https://modelscope.cn/collections/svdquant-468e8f780c2641"><b>ModelScope</b></a>
</h3>

<h3 align="center">
<a href="README.md"><b>English</b></a> | <a href="README_ZH.md"><b>中文</b></a>
</h3>

This repository provides the ComfyUI node for [**Nunchaku**](https://github.com/mit-han-lab/nunchaku), an efficient inference engine for 4-bit neural networks quantized with [SVDQuant](http://arxiv.org/abs/2411.05007). For the quantization library, check out [DeepCompressor](https://github.com/mit-han-lab/deepcompressor).

Join our user groups on [**Slack**](https://join.slack.com/t/nunchaku/shared_invite/zt-3170agzoz-NgZzWaTrEj~n2KEV3Hpl5Q), [**Discord**](https://discord.gg/Wk6PnwX9Sm) and [**WeChat**](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/wechat.jpg) for discussions—details [here](https://github.com/mit-han-lab/nunchaku/issues/149). If you have any questions, run into issues, or are interested in contributing, feel free to share your thoughts with us!

# Nunchaku ComfyUI Node

![comfyui](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/ComfyUI-nunchaku/assets/comfyui.jpg)

## News

- **[2025-06-29]** 🔥 **v0.3.3** now supports [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)! Download the quantized model from [HuggingFace](https://huggingface.co/mit-han-lab/nunchaku-flux.1-kontext-dev) or [ModelScope](https://modelscope.cn/models/Lmxyy1999/nunchaku-flux.1-kontext-dev) and use this [workflow](./example_workflows/nunchaku-flux.1-kontext-dev.json) to get started.
- **[2025-06-11]** Starting from **v0.3.2**, you can now **easily install or update the [Nunchaku](https://github.com/mit-han-lab/nunchaku) wheel** using this [workflow](https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/install_wheel.json)!
- **[2025-06-07]** 🚀 **Release Patch v0.3.1!** We bring back **FB Cache** support and fix **4-bit text encoder loading**. PuLID nodes are now optional and won’t interfere with other nodes. We've also added a **NunchakuWheelInstaller** node to help you install the correct [Nunchaku](https://github.com/mit-han-lab/nunchaku) wheel.
- **[2025-06-01]** 🚀 **Release v0.3.0!** This update adds support for multiple-batch inference, [**ControlNet-Union-Pro 2.0**](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0) and initial integration of [**PuLID**](https://github.com/ToTheBeginning/PuLID). You can now load Nunchaku FLUX models as a single file, and our upgraded [**4-bit T5 encoder**](https://huggingface.co/mit-han-lab/nunchaku-t5) now matches **FP8 T5** in quality!
- **[2025-04-16]** 🎥 Released tutorial videos in both [**English**](https://youtu.be/YHAVe-oM7U8?si=cM9zaby_aEHiFXk0) and [**Chinese**](https://www.bilibili.com/video/BV1BTocYjEk5/?share_source=copy_web&vd_source=8926212fef622f25cc95380515ac74ee) to assist installation and usage.
- **[2025-04-09]** 📢 Published the [April roadmap](https://github.com/mit-han-lab/nunchaku/issues/266) and an [FAQ](https://github.com/mit-han-lab/nunchaku/discussions/262) to help the community get started and stay up to date with Nunchaku’s development.
- **[2025-04-05]** 🚀 **Release v0.2.0!** This release introduces [**multi-LoRA**](example_workflows/nunchaku-flux.1-dev.json) and [**ControlNet**](example_workflows/nunchaku-flux.1-dev-controlnet-union-pro.json) support, with enhanced performance using FP16 attention and First-Block Cache. We've also added [**20-series GPU**](examples/flux.1-dev-turing.py) compatibility and official workflows for [FLUX.1-redux](example_workflows/nunchaku-flux.1-redux-dev.json)!

## Installation

We provide tutorial videos to help you install and use Nunchaku on Windows, available in both [**English**](https://youtu.be/YHAVe-oM7U8?si=cM9zaby_aEHiFXk0) and [**Chinese**](https://www.bilibili.com/video/BV1BTocYjEk5/?share_source=copy_web&vd_source=8926212fef622f25cc95380515ac74ee). You can also follow the corresponding step-by-step text guide at [`docs/setup_windows.md`](docs/setup_windows.md). If you run into issues, these resources are a good place to start.

### Step 1: Install the ComfyUI Plugin

You can use the the following way to install the `ComfyUI-nunchaku` plugin.

#### Comfy-CLI

You can easily use [`comfy-cli`](https://github.com/Comfy-Org/comfy-cli) to run ComfyUI with Nunchaku:

```shell
pip install comfy-cli  # Install ComfyUI CLI
comfy install          # Install ComfyUI
comfy node registry-install ComfyUI-nunchaku  # Install Nunchaku
```

#### ComfyUI-Manager

1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI/tree/master) with

   ```shell
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   pip install -r requirements.txt
   ```

1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) with the following commands:

   ```shell
   cd custom_nodes
   git clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager
   ```

1. Launch ComfyUI

   ```shell
   cd ..  # Return to the ComfyUI root directory
   python main.py
   ```

1. Open the Manager, search `ComfyUI-nunchaku` in the Custom Nodes Manager and then install it.

#### Manual Installation

1. Set up [ComfyUI](https://github.com/comfyanonymous/ComfyUI/tree/master) with the following commands:

   ```shell
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   pip install -r requirements.txt
   ```

1. Clone this repository into the `custom_nodes` directory inside ComfyUI:

   ```shell
   cd custom_nodes
   git clone https://github.com/mit-han-lab/ComfyUI-nunchaku nunchaku_nodes
   ```

### Step 2: Install the [Nunchaku](https://github.com/mit-han-lab/nunchaku) Backend

Starting from **ComfyUI-nunchaku v0.3.2**, you can easily install or update the [Nunchaku](https://github.com/mit-han-lab/nunchaku) wheel using this [workflow file](https://github.com/mit-han-lab/ComfyUI-nunchaku/blob/main/example_workflows/install_wheel.json), once all dependencies are installed.

Alternatively, you can follow the manual installation instructions in the [Nunchaku README](https://github.com/mit-han-lab/nunchaku?tab=readme-ov-file#installation).

## Usage

1. **Set Up ComfyUI and Nunchaku**:

   - Nunchaku workflows can be found at [`workflows`](example_workflows). To use them, copy the files to `user/default/workflows` in the ComfyUI root directory:

     ```shell
     cd ComfyUI

     # Create the example_workflows directory if it doesn't exist
     mkdir -p user/default/example_workflows

     # Copy workflow configurations
     cp custom_nodes/nunchaku_nodes/example_workflows/* user/default/example_workflows/
     ```

   - Install any missing nodes (e.g., `comfyui-inpainteasy`) by following [this tutorial](https://github.com/ltdrdata/ComfyUI-Manager?tab=readme-ov-file#support-of-missing-nodes-installation).

1. **Download Required Models**: Follow [this tutorial](https://comfyanonymous.github.io/ComfyUI_examples/flux/) to download the necessary models into the appropriate directories. Alternatively, use the following commands:

   ```shell
   huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir models/text_encoders
   huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir models/text_encoders
   huggingface-cli download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir models/vae
   ```

1. **Run ComfyUI**: To start ComfyUI, navigate to its root directory and run `python main.py`. If you are using `comfy-cli`, simply run `comfy launch`.

1. **Select the Nunchaku Workflow**: Choose one of the Nunchaku workflows (workflows that start with `nunchaku-`) to get started. For the `flux.1-fill` workflow, you can use the built-in **MaskEditor** tool to apply a mask over an image.

1. All the 4-bit models are available at our [HuggingFace](https://huggingface.co/collections/mit-han-lab/svdquant-67493c2c2e62a1fc6e93f45c) or [ModelScope](https://modelscope.cn/collections/svdquant-468e8f780c2641) collection. Except [`svdq-flux.1-t5`](https://huggingface.co/mit-han-lab/svdq-flux.1-t5), please download the **entire model folder** to `models/diffusion_models`.

## Nunchaku Nodes

- **Nunchaku Flux DiT Loader**: A node for loading the FLUX diffusion model.

  - `model_path`: Path to the model folder. You must manually download the model from our [Hugging Face collection](https://huggingface.co/collections/mit-han-lab/nunchaku-6837e7498f680552f7bbb5ad) or [ModelScope collection](https://modelscope.cn/collections/Nunchaku-519fed7f9de94e). Once downloaded, set `model_path` to the corresponding directory.

    > **Note**: Legacy model folders are still supported but will be deprecated in v0.4. To migrate, use our [`merge_safetensors.json`](example_workflows/merge_safetensors.json) workflow to merge your legacy folder into a single `.safetensors` file or redownload the model from the above collections.

  - `cache_threshold`: Controls the [First-Block Cache](https://github.com/chengzeyi/ParaAttention?tab=readme-ov-file#first-block-cache-our-dynamic-caching) tolerance, similar to `residual_diff_threshold` in [WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed). Increasing this value improves speed but may reduce quality. A typical value is 0.12. Setting it to 0 disables the effect.

  - `attention`: Defines the attention implementation method. You can choose between `flash-attention2` or `nunchaku-fp16`. Our `nunchaku-fp16` is approximately 1.2× faster than `flash-attention2` without compromising precision. For Turing GPUs (20-series), where `flash-attention2` is unsupported, you must use `nunchaku-fp16`.

  - `cpu_offload`: Enables CPU offloading for the transformer model. While this reduces GPU memory usage, it may slow down inference.

    - When set to `auto`, it will automatically detect your available GPU memory. If your GPU has more than 14GiB of memory, offloading will be disabled. Otherwise, it will be enabled.
    - **Memory usage will be further optimized in node later.**

  - `device_id`: Indicates the GPU ID for running the model.

  - `data_type`: Defines the data type for the dequantized tensors. Turing GPUs (20-series) do not support `bfloat16` and can only use `float16`.

  - `i2f_mode`: For Turing (20-series) GPUs, this option controls the GEMM implementation mode. `enabled` and `always` modes exhibit minor differences. This option is ignored on other GPU architectures.

- **Nunchaku FLUX LoRA Loader**: A node for loading LoRA modules for SVDQuant FLUX models.

  - Place your LoRA checkpoints in the `models/loras` directory. These will appear as selectable options under `lora_name`.
  - `lora_strength`: Controls the strength of the LoRA module.
  - You can connect **multiple LoRA nodes** together.
  - **Note**: Starting from version 0.2.0, there is no need to convert LoRAs. Simply provide the **original LoRA files** to the loader.

- **Nunchaku Text Encoder Loader V2**: A node for loading the text encoders.

  - Select the CLIP and T5 models to use as `text_encoder1` and `text_encoder2`, following the same convention as in `DualCLIPLoader`. In addition, you may choose to use our enhanced [4-bit T5XXL model](https://huggingface.co/mit-han-lab/nunchaku-t5/resolve/main/awq-int4-flux.1-t5xxl.safetensors) for saving more GPU memory.
  - `t5_min_length`: Sets the minimum sequence length for T5 text embeddings. The default in `DualCLIPLoader` is hardcoded to 256, but for better image quality, use 512 here.

- **Nunchaku Wheel Installer**: A utility node for automatically installing the correct version of [Nunchaku](https://github.com/mit-han-lab/nunchaku) wheels. After installation, please **restart ComfyUI** to apply the changes.

  - `source`: Select the source of the wheel. Available options include [`GitHub Release`](https://github.com/mit-han-lab/nunchaku/releases), [`HuggingFace`](https://huggingface.co/mit-han-lab/nunchaku), and [`ModelScope`](https://modelscope.cn/models/Lmxyy1999/nunchaku).
  - `version`: Choose the compatible [Nunchaku](https://github.com/mit-han-lab/nunchaku) version to install.

- **Nunchaku Text Encoder Loader (will be deprecated in v0.4)**: A node for loading the text encoders.

  - For FLUX, use the following files:

    - `text_encoder1`: `t5xxl_fp16.safetensors` (or FP8/GGUF versions of T5 encoders).
    - `text_encoder2`: `clip_l.safetensors`

  - `t5_min_length`: Sets the minimum sequence length for T5 text embeddings. The default in `DualCLIPLoader` is hardcoded to 256, but for better image quality, use 512 here.

  - `use_4bit_t5`: Specifies whether you need to use our quantized 4-bit T5 to save GPU memory.

  - `int4_model`: Specifies the INT4 T5 location. This option is only used when `use_4bit_t5` is enabled. You can download our INT4 T5 model folder to `models/text_encoders` from [HuggingFace](https://huggingface.co/mit-han-lab/svdq-flux.1-t5) or [ModelScope](https://modelscope.cn/models/Lmxyy1999/svdq-flux.1-t5). For example, you can run the following command:

    ```shell
    huggingface-cli download mit-han-lab/svdq-flux.1-t5 --local-dir models/text_encoders/svdq-flux.1-t5
    ```

    After downloading, specify the corresponding folder name as the `int4_model`.

- **FLUX.1 Depth Preprocessor (will be deprecated in v0.4)** : A legacy node for loading a depth estimation model and producing a corresponding depth map. The `model_path` parameter specifies the location of the model checkpoint. You can manually download the model repository from [Hugging Face](https://huggingface.co/LiheYoung/depth-anything-large-hf) and place it under the `models/checkpoints` directory. Alternatively, use the following CLI command:

  ```shell
  huggingface-cli download LiheYoung/depth-anything-large-hf --local-dir models/checkpoints/depth-anything-large-hf
  ```

  **Note**: This node is deprecated and will be removed in a future release. Please use the updated **"Depth Anything"** node with the `depth_anything_vitl14.pth` model file instead.
