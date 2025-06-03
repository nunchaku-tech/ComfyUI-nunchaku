<div align="center" id="nunchaku_logo">
  <img src="https://raw.githubusercontent.com/mit-han-lab/nunchaku/96615bd93a1f0d2cf98039fddecfec43ce34cc96/assets/nunchaku.svg" alt="logo" width="220"></img>
</div>
<h3 align="center">
<a href="http://arxiv.org/abs/2411.05007"><b>论文</b></a> | <a href="https://hanlab.mit.edu/projects/svdquant"><b>网站</b></a> | <a href="https://hanlab.mit.edu/blog/svdquant"><b>博客</b></a> | <a href="https://svdquant.mit.edu"><b>演示</b></a> | <a href="https://huggingface.co/collections/mit-han-lab/svdquant-67493c2c2e62a1fc6e93f45c"><b>HuggingFace</b></a> | <a href="https://modelscope.cn/collections/svdquant-468e8f780c2641"><b>ModelScope</b></a>
</h3>

<h3 align="center">
<a href="README.md"><b>English</b></a> | <a href="README_ZH.md"><b>中文</b></a>
</h3>

该储存库为[**Nunchaku**](https://github.com/mit-han-lab/nunchaku)提供了ComfyUI节点，这是一个用于使用[SVDQuant](http://arxiv.org/abs/2411.05007)量化的 4 位神经网络的高效推理引擎。有关量化库，请查看 [DeepCompressor](https://github.com/mit-han-lab/deepcompressor).

加入我们，在[**Slack**](https://join.slack.com/t/nunchaku/shared_invite/zt-3170agzoz-NgZzWaTrEj~n2KEV3Hpl5Q), [**Discord**](https://discord.gg/Wk6PnwX9Sm) 和 [**微信**](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/nunchaku/assets/wechat.jpg) 上的社区群组进行讨论———详情点击[这里](https://github.com/mit-han-lab/nunchaku/issues/149). 如果您有任何问题、遇到问题或有兴趣做出贡献，请随时与我们分享您的想法！

# Nunchaku ComfyUI节点

![comfyui](https://huggingface.co/mit-han-lab/nunchaku-artifacts/resolve/main/ComfyUI-nunchaku/assets/comfyui.jpg)

## 最新消息

- **[2025-06-01]** 🚀 **发布 v0.3.0 版本！** 本次更新新增了对多批次推理的支持，集成了 [**ControlNet-Union-Pro 2.0**](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0) 并初步整合了 [**PuLID**](https://github.com/ToTheBeginning/PuLID)。您现在可以将 Nunchaku FLUX 模型作为单个文件加载，而我们升级后的 [**4位 T5 编码器**](https://huggingface.co/mit-han-lab/nunchaku-t5) 在质量上已可媲美 **FP8 T5**！
- **[2025-04-16]** 🎥 发布了[**英文**](https://youtu.be/YHAVe-oM7U8?si=cM9zaby_aEHiFXk0)和[**中文**](https://www.bilibili.com/video/BV1BTocYjEk5/?share_source=copy_web&vd_source=8926212fef622f25cc95380515ac74ee)教程视频，协助安装和使用Nunchaku。
- **[2025-04-09]** 📢 发布了 [4月更新计划](https://github.com/mit-han-lab/nunchaku/issues/266)和[常见问题解答](https://github.com/mit-han-lab/nunchaku/discussions/262)来帮助社区朋友快速入门并及时了解Nunchaku的发展情况。
- **[2025-04-05]** 🚀 **v0.2.0发布!** 这个版本支持了[**多LoRA**](example_workflows/nunchaku-flux.1-dev.json)和[**ControlNet**](example_workflows/nunchaku-flux.1-dev-controlnet-union-pro.json)，并且使用FP16 attention和First-Block Cache来增强性能. 我们添加了对[**Invidia20系显卡**](examples/flux.1-dev-turing.py)的支持，并制作了[FLUX.1-redux](example_workflows/nunchaku-flux.1-redux-dev.json)的官方工作流。

## 安装方法

我们提供了在 Windows 上安装和使用 Nunchaku 的教学视频，支持[**英文**](https://youtu.be/YHAVe-oM7U8?si=cM9zaby_aEHiFXk0)和[**中文**](https://www.bilibili.com/video/BV1BTocYjEk5/?share_source=copy_web&vd_source=8926212fef622f25cc95380515ac74ee)两个版本。同时，你也可以参考对应的图文教程 [`docs/setup_windows.md`](docs/setup_windows.md)。如果在安装过程中遇到问题，建议优先查阅这些资源。

### 前置工作

请先参阅[README.md](https://github.com/mit-han-lab/nunchaku?tab=readme-ov-file#installation)来安装 `nunchaku`。

### Comfy-CLI：ComfyUI的命令工具

您可以使用[`comfy-cli`](https://github.com/Comfy-Org/comfy-cli)在ComfyUI中运行Nunchaku：

```shell
pip install comfy-cli  # Install ComfyUI CLI
comfy install          # Install ComfyUI
comfy node registry-install ComfyUI-nunchaku  # Install Nunchaku
```

### ComfyUI-Manager

1. 首先使用以下指令安装[ComfyUI](https://github.com/comfyanonymous/ComfyUI/tree/master)

   ```shell
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   pip install -r requirements.txt
   ```

1. 使用以下命令安装[ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)（这是一个节点管理插件）

   ```shell
   cd custom_nodes
   git clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager
   ```

1. 启动ComfyUI

   ```shell
   cd ..  # Return to the ComfyUI root directory
   python main.py
   ```

1. 打开Manager后, 在Custom Nodes Manager中搜索`ComfyUI-nunchaku`节点并且下载它then install it.

### 手动安装

1. 使用以下命令设置[ComfyUI](https://github.com/comfyanonymous/ComfyUI/tree/master)

   ```shell
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   pip install -r requirements.txt
   ```

1. 将此仓库克隆到 ComfyUI 中的目录中：`custom_nodes`

   ```shell
   cd custom_nodes
   git clone https://github.com/mit-han-lab/ComfyUI-nunchaku nunchaku_nodes
   ```

## 使用说明

1. **设置ComfyUI和Nunchaku**:

   - Nunchaku的工作流可以在[`workflows`](example_workflows)找到。想要找到它们，请将文件复制到ComfyUI的根目录中： `user/default/workflows`

     ```shell
     cd ComfyUI

     # Create the example_workflows directory if it doesn't exist
     mkdir -p user/default/example_workflows

     # Copy workflow configurations
     cp custom_nodes/nunchaku_nodes/example_workflows/* user/default/example_workflows/
     ```

   - 按照[本教程](https://github.com/ltdrdata/ComfyUI-Manager?tab=readme-ov-file#support-of-missing-nodes-installation).安装所有缺失节点 (例如 `comfyui-inpainteasy`)

1. **下载必要模型**: 按照[本教程](https://comfyanonymous.github.io/ComfyUI_examples/flux/)把必要的模型下载到对应的目录中。或者使用以下命令：

   ```shell
   huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir models/text_encoders
   huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir models/text_encoders
   huggingface-cli download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir models/vae
   ```

1. **运行ComfyUI**： 要启动 ComfyUI,请导航到其根目录并运行：`python main.py`。如果您使用的是 `comfy-cli`, 请运行 `comfy launch`.

1. **选择Nunchaku工作流**：选择一个Nunchaku工作流开始使用(文件名以`nunchaku-`为开头的工作流)。在使用`flux.1-fill`的工作流时, 可以使用ComfyUI内置的**MaskEditor**工具来涂抹遮罩。

5.所有四位模型都可以在[HuggingFace](https://huggingface.co/collections/mit-han-lab/svdquant-67493c2c2e62a1fc6e93f45c)或者[ModelScope](https://modelscope.cn/collections/svdquant-468e8f780c2641)中找到。除了[`svdq-flux.1-t5`](https://huggingface.co/mit-han-lab/svdq-flux.1-t5)，请将**整个模型文件夹**下载并放入到`models/diffusion_models`文件夹中。

## Nunchaku节点

- **Nunchaku Flux DiT Loader节点**：用于加载Flux扩散模型的节点

  - `model_path`：指定模型的位置。您需要从我们的[Hugging Face](https://huggingface.co/collections/mit-han-lab/svdquant-67493c2c2e62a1fc6e93f45c)或者[ModelScope](https://modelscope.cn/collections/svdquant-468e8f780c2641)中手动下载模型文件夹。例如：运行

    > **注意**：旧版模型文件夹仍然受支持，但将在 v0.4 中弃用。要迁移，请使用我们的 [`merge_safetensors.json`](example_workflows/merge_safetensors.json) 工作流程将旧版文件夹合并为单个 `.safetensors` 文件，或从上述集合中重新下载模型。

  - `cache_threshold`：控制[First-Block Cache](https://github.com/chengzeyi/ParaAttention?tab=readme-ov-file#first-block-cache-our-dynamic-caching)的容差，类似于[WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed)中的`residual_diff_threshold`。增加此值可以提高速度，但可能会降低质量。典型值为 0.12。将其设置为 0 将禁用该效果。

  - `attention`：定义 attention 的实现方法. 您可以在`flash-attention2`或`nunchaku-fp16`之间进行选择。我们的`nunchaku-fp16`在不影响精度的情况下大约比`flash-attention2`快1.2x倍。对于Turing架构的显卡(20系), 如果不支持`flash-attention2`，则必须使用 `nunchaku-fp16`。

  - `cpu_offload`：为transformer模型启用CPU卸载。虽然这减少了GPU内存的使用，但它可能会减慢推理速度。

    - 当设置为`auto`的时候，它将自动检测您的可用 GPU 内存。如果您的GPU内存超过14GiB，则将禁用卸载。否则，它将启用。
    - **将来将在节点中进一步优化内存使用。**

  - `device_id`：模型运行时使用的GPU ID。

  - `data_type`：定义去量子化张量的数据类型。Turing架构的GPU(20系)不支持`bfloat16`，只能只用`float16`.

  - `i2f_mode`：对于Turing架构的GPU(20系)，此选项控制GEMM的实现模式。`enabled`和`always`模式的差异细微。在其他架构的GPU上可以忽略这个选项。

- **Nunchaku FLUX LoRA Loader**：用于加载SVDQuant FLUX模型的LoRA模型的节点

  - 将LoRA Checkpints文件放在`models/loras`目录中。这些LoRA模型将在`lora_name`下显示为可选选项。
  - `lora_strength`：控制LoRA模型的强度。
  - 您可以将多个**multiple LoRA nodes**模型连接使用
  - **注**：从0.2.0版本开始，不需要转换LoRA了。可以在加载器中加载原始的LoRA文件

- **Nunchaku Text Encoder Loader V2**：用于加载文本编码器的节点。

- 选择 CLIP 和 T5 模型作为 `text_encoder1` 和 `text_encoder2`，遵循与 `DualCLIPLoader` 相同的方式。此外，您可以选择使用我们增强的 [4 位 T5XXL 模型](https://huggingface.co/mit-han-lab/nunchaku-t5/resolve/main/awq-int4-flux.1-t5xxl.safetensors)，以节省更多 GPU 内存。

- `t5_min_length`：设置 T5 文本嵌入的最小序列长度。`DualCLIPLoader` 中的默认值硬编码为 256，但为了获得更好的图像质量，此处请使用 512。

- **Nunchaku Text Encoder Loader (将在v0.4版本弃用)**：用于加载文本编码器的节点。

  - 对于FLUX，请使用以下文件：

    - `text_encoder1`: `t5xxl_fp16.safetensors`（或 T5 编码器的 FP8/GGUF 版本）。
    - `text_encoder2`: `clip_l.safetensors`

  - `t5_min_length`：设置 T5 文本嵌入的最小序列长度。在`DualCLIPLoader`中的默认硬编码为256，但为了获得更好的图像质量，请在此处使用 512。

  - `use_4bit_t5`：指定您是否需要使用我们的量化4位T5来节省GPU内存

  - `int4_model`：指定INT4 T5的位置。这个选项仅在`use_4bit_t5`启用时使用。您可以从[HuggingFace](https://huggingface.co/mit-han-lab/svdq-flux.1-t5)或[ModelScope](https://modelscope.cn/models/Lmxyy1999/svdq-flux.1-t5)下载模型到`models/text_encoders`文件夹。例如，您可以使用以下命令：

    ```shell
    huggingface-cli download mit-han-lab/svdq-flux.1-t5 --local-dir models/text_encoders/svdq-flux.1-t5
    ```

    After downloading, specify the corresponding folder name as the `int4_model`.

- **FLUX.1 Depth Preprocessor (将在v0.4版本弃用)**：一个用于加载depth模型并生成相应深度图的旧节点。`model_path`参数指定checkpoint模型的位置。您可以从[Hugging Face](https://huggingface.co/LiheYoung/depth-anything-large-hf) 下载模型并放在`models/checkpoints`目录中。或者，使用以下CLI命令：

  ```shell
  huggingface-cli download LiheYoung/depth-anything-large-hf --local-dir models/checkpoints/depth-anything-large-hf
  ```

  **注意**：此节点已弃用，并将在未来发行版中删除。请改用更新后的 **"Depth Anything"** 节点来替代加载`depth_anything_vitl14.pth`。
