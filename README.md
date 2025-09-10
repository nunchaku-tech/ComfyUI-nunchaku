<div align="center" id="nunchaku_logo">
  <img src="https://raw.githubusercontent.com/nunchaku-tech/nunchaku/96615bd93a1f0d2cf98039fddecfec43ce34cc96/assets/nunchaku.svg" alt="logo" width="220"></img>
</div>
<h3 align="center">
<a href="http://arxiv.org/abs/2411.05007"><b>Paper</b></a> | <a href="https://nunchaku.tech/docs/ComfyUI-nunchaku/"><b>Docs</b></a> | <a href="https://hanlab.mit.edu/projects/svdquant"><b>Website</b></a> | <a href="https://hanlab.mit.edu/blog/svdquant"><b>Blog</b></a> | <a href="https://svdquant.mit.edu"><b>Demo</b></a> | <a href="https://huggingface.co/nunchaku-tech"><b>Hugging Face</b></a> | <a href="https://modelscope.cn/organization/nunchaku-tech"><b>ModelScope</b></a>
</h3>

<h3 align="center">
<a href="README.md"><b>English</b></a> | <a href="README_ZH.md"><b>中文</b></a>
</h3>

This repository provides the ComfyUI plugin for [**Nunchaku**](https://github.com/nunchaku-tech/nunchaku), an efficient inference engine for 4-bit neural networks quantized with [SVDQuant](http://arxiv.org/abs/2411.05007). For the quantization library, check out [DeepCompressor](https://github.com/nunchaku-tech/deepcompressor).

Join our user groups on [**Discord**](https://discord.gg/Wk6PnwX9Sm) and [**WeChat**](https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/nunchaku/assets/wechat.jpg) for discussions—details [here](https://github.com/nunchaku-tech/nunchaku/issues/149). If you have any questions, run into issues, or are interested in contributing, feel free to share your thoughts with us!

# Nunchaku ComfyUI Plugin

![comfyui](https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/comfyui.jpg)

## News

- **[2025-09-09]** 🔥 Released **4-bit Qwen-Image-Edit** together with the [4/8-step Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning) variants! Models are available on [Hugging Face](https://huggingface.co/nunchaku-tech/nunchaku-qwen-image). Try them out with this [workflow](./example_workflows/nunchaku-qwen-image-edit.json)!
- **[2025-09-04]** 🚀 Official release of **Nunchaku v1.0.0**! Qwen-Image now supports **asynchronous offloading**, cutting Transformer VRAM usage to as little as **3 GiB** with no performance loss. You can also try our pre-quantized [4/8-step Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning) models on [Hugging Face](https://huggingface.co/nunchaku-tech/nunchaku-qwen-image) or [ModelScope](https://modelscope.cn/models/nunchaku-tech/nunchaku-qwen-image).
- **[2025-08-23]** 🚀 **v1.0.0** adds support for [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)! Check [this workflow](example_workflows/nunchaku-qwen-image.json) to get started. LoRA support is coming soon.
- **[2025-07-17]** 📘 The official [**ComfyUI-nunchaku documentation**](https://nunchaku.tech/docs/ComfyUI-nunchaku/) is now live! Explore comprehensive guides and resources to help you get started.
- **[2025-06-29]** 🔥 **v0.3.3** now supports [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)! Download the quantized model from [Hugging Face](https://huggingface.co/nunchaku-tech/nunchaku-flux.1-kontext-dev) or [ModelScope](https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-kontext-dev) and use this [workflow](./example_workflows/nunchaku-flux.1-kontext-dev.json) to get started.

<details>
<summary>More</summary>

- **[2025-06-11]** Starting from **v0.3.2**, you can now **easily install or update the [Nunchaku](https://github.com/nunchaku-tech/nunchaku) wheel** using this [workflow](https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/install_wheel.json)!
- **[2025-06-07]** 🚀 **Release Patch v0.3.1!** We bring back **FB Cache** support and fix **4-bit text encoder loading**. PuLID nodes are now optional and won’t interfere with other nodes. We've also added a **NunchakuWheelInstaller** node to help you install the correct [Nunchaku](https://github.com/nunchaku-tech/nunchaku) wheel.
- **[2025-06-01]** 🚀 **Release v0.3.0!** This update adds support for multiple-batch inference, [**ControlNet-Union-Pro 2.0**](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0) and initial integration of [**PuLID**](https://github.com/ToTheBeginning/PuLID). You can now load Nunchaku FLUX models as a single file, and our upgraded [**4-bit T5 encoder**](https://huggingface.co/nunchaku-tech/nunchaku-t5) now matches **FP8 T5** in quality!
- **[2025-04-16]** 🎥 Released tutorial videos in both [**English**](https://youtu.be/YHAVe-oM7U8?si=cM9zaby_aEHiFXk0) and [**Chinese**](https://www.bilibili.com/video/BV1BTocYjEk5/?share_source=copy_web&vd_source=8926212fef622f25cc95380515ac74ee) to assist installation and usage.
- **[2025-04-09]** 📢 Published the [April roadmap](https://github.com/nunchaku-tech/nunchaku/issues/266) and an [FAQ](https://github.com/nunchaku-tech/nunchaku/discussions/262) to help the community get started and stay up to date with Nunchaku’s development.
- **[2025-04-05]** 🚀 **Release v0.2.0!** This release introduces [**multi-LoRA**](example_workflows/nunchaku-flux.1-dev.json) and [**ControlNet**](example_workflows/nunchaku-flux.1-dev-controlnet-union-pro.json) support, with enhanced performance using FP16 attention and First-Block Cache. We've also added [**20-series GPU**](examples/flux.1-dev-turing.py) compatibility and official workflows for [FLUX.1-redux](example_workflows/nunchaku-flux.1-redux-dev.json)!

</details>

## Getting Started

- [Installation Guide](https://nunchaku.tech/docs/ComfyUI-nunchaku/get_started/installation.html)
- [Usage Tutorial](https://nunchaku.tech/docs/ComfyUI-nunchaku/get_started/usage.html)
- [Example Workflows](https://nunchaku.tech/docs/ComfyUI-nunchaku/workflows/toc.html)
- [Node Reference](https://nunchaku.tech/docs/ComfyUI-nunchaku/nodes/toc.html)
- [API Reference](https://nunchaku.tech/docs/ComfyUI-nunchaku/api/toc.html)
- [Custom Model Quantization: DeepCompressor](https://github.com/mit-han-lab/deepcompressor)
- [Contribution Guide](https://nunchaku.tech/docs/ComfyUI-nunchaku/developer/contribution_guide.html)
- [Frequently Asked Questions](https://nunchaku.tech/docs/nunchaku/faq/faq.html)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=nunchaku-tech/ComfyUI-nunchaku&type=Date)](https://www.star-history.com/#nunchaku-tech/ComfyUI-nunchaku&Date)
