<div align="center" id="nunchaku_logo">
  <img src="https://raw.githubusercontent.com/nunchaku-tech/nunchaku/96615bd93a1f0d2cf98039fddecfec43ce34cc96/assets/nunchaku.svg" alt="logo" width="220"></img>
</div>
<h3 align="center">
<a href="http://arxiv.org/abs/2411.05007"><b>论文</b></a> | <a href="https://nunchaku.tech/docs/ComfyUI-nunchaku/"><b>文档</b></a> | <a href="https://hanlab.mit.edu/projects/svdquant"><b>官网</b></a> | <a href="https://hanlab.mit.edu/blog/svdquant"><b>博客</b></a> | <a href="https://svdquant.mit.edu"><b>演示</b></a> | <a href="https://huggingface.co/nunchaku-tech"><b>Hugging Face</b></a> | <a href="https://modelscope.cn/organization/nunchaku-tech"><b>魔搭社区</b></a>
</h3>

<h3 align="center">
<a href="README.md"><b>English</b></a> | <a href="README_ZH.md"><b>中文</b></a>
</h3>

本仓库为 [**Nunchaku**](https://github.com/nunchaku-tech/nunchaku) 提供了 ComfyUI 插件，Nunchaku 是一个高效的 4-bit 神经网络推理引擎，采用 [SVDQuant](http://arxiv.org/abs/2411.05007) 量化方法。量化库请参考 [DeepCompressor](https://github.com/nunchaku-tech/deepcompressor)。

欢迎加入我们的用户群：[**Slack**](https://join.slack.com/t/nunchaku/shared_invite/zt-3170agzoz-NgZzWaTrEj~n2KEV3Hpl5Q)、[**Discord**](https://discord.gg/Wk6PnwX9Sm) 和 [**微信**](https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/nunchaku/assets/wechat.jpg)（详情见[这里](https://github.com/nunchaku-tech/nunchaku/issues/149)）。如有任何问题、遇到 bug 或有意贡献代码，欢迎随时与我们交流！

# Nunchaku ComfyUI 插件

![comfyui](https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/comfyui.jpg)

## 最新动态

- **[2025-08-22]** 🚀 **v1.0.0** 新增对[Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)的支持！查看[示例工作流](example_workflows/nunchaku-qwen-image.json)即可快速上手。LoRA支持即将推出。
- **[2025-07-17]** 🚀 [**ComfyUI-nunchaku 官方文档**](https://nunchaku.tech/docs/ComfyUI-nunchaku/)上线！提供详细的入门指南和资源。
- **[2025-06-29]** 📘 **v0.3.3** 现已支持 [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)！可从 [Hugging Face](https://huggingface.co/nunchaku-tech/nunchaku-flux.1-kontext-dev) 或 [魔搭社区](https://modelscope.cn/models/nunchaku-tech/nunchaku-flux.1-kontext-dev) 下载量化模型，并参考此 [工作流](./example_workflows/nunchaku-flux.1-kontext-dev.json) 快速上手。
- **[2025-06-11]** 自 **v0.3.2** 起，您可以通过此 [工作流](https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/install_wheel.json) **轻松安装或升级 [Nunchaku](https://github.com/nunchaku-tech/nunchaku) wheel 包**！
- **[2025-06-07]** 🚀 **v0.3.1 补丁发布！** 恢复了 **FB Cache** 支持，修复了 **4-bit 文本编码器加载**问题。PuLID 节点现为可选，不会影响其他节点。新增 **NunchakuWheelInstaller** 节点，帮助您安装正确的 [Nunchaku](https://github.com/nunchaku-tech/nunchaku) wheel。

<details>
<summary>更多动态</summary>

- **[2025-06-01]** 🚀 **v0.3.0 发布！** 新增多 batch 推理、[**ControlNet-Union-Pro 2.0**](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0) 支持，以及初步集成 [**PuLID**](https://github.com/ToTheBeginning/PuLID)。现可将 Nunchaku FLUX 模型作为单文件加载，升级后的 [**4-bit T5 编码器**](https://huggingface.co/nunchaku-tech/nunchaku-t5) 质量媲美 **FP8 T5**！
- **[2025-04-16]** 🎥 发布了 [**英文**](https://youtu.be/YHAVe-oM7U8?si=cM9zaby_aEHiFXk0) 和 [**中文**](https://www.bilibili.com/video/BV1BTocYjEk5/?share_source=copy_web&vd_source=8926212fef622f25cc95380515ac74ee) 教学视频，助力安装与使用。
- **[2025-04-09]** 📢 发布了 [四月路线图](https://github.com/nunchaku-tech/nunchaku/issues/266) 和 [FAQ](https://github.com/nunchaku-tech/nunchaku/discussions/262)，帮助社区用户快速上手并了解最新进展。
- **[2025-04-05]** 🚀 **v0.2.0 发布！** 支持 [**多 LoRA**](example_workflows/nunchaku-flux.1-dev.json) 和 [**ControlNet**](example_workflows/nunchaku-flux.1-dev-controlnet-union-pro.json)，FP16 attention 与 First-Block Cache 性能增强。新增 [**20 系显卡**](examples/flux.1-dev-turing.py) 兼容性，并提供 [FLUX.1-redux](example_workflows/nunchaku-flux.1-redux-dev.json) 官方工作流！

</details>

## 快速上手

- [安装指南](https://nunchaku.tech/docs/ComfyUI-nunchaku/get_started/installation.html)
- [使用教程](https://nunchaku.tech/docs/ComfyUI-nunchaku/get_started/usage.html)
- [示例工作流](https://nunchaku.tech/docs/ComfyUI-nunchaku/workflows/toc.html)
- [节点参考](https://nunchaku.tech/docs/ComfyUI-nunchaku/nodes/toc.html)
- [API 参考](https://nunchaku.tech/docs/ComfyUI-nunchaku/api/toc.html)
- [自定义模型量化：DeepCompressor](https://github.com/nunchaku-tech/deepcompressor)
- [贡献指南](https://nunchaku.tech/docs/ComfyUI-nunchaku/developer/contribution_guide.html)
- [常见问题 FAQ](https://nunchaku.tech/docs/nunchaku/faq/faq.html)
