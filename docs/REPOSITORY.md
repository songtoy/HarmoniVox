# 发布仓库范围

本仓库从整理后的推理源码建立独立 Git 历史，主分支为 `main`。
原始 HarmoniVox 研究仓库及其提交、分支和备份单独保留，不导入本仓库。
代码来源记录中的旧提交号用于溯源，不代表这些提交属于本仓库的历史。

## 源码边界

`scripts/package_release.py` 的 `FILES` 同时定义 Git 源码和源码 ZIP 的范围，
共 66 个文件。`python scripts/check_repository.py` 检查仓库是否符合此范围。
模型权重、训练数据、环境和运行产物均不进入 Git，也无需 Git LFS。

正式入口为 `inference.py`，调用 `infer_web_qformer_textual_multi.py`。
仓库包含推理模型与语言模块、资源准备和权重导出工具、溯源工具、测试、文档和 CI 配置。
训练、消融、批量评测、旧 WebUI 和无关的 vendored annotator 均不在本仓库内。
保留文件中的第三方版权声明继续保留，来源见 [CODE_ORIGINS.md](CODE_ORIGINS.md)。

首版验证范围为 EN/ZH；日文旧接口保留，但不在本次验证和支持承诺内。
CI 只检查源码和无需真实权重的测试，不下载或运行完整模型。
中英文 CUDA 端到端记录见 [CUDA_VALIDATION.md](CUDA_VALIDATION.md)。

## 模型与 GitHub 分开发布

Hugging Face 目标为 `sonterzhuo/HarmoniVox`，只发布权重、配置、模型卡与校验清单，
不上传代码；操作见 [HUGGINGFACE.md](HUGGINGFACE.md)。GitHub 只发布本仓库的源码。
本地准备完成不表示远端上传已完成，下载时应使用实际上传后取得的完整模型 revision。

本地发布仓库准备时未绑定远端。上传前应选定空的目标 GitHub 仓库，再添加 `origin`
并推送 `main`。原研究仓库的远端已有旧分支；向同一远端新增 `main` 不会移除旧历史，
不能据此声称只公开整理后的源码。不要使用 `--mirror`、`--all` 或强制覆盖来完成首次发布。

## 发布清单

代码和模型许可证暂未选择，第三方来源记录保留。
远端地址、模型 revision、可分发样例等剩余事项见 [RELEASE.md](RELEASE.md)。
