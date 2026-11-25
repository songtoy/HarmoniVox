# Hugging Face 文件准备、上传与下载

目标模型仓库为 `sonterzhuo/HarmoniVox`。本地现成上传目录在仓库同级
`outputs/harmonivox-huggingface/`，共 7 个文件：G/E、语音配置、Q-Former 配置、
模型卡、manifest 和 SHA256SUMS。权重合计约 4.18 GB，不含优化器状态、原始
CLIP/XLM-R 权重、训练/评测数据或本地凭据。

HF 只上传模型资源。代码在代码仓库中单独维护，不随模型上传；源码 ZIP 和任何额外
文件都会被上传目录检查拒绝。单独生成的本地源码包仍保留，便于代码审阅。

## 本地检查

在 HarmoniVox 根目录运行。此命令只依赖标准库，不联网、不上传：

```bash
python scripts/huggingface_release.py check --folder ../outputs/harmonivox-huggingface
python scripts/huggingface_release.py upload --repo-id sonterzhuo/HarmoniVox --folder ../outputs/harmonivox-huggingface --dry-run
```

检查覆盖文件边界、完整 SHA-256、最终模型 profile、精简推理权重身份、配置匹配，
以及 Q-Former 配置。源码 ZIP、额外文件、LFS 指针、同名消融权重、
损坏文件或包含优化器的原始权重均不能通过此上传检查。

## 重新生成上传目录

现成目录无需重复生成。需要重新打包时，使用已经验证过的推理导出产物，并指定
一个尚不存在的新目录；脚本不会覆盖已有目录：

```bash
python scripts/huggingface_release.py prepare \
  --export-dir ../outputs/harmonivox-release/harmonivox \
  --output-dir ../outputs/harmonivox-huggingface-new \
  --repo-id sonterzhuo/HarmoniVox
```

脚本从导出 manifest 取 G/E 和语音配置，将 `model/` 下的文件移到发布布局的根层，
加入 Q-Former 配置并生成模型卡，不打包或复制推理代码。大权重在同文件系统中
通过硬链接复用，跨文件系统才复制；生成后校验全部文件，再原子安装到输出目录。
不要修改发布目录中的权重文件；它们可能与已验证的导出文件共享硬链接。

## 执行上传

上传工具使用推理环境已固定的 `huggingface_hub==0.32.5`。在自己的终端登录，
或通过标准 `HF_TOKEN` 环境变量提供写入目标模型仓库的凭据；不要把 token 写进脚本、
命令参数、README 或提交到 Git：

```bash
python -c "from huggingface_hub import login; login()"
```

上传并创建一个新的私有模型仓库：

```bash
python scripts/huggingface_release.py upload \
  --repo-id sonterzhuo/HarmoniVox \
  --folder ../outputs/harmonivox-huggingface
```

如需直接创建新的公开仓库，增加 `--public`。该参数只影响新仓库；已存在仓库的可见性
保持原状。上传只提交这 7 个文件，不删除远端其他文件；同名文件会按本次产物更新。
工具用上传前的远端 commit 作为并发检查条件，遇到其他提交插入时由 Hub 拒绝该次提交。

上传成功后，工具在上传目录之外写入 `harmonivox-upload-<完整commit>.json`，
记录仓库、不可变 revision、提交链接、模型相关文件哈希。不要把回执放进模型上传目录。
本轮只执行本地准备和 dry-run，不调用实际上传。

## 固定版本下载与验证

用上传回执中的完整 40 位 commit SHA 替换 `FULL_COMMIT_SHA`，下载到尚不存在的目录：

```bash
python scripts/huggingface_release.py download \
  --repo-id sonterzhuo/HarmoniVox \
  --revision FULL_COMMIT_SHA \
  --output-dir ../checkpoint
```

下载只取该 revision 的发布文件；全部哈希和模型身份验证成功后才落入目标目录。
失败时不会留下可误用的半成品，也不会覆盖已有 checkpoint。目录外还会生成下载回执。
已有本地 HF 缓存时可加 `--local-files-only`。不接受 `main` 或短 SHA，避免下载版本漂移。

代码需要单独获取并按其 README 安装依赖。也可在 HarmoniVox 根目录用单独安装的
`hf` CLI 下载模型文件：

```bash
hf download sonterzhuo/HarmoniVox --revision FULL_COMMIT_SHA --local-dir ../checkpoint
python scripts/huggingface_release.py check --folder ../checkpoint
python scripts/prepare_assets.py --languages EN ZH
python inference.py --language EN --check-only --verify-weights
```

HF 远端完整上传/下载回环需在实际上传后验证；本轮不将本地模拟下载视为远端验证成功。


## 本地验证范围

当前 16 项自动检查通过，覆盖配置初始化、源码包、导出张量一致性、上传包身份与边界、
离线 dry-run、上传 API 参数/回执，以及固定 revision 下载的校验与失败清理。
其中上传 API 与下载传输使用本地替身，不发送远端写入请求。真实权重还会执行完整哈希检查，
并检查代码的默认模型路径；此前 EN/ZH 的 CPU/CUDA 回归记录保留在安装与 CUDA 文档中。
