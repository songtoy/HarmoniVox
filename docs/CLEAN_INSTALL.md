# 独立环境安装与端到端复现记录

验证日期：2026-11-25。平台：Linux x86_64。

## 结果

从全新 Python venv 安装依赖、解压推理源码包后，直接使用默认路径完成中英文 CPU 推理。

| 项目 | 英文 | 中文 |
| --- | ---: | ---: |
| 采样率 | 16,000 Hz | 16,000 Hz |
| 输出采样点数 | 34,048 | 40,192 |
| 音频时长 | 2.128 秒 | 2.512 秒 |
| 与原环境 PCM 比较 | 完全相同 | 完全相同 |
| 最大 PCM 差值 | 0 | 0 |

11 项自动检查通过，`pip check` 通过。完整 CPU 依赖锁包含 71 个发行包版本，
已核对它与实际安装环境一致。

## 配置初始化回归（后续优化）

当前推理已移除原始 CLIP/XLM-RoBERTa 权重和 XLM-R tokenizer 的读取、下载及预检依赖。
CLIP 使用固定 OpenCLIP 版本附带的 `ViT-H-14` 结构；Q-Former 使用源码中随包提供的
`configs/qformer.json`。后者保留原 XLM-R 配置的结构字段，再由代码设置视觉交叉注意力，
严格加载 E 的全部 736 项 state dict；G 的全部 835 项也继续严格加载。
原始 CLIP 3,944,692,325 字节和 XLM-R 1,115,590,446 字节共 5,060,282,771 字节
不再下载。G/E 文件、模型精度、图像预处理和推理计算均未改变。

在独立解压目录运行正式 `inference.py`，只放 G/E、英文/中文 BERT 和 NLTK 数据。
CLIP/XLM-R 目录均不存在；Hugging Face 缓存为空、离线模式开启，且显式拦截
原始 CLIP 权重加载器和 `XLMRobertaModel.from_pretrained`，一旦调用即报错。
实际 PyTorch 权重读取记录只有 G、E 和当前语言的 BERT。

| 路径 | 采样点数 | 与优化前同环境的 PCM 比较 | 最大差值 |
| --- | ---: | --- | ---: |
| CPU EN | 34,048 | 完全相同 | 0 |
| CPU ZH | 40,192 | 完全相同 | 0 |
| CUDA EN | 34,048 | 完全相同 | 0 |
| CUDA ZH | 40,192 | 完全相同 | 0 |

以上比较固定 seed=42、相同图片/短句、相同设备类别和依赖版本；没有跨 CPU/GPU 逐点一致的承诺。
语言 BERT 仍用于实际文本前向计算，不能由 G/E 或结构配置替代。英文还需 NLTK 资源。
资源准备脚本在这个无 CLIP/XLM-R 的目录成功验证了 13 个文件，新增小型配置构建测试后
共 12 项自动检查通过。既有模型资产文件未删除，参考研究目录未修改。
原始日志、加载记录、WAV 和 JSON 在仓库外 `outputs/harmonivox-config-init-validation/`。

## 首次安装验证的隔离条件

- 使用 Python 3.9.21 创建 venv，未启用系统 site-packages，禁用用户 site-packages。
- 新环境创建后确认不存在 torch、transformers、numpy、open_clip，再从官方索引安装。
- PyTorch、Transformers、OpenCLIP、librosa 的模块路径均位于新 venv 中。
- 从 ZIP 解压出独立源码目录，不带工作区的 `cmudict_cache.pickle` 或 Python 缓存。
- 使用空的 Hugging Face 缓存，运行时设置 Hub/Transformers 离线模式；运行进程的沙箱也禁止联网。
- 英文 NLTK 路径被限定为准备好的 `pretrained_models/nltk_data/`，没有使用全局 NLTK 资源。
- 测试工作区按 README 布局放置 `HarmoniVox/`、`checkpoint/`、`pretrained_models/`；
  未传入模型目录、基础模型目录、CLIP 或 XLM-RoBERTa 的路径覆盖参数。

模型是输入数据。本轮复用了已经定位的 MM25 G/E 文件，以及校验过的本地基础模型；
测试工作区通过目录链接复用这些大文件。模型身份和基础模型文件内容都经过校验。

## 安装版本

| 组件 | 验证版本 |
| --- | --- |
| Python | 3.9.21 |
| PyTorch / torchaudio / torchvision | 2.3.0+cpu / 2.3.0+cpu / 0.18.0+cpu |
| Transformers | 4.44.2 |
| OpenCLIP | 2.32.0 |
| NumPy | 1.26.3 |
| librosa | 0.9.1 |
| NLTK | 3.8.1 |
| setuptools | 69.5.1 |

`requirements-inference.txt` 保留主要依赖，`requirements-lock-cpu.txt` 记录这次安装的完整版本。
CLI 不依赖 Gradio 或语言自动检测库；它们仍属于原研究/旧 WebUI 的依赖范围。

## 首次安装验证中的修正（历史记录）

1. 默认模型目录统一为仓库同级的 `checkpoint/`，并支持 `HARMONIVOX_CHECKPOINT_DIR`。
   发布模型依然是 `paper-proposed`，权重张量未变化。
2. CLIP、XLM-RoBERTa 与 BERT 统一放在同级 `pretrained_models/`，保留自定义路径覆盖能力。
3. 增加 `scripts/prepare_assets.py` 和 `configs/assets.json`，固定基础模型提交版本及文件校验值。
   支持下载或从已有模型目录复制；遇到内容不同的现有文件会报错。
4. 英文 tagger 与 CMU 词典固定在 NLTK 数据仓库的不可变提交，并校验 ZIP 的 SHA-256。
   文本资源提前准备，避免首条推理依赖隐式下载。
5. 固定兼容的 setuptools，确保 librosa 0.9.1 使用的 `pkg_resources` 可用。
6. 本机 Hugging Face Hub 的 HEAD 元数据下载接口返回错误。准备脚本采用固定提交的文件 URL
   流式下载，并校验文件内容。实际下载并校验了一个 tokenizer 配置和两个完整 NLTK ZIP；
   四个大权重 URL 均以 Range 请求成功读取了 1,024 字节。完整大权重通过本地副本校验，未重复联网下载。

## 按 README 复现

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.0 setuptools==69.5.1 wheel==0.45.1
python -m pip install torch==2.3.0 torchaudio==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock-cpu.txt

# 将 MM25 的 G_12000.pth / E_12000.pth 放入 ../checkpoint/
python scripts/prepare_assets.py --languages EN ZH
python inference.py --language EN --check-only --verify-weights

python inference.py --image /path/to/prompt.jpg --text "Hello, this is HarmoniVox." --language EN --device cpu --output outputs/en.wav
python inference.py --image /path/to/prompt.jpg --text "你好，欢迎使用语音合成。" --language ZH --device cpu --output outputs/zh.wav
```

可选 `--local-source /path/to/model-collection` 复用已下载的基础模型。该参数可重复指定，
每个目录包含 `bert/` 目录。当前版本不需要 CLIP/XLM-R 文件夹；模型检查点独立准备。

## 证据与未验证项

本机原始记录保存在工作区 `outputs/harmonivox-clean-validation/`：安装日志、依赖锁检查、
基础模型准备日志、上游元数据、下载探测记录、推理日志、两条 WAV 和 `results.json`。
机器路径和测试图片/音频不进入发行包。

后续 CUDA 硬件验证见 [CUDA_VALIDATION.md](CUDA_VALIDATION.md)。日文按维护者要求不验证，
其他操作系统和 Python 版本尚未验证。HarmoniVox 权重的
公开下载地址和许可证仍待确定。此处记录软件环境与推理复现，不替代论文指标和主观听感评测。
