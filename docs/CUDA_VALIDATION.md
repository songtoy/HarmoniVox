# CUDA 中英文推理验证

验证日期：2026-11-25。正式入口为 `inference.py`，模型为已确认的 `paper-proposed`，
显式加载 `G_12000.pth` / `E_12000.pth` 并验证完整 SHA-256。

## 环境与隔离

- Linux x86_64，Python 3.9.21 独立 venv，无系统或用户 site-packages。
- NVIDIA RTX A6000（48 GB），驱动 570.86.10。
- PyTorch / torchaudio 2.3.0+cu118，torchvision 0.18.0+cu118；CUDA runtime 11.8。
- Transformers 4.44.2、OpenCLIP 2.32.0，完整版本见 `requirements-lock-cu118.txt`。
- 从官方 PyTorch cu118 索引及 PyPI 安装；`pip check` 通过。
- 从源码 ZIP 解压独立目录；默认同级 checkpoint/pretrained_models 目录通过链接复用已校验资源。
- 空 Hugging Face 缓存、Hub/Transformers 离线模式；英文只使用已准备的 NLTK 数据。
- 选择空闲 GPU，测试包装器检查 E/G 的全部参数确实位于 CUDA 设备；使用正式 CLI 的参数和逻辑。
- seed=42、CPU 线程数为 4，默认 noise-scale=0.6、length-scale=1.0。

普通沙箱无法访问 GPU；允许设备访问后 CUDA 初始化、张量运算及完整推理均成功。
无需修改驱动或 CUDA 服务。日文按维护者要求不验证，首版验证范围为 EN/ZH。

## 结果

中英文均成功生成单声道 PCM-16、16 kHz WAV，样本非空、数值有限且非静音。
英文短句为 `Hello, this is HarmoniVox.`，中文为 `你好，欢迎使用语音合成。`。
测试图片为已有本地评测图片，不随发行包分发。

英文输出 34,048 点（2.128 秒），中文输出 40,192 点（2.512 秒）。

| 项目 | 英文 | 中文 |
| --- | ---: | ---: |
| 音频时长（秒） | 2.128 | 2.512 |
| 含模型加载/校验的耗时（秒） | 22.87 | 20.53 |
| 峰值已分配显存（GiB） | 5.72 | 5.31 |
| 峰值预留显存（GiB） | 5.85 | 5.44 |

中英文分别在独立进程运行，避免跨语言缓存影响显存。计时包含权重身份校验、模型构建、
权重加载和合成，不包含依赖安装与测试包装器预先导入的模块；不是纯合成速度。
本次使用原始 G/E 文件，不应据此认定最低显存要求或长文本性能。
同 GPU、同 seed 重复运行，两条音频各自的 PCM 均完全一致。

这是短句执行回归，不替代听感评测或论文指标复现。GPU 与 CPU 的 PCM 不承诺逐点相同，
跨设备/依赖版本也不承诺逐位一致。

## 移除冗余原始权重后的回归

前面的耗时表是原加载器的基线。当前版本已按本地配置构建 CLIP/Q-Former 并严格加载 E，
不再读取原始 CLIP/XLM-R 权重或 XLM-R tokenizer。无这些目录且显式阻止原权重加载时，
CUDA 英文/中文输出分别与上述基线 PCM 完全一致，最大差值均为 0。
详细隔离条件及 CPU 对照见 [配置初始化回归](CLEAN_INSTALL.md#配置初始化回归后续优化)。
日志位于仓库外 `outputs/harmonivox-config-init-validation/`。

## 复现

按 README 创建独立环境，将 CPU 依赖安装步骤换成：

```bash
python -m pip install torch==2.3.0 torchaudio==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-lock-cu118.txt
python -m pip check
python inference.py --image /path/to/prompt.jpg --text "Hello, this is HarmoniVox." --language EN --device cuda --output outputs/en-cuda.wav
python inference.py --image /path/to/prompt.jpg --text "你好，欢迎使用语音合成。" --language ZH --device cuda --output outputs/zh-cuda.wav
```

原始硬件探测、安装日志、推理日志、WAV 和结果 JSON 保存在仓库外的
`outputs/harmonivox-cuda-validation/`，不进入源码包。
