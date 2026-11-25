# 推理代码与模型发布清单

本轮范围已确定为推理代码和模型权重。代码和模型许可证暂不选择；
此处记录本地候选产物及发布前缺项。
源码整理自原始仓库的 `release_prep` 工作区，起始提交为 `c6728ad`；
本发布仓库从整理后的 66 个文件建立独立历史，不导入原研究提交。
原始研究目录只用于参考，不修改或移动其中的文件。

## 发布边界

- 主线：`qformer_textual_multi`，与研究目录路线图的 proposed 路线一致。
- 候选权重：当前工作区的 `G_12000.pth` / `E_12000.pth`。
- 对外入口：`inference.py`，单张图片、文本、显式语言，输出 WAV。
- 模型加载：匹配同一步数的 G/E 文件；不完整或不兼容的 state dict 直接报错。
- 训练、消融、数据预处理、批量评测和旧 WebUI 入口不进入推理源码 ZIP。
- 研究脚本与无关第三方目录在完整备份后移出工作区；原有删除记录和 WebUI 修改内容保存在备份中。
- Git 工作区和源码 ZIP 使用同一文件边界；原始研究历史单独保留。见 [仓库范围](REPOSITORY.md)。
- 首版验证范围为 EN/ZH；日文按维护者要求不验证，不作支持承诺。

## 本地整理结果

- [x] 补齐安装、资源目录、命令行推理和模型来源文档。
- [x] 主推理链路使用可配置路径，不再写死开发者机器路径。
- [x] 按所选语言加载文本处理器和 BERT，避免英文推理强制要求日文权重。
- [x] 增加资源预检，识别缺失文件、空文件和 Git LFS 指针。
- [x] 独立严格加载推理权重，避免训练加载器静默补入随机参数。
- [x] 导出推理模型，保留精度、移除优化器状态，并逐张量验证一致性。
- [x] 导出配置移除训练语料路径、原始说话人名单及 WavLM 配置。
- [x] 权重清单记录源文件与导出文件的 SHA-256、体积、张量数量。
- [x] 通过显式文件列表生成推理源码包，排除无关 vendored annotator。
- [x] 比较研究目录全部候选 G/E 哈希，确认当前权重与 `paper_used/proposed` 完全相同。
- [x] 对照原 WebUI backend，确认公开 CLI 的推理路径与英文 PCM 一致。
- [x] 增加模型身份 profile、完整 SHA-256 校验及显式步数，防止误选同名消融权重。
- [x] 移除当前模型实际不生效的 SDP ratio / duration noise 控制项。
- [x] 默认目录统一到 `../checkpoint/`，基础模型与文本资源统一到 `../pretrained_models/`。
- [x] 新增语言 BERT/文本资源准备脚本，固定上游提交及文件校验值。
- [x] CLIP/Q-Former 按本地配置构建并严格加载 E，移除原始 CLIP/XLM-R 权重和 XLM-R tokenizer 的下载、初始化与预检依赖。
- [x] 独立 venv、解压源码包、默认路径与离线条件下完成中英文 CPU 推理；见 [复现记录](CLEAN_INSTALL.md)。
- [x] 为 `sonterzhuo/HarmoniVox` 准备七文件 HF 发布目录，仅含模型权重、配置、模型卡与校验清单，不上传代码或源码 ZIP。
- [x] 提供 prepare/check/upload/download 工具、离线 dry-run、全量哈希检查、完整 revision 下载与上传回执；见 [HF 操作说明](HUGGINGFACE.md)。

## 复现导出

在仓库根目录执行。输出目录必须尚不存在，脚本不会覆盖已有产物。

```bash
python scripts/export_inference.py \
  --model-dir ../checkpoint \
  --config ../checkpoint/config.json \
  --output-dir ../outputs/harmonivox-release/harmonivox

python scripts/package_release.py \
  --output ../outputs/harmonivox-release/HarmoniVox-inference.zip

python -m unittest discover -s tests -v
```

模型导出需要 PyTorch；源码打包、基础模型准备和资源预检只依赖 Python 标准库。导出工具
保留 `model/` 子目录，测试该导出产物时显式传入 `--model-dir <导出目录>/model`。
默认安装的两份权重直接放在 `../checkpoint/`，基础模型通过 README 的准备脚本单独准备。

## 验证记录

使用本机已有的 `harmonivox` 环境验证：Python 3.9.21、PyTorch 2.3.0+cu118、
Transformers 4.44.2、OpenCLIP 2.32.0。早期验证以 CPU 执行；后续已完成独立 CUDA 环境验证。

- 原始权重英文 CLI：成功生成 16 kHz 音频。
- 权重导出：G/E 共 1,571 个张量逐一核对，数值完全一致。
- 原始权重总计 5,487,754,566 字节；导出权重总计 4,179,749,092 字节，
  减少 1,308,005,474 字节（约 23.8%）。
- 精简权重英文 CLI：生成 2.128 秒、16 kHz 音频，与原始权重输出的 PCM 数据完全一致。
- 精简权重中文 CLI：生成 2.512 秒、16 kHz 音频，非空、非静音且数值有限。
- 自动检查覆盖资源预检、检查点配对、严格加载、导出一致性、源码包边界和同名消融权重拒绝。
- 源码包：解压到临时目录，从仓库之外运行英文推理；未携带发音词典缓存，
  输出仍与工作区版本的 PCM 数据完全一致。
- 发布文件列表的静态检查未发现遗漏的本地模块导入或机器专属路径字面量。
- 现有环境 `pip check` 通过。源码包生成时保留已有文件的换行格式；
  `git -c core.whitespace=cr-at-eol diff --check` 通过。
- 测试音频、日志和机器相关路径保存在工作区的 `outputs/` 中，不进入发行包。

后续已完成独立环境安装验证：71 个发行包版本锁定，原有 11 项自动检查通过；新环境中英文
PCM 与原环境完全一致。独立 CUDA 环境的中英文推理、参数设备与显存记录见
[CUDA 验证](CUDA_VALIDATION.md)。日文不在本次验证范围，论文指标另见结果溯源记录。
后续配置初始化优化在无 CLIP/XLM-R 资源目录、原权重加载器被阻止时完成 CPU/CUDA
中英文四路回归，PCM 与各自旧基线完全一致，配置初始化回归时共 12 项自动检查通过；见安装记录。
本次新增 HF 准备/上传/固定版本下载工具后共 16 项本地自动检查通过，远端写入与下载回环尚未执行。

## 正式公开之前

- [x] 结合 MM25 论文 Table 2/3 的九项分类指标，定位发表结果对应 `paper-proposed`，排除后续 `experimental-proposed-1e-6` 实验；见 [结果溯源](MM25_FINAL_MODEL.md)。
- [x] 结合 Image2Voice、初始提交和公开上游 v2.3 完成代码许可溯源：18 文件对照，确认继承 AGPL 生效后的代码；见 [审计](THIRD_PARTY.md)。
- [x] 按维护者最新要求撤下本轮新增的许可文本、项目许可草案和 NOTICE；保留原有第三方版权头。
- [x] README 加入 Bert-VITS2 致谢，保留代码来源记录。
- [ ] 后续确定代码和模型许可证，再补相应许可文件及所需第三方文本。
- [ ] 按 [THIRD_PARTY.md](THIRD_PARTY.md) 核实剩余导入版本、发音映射和基础模型条款。
- [ ] 补充训练数据来源、版本、授权与模型再分发条件，完成模型卡。
- [x] 添加 ACM MM 2025 论文引用 `CITATION.cff`。
- [x] 确定 HF 目标 `sonterzhuo/HarmoniVox` 并补齐下载命令。
- [ ] 实际上传后记录远端完整 commit revision、完成远端下载回环，再确定 release 版本。
- [x] 在独立环境安装推理依赖并验证 CPU 中英文推理，记录完整依赖锁定结果。
- [x] 在独立 CUDA 11.8 环境完成中英文回归，确认 GPU 参数位置，记录显存与含加载耗时。
- [x] 明确首版验证范围为中英文；日文按维护者要求跳过。
- [ ] 选取允许公开分发的图片和音频样例，进行听感确认。
- [ ] 对源码包和权重清单复核后再创建 tag、上传模型和发布。

本发布仓库只导入整理后的推理文件，源码 ZIP 采用相同文件列表；原研究提交不进入发布历史。
已添加 CPU 源码 CI 和仓库边界检查。远程 CI、正式 tag 和远端上传仍需在实际发布时确认。
