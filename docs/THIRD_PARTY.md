# Third-party inventory — incomplete

This inventory records code headers, local model-card metadata and a Git/source
audit of Bert-VITS2 v2.3 performed on 2026-11-25.
The project and model licenses are not selected. Newly prepared license files
and NOTICE have been withdrawn at the maintainer's request. Existing copyright
headers remain in the source. This document preserves source evidence rather
than declaring a project license. Original contributions are recorded in
[CODE_ORIGINS.md](CODE_ORIGINS.md).

| Component | Evidence / upstream | Remaining work |
| --- | --- | --- |
| `blip2_models/Qformer.py`, `blip2.py`, `base_model.py`, `common/dist_utils.py` | Salesforce copyright and BSD-3-Clause identifiers in file headers; [LAVIS](https://github.com/salesforce/LAVIS) | Retain source headers; original import revision still needs identification |
| Q-Former's Hugging Face-derived code | `Qformer.py` header points to Transformers v4.15.0 BERT | Retain source attributions; review original notices and changes |
| Speech model and text frontends | Source fingerprints match [Bert-VITS2 v2.3](https://github.com/fishaudio/Bert-VITS2/tree/76653b5b6d657143721df2ed6c5c246b4b1d9277), whose license is AGPLv3; see the audit below | Retain upstream source evidence; project license decision deferred |
| CLIP ViT-H/14 training/base provenance (not a separate inference download) | [LAION model card](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K); local card says MIT | Record downloaded revision, full terms and retained notices |
| XLM-RoBERTa training/base provenance (not a separate inference download) | [FacebookAI model repository](https://huggingface.co/FacebookAI/xlm-roberta-base) | Record revision and license text |
| English DeBERTa | [Microsoft model repository](https://huggingface.co/microsoft/deberta-v3-large); local card says MIT | Record revision and retained notices |
| Chinese RoBERTa | [HFL model repository](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large); local card says Apache-2.0 | Record revision and retained notices |
| Japanese DeBERTa | [Kyoto University model repository](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm); local card says CC-BY-SA-4.0 | Review separately before including or redistributing Japanese model assets |
| `text/cmudict.rep` | File header identifies CMU dictionary 0.6 (1998), permits research/commercial use and requests acknowledgement; byte-identical to Bert-VITS2 v2.3 | Preserve the complete embedded notice; keep separate from current downloaded NLTK CMU resources |
| `text/opencpop-strict.txt` | Byte-identical to Bert-VITS2 v2.3 | Trace the original pronunciation mapping and preserve applicable notices |

The inference working tree and source ZIP exclude `blip2_models/common/annotator/`,
vendored MMCV/MMseg and unrelated research entry points. They were backed up and
archived outside this repository. They remain only in the original research
repository's history, which is not imported into this publication repository.
See [repository scope](REPOSITORY.md).

The trained E checkpoint includes a visual backbone and Q-Former parameters.
Their original checkpoint files are no longer downloaded for inference; E is
loaded into architectures constructed from local configurations.
Publishing it needs a separate review even when original base-model downloads
are supplied by links instead of being bundled.


## Bert-VITS2 来源与许可证溯源（2026-11-25）

### 本地 Git 能证明的内容

- `Image2Voice/Bert-VITS2-master` 的 remote 实际指向 HarmoniVox，含两个提交：
  `f474515ccc9e1136fa2ca091d37e170fece96935`（2025-06-12 初始提交）和
  `657d44e6584dc817d98c4339f635558ab8824165`（2025-09-21）。
- 整理所用的原始 HarmoniVox 仓库继承这两个提交，再加 `c6728ad7569286ecdfde0b2ecfc98159978b17d7`。
  本发布仓库从整理后的源码建立新历史，不包含上述研究提交；这些提交号仅用于记录来源。
- 两处现存 refs、reflog/对象检查没有提供更早的 Bert-VITS2 父提交；许可证路径历史为空。
  初始提交是导入研究代码后的新根节点，不能据此认定研究代码全部原创或无上游约束。
- 因而不能恢复当时实际下载/复制的唯一上游 commit；下面通过内容指纹确定版本对应关系。

### 与公开上游 v2.3 的逐文件对应

对照仓库：[fishaudio/Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)。
对照标签 `v2.3` 对应不可变提交
[`76653b5b6d657143721df2ed6c5c246b4b1d9277`](https://github.com/fishaudio/Bert-VITS2/tree/76653b5b6d657143721df2ed6c5c246b4b1d9277)，
提交时间为 2023-12-19。选取 18 个当前推理相关文件，分别对照 Image2Voice 现存版本、
HarmoniVox 初始提交及当前发布工作区。这里的相同指完整字节相同，非仅文件名相同。

| 证据 | 结果 |
| --- | --- |
| `commons.py`、`transforms.py`、`mel_processing.py` | 三份本地版本均与 v2.3 完全相同 |
| `text/symbols.py`、`text/cmudict.rep`、`text/opencpop-strict.txt` | 三份本地版本均与 v2.3 完全相同 |
| `monotonic_align/__init__.py`、`monotonic_align/core.py` | 三份本地版本均与 v2.3 完全相同 |
| `text/english_bert_mock.py`、`text/chinese_bert.py` | Image2Voice 和初始提交均与 v2.3 完全相同；发布版本只在此基础上做路径/加载调整 |
| `models.py`、`modules.py`、`attentions.py` 等 | 属于修改版本；原始行序列相似度分别约 87.36%、94.98%、99.57%，仅作为辅助证据 |

共 10 个初始/研究文件与 v2.3 字节完全一致，其中 8 个当前仍完全一致。
尤其两份 BERT 前端在上游最后修改于 v2.3 提交本身；它们与 AGPL 改动前的文件不同。
这支持使用了 AGPL 生效后的 Bert-VITS2 代码，不能仅因更早版本用 MIT 就将现有代码整体改标 MIT。
行序列相似度不是版权比例，也不能代替逐组件许可判断。

### 当时的 LICENSE，而非仅当前 master

直接读取公开上游 Git 历史得到：

| 提交时间 | 提交 | LICENSE 内容 |
| --- | --- | --- |
| 2023-07-21 | `ae26c0d65788e83f517d60820c91c2c31fbecc73` | MIT |
| 2023-08-31 | `1db84f5bcfa22380cbd880359373459354b0aa01` | GPLv3 |
| 2023-09-02 | `d57271a32c305916390bdc590e020bd4eb6c4527` | AGPLv3，提交说明为“更改为AGPL-V3” |
| 2023-12-19 | `76653b5b6d657143721df2ed6c5c246b4b1d9277`（v2.3） | AGPLv3 |

[v2.3 的原始 LICENSE](https://github.com/fishaudio/Bert-VITS2/blob/76653b5b6d657143721df2ed6c5c246b4b1d9277/LICENSE)
SHA-256 为 `8486a10c4393cee1c25392769ddd3b2d6c242d6ec7928e1414efff7dfb2f07ef`。
以上记录上游当时的许可内容；当前项目暂不选择代码或模型许可证。
原始来源证据保存在仓库外审计目录，未改写参考代码或历史。

### 审计原始记录

仓库外 `outputs/harmonivox-license-audit/` 保存：

- `git-provenance.json`：本地提交、上游标签、许可证历史与 SHA-256。
- `bert-vits2-v23-comparison.json`：18 个文件的三组逐字节/行序列对照。
- 两份 BERT 前端的 v2.3 源文件及 AGPL 改动前后差异。
- 上游 v2.3 LICENSE、README 的原始内容。

审计只读取 Image2Voice 和原始 Git 历史，没有更改参考代码或原仓库历史。
