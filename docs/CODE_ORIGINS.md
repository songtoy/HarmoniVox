# 代码来源与 HarmoniVox 改动

推理代码同时包含 Bert-VITS2 来源与 HarmoniVox 自行编写的部分。
当前暂不选择代码或模型许可证；此文仅记录来源与实现关系。

## 来源记录

| 部分 | 可核实的来源或关系 |
| --- | --- |
| `models.py` 中语音合成器、`modules.py`、`attentions.py`、`text/` 等 | 多文件匹配或修改自 Bert-VITS2 v2.3；部分更早继承 VITS、PaddleSpeech 等 |
| `blip2_models/` 中四份保留的实现 | 保留 Salesforce 文件头，Qformer 还指向 Transformers BERT |
| `models_qformer_textual_multi.py` | HarmoniVox 的图像/文本/音频研究改动，结合 LAVIS/BLIP-2 结构并继承 `Blip2Base` |
| `models.py` 中 `AdversarialClassifier`、`MelStyleEncoder`、`ReferenceEncoderVAE` | 不存在于对照 v2.3 的同名文件，属于本地新增候选；更早来源待维护者补充 |
| `inference.py`、`release_assets.py`、发布工具、测试与文档 | 本轮为 HarmoniVox 编写的发布支持代码 |

逐文件和 Git 对照见 [THIRD_PARTY.md](THIRD_PARTY.md)。原始第三方版权头保留在源文件中，
README 的 [Acknowledgements](../readme.md#acknowledgements) 包含 Bert-VITS2 致谢。

## 推理连接关系

`inference.py` 调用 `infer_web_qformer_textual_multi.py`，后者构造
`models.SynthesizerTrn` 和 `models_qformer_textual_multi.Blip2Qformer`，
将图像编码器输出直接传给 `net_g.infer(..., g=image_feats)`。

图像编码器的 `from models import *` 引用 `AdversarialClassifier`，
该类使用 `modules.GradientReversal`。本轮只整理文档、来源说明和发布文件，
没有为许可证选择改变模型结构或推理计算。
