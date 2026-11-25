# MM25 最终发表模型：依据实验结果溯源

核查日期：2026-11-25。

## 结论

**与 MM25 发表论文 Proposed 行对应的发布候选是 `paper_used/proposed` 下的 G/E 12000-step 配对。当前工作区已拷贝的就是这套权重。**

```text
data/emotional/paper_used/proposed/
  models_dp_bn_gan_pt_5e-5_norm_330k_ft_5e-6_qformer_textual_iam_iac_stage1_wo_ia_add_ia_stage2_unift/
    G_12000.pth
    E_12000.pth
```

较新的 `stage1_anneal_1e-6_200_stage2_unift` 是另一套完整实验，不能因为最后的配置或脚本名含 MM25，就认定它是最终发表模型。上一轮基于这些配置推荐较新实验的判断，现由以下数值证据纠正。

默认 profile 继续使用 `paper-proposed`；较新实验改名为 `experimental-proposed-1e-6`。本轮没有替换模型张量。

## 发表版本与比对依据

核查的是作者所在实验室主页提供、带 MM ’25 页眉和 DOI `10.1145/3746027.3755736` 的论文 PDF：
[HarmoniVox: Painting Voices to Match the Avatar’s Soul](https://hcsi.cs.tsinghua.edu.cn/Paper/paper25/MM2025_Songtao%20Zhou_HarmoniVox_Painting%20Voices%20to%20Match%20the%20Avatar%E2%80%99s%20Soul.pdf)。

本次下载文件共 10 页，PDF 元数据记录生成于 2025-07-31、修改于 2025-09-03。Table 2 在第 7 页，Table 3 在第 8 页；已核对页面图像，确认文本提取的表格列对应关系。

## 九项结果指纹

下表百分数为现存分类标签的统计重算，均按原分析代码的口径，保留两位小数。

| 范围 | 指标 | MM25 Table 2 | paper_used 对应结果 | 后续 1e-6 完整实验 |
| --- | --- | ---: | ---: | ---: |
| 域内 | Age | 56.99 | 56.99 | 59.14 |
| 域内 | Gender | 97.85 | 97.85 | 97.85 |
| 域内 | Pitch | 65.59 | 65.59 | 54.84 |
| 域内 | Energy | 65.59 | 65.59 | 66.67 |
| 域内 | Emotion | 58.06 | 58.06 | 55.91 |
| 域内 | Speed | 96.77 | 96.77 | 96.77 |
| 域外 | Age | 23.81 | 23.81 | 30.54 |
| 域外 | Gender | 90.64 | 90.64 | 89.16 |
| 域外 | Emotion | 40.39 | 40.39 | 41.05 |

`paper_used` 对应的结果在九项上全部吻合。较新 1e-6 实验只有 Gender-I 和 Speed-I 两项相同，其他七项不同。Table 3 的 HarAvaSpeech 行重复了相同的域外三项，也与旧结果吻合。

为排除错误口径恰好匹配，另外重算了 Baseline、Proposed w/o descs 的域内/域外行，以及 MEAD-TTS 的域外行；这 21 个对照数值也全部与发表表格一致。这不是按性能高低选 checkpoint，而是识别已经发表的结果来自哪一组实验。

## 结果文件与计算口径

域内采用：

```text
Evaluation/FINAL_eval_scps/
  eval_emotional_dp_bn_gan_norm_330_qformer_textual_iam_iac_stage1_wo_ia_add_ia_stage2_unift_12k.speed.scp
```

- 真实标签来自 `testset_ground_truth_speed.scp`，共 866 条。
- 按原 `analysis.py` 使用 `testset_filtered_all_acc.json` 中的 93 个样本。
- 每个样本包含 `_False` / `_True` 两次生成；某属性只要其中一次预测正确即计为正确。
- 本次既按样本键关联，也独立重放原脚本的顺序配对，两种统计完全一致。
- 后续实验使用 `rebuttal_results/intra-domain_quant_proposed_1e-6.scp`。该文件有 1,732 条生成标签和 866 条 ground-truth 标签；重算时明确排除后者，避免混算。

域外采用：

```text
Evaluation/rebuttal_results/quant_proposed.scp
```

- 真实标签来自 `Bert-VITS2-master/data/real_eval/real_eval.json`。
- 共 609 个唯一图像键；采用 `analysis_rebuttal_mm25.py` 中相同的年龄标签归并规则。
- 后续实验对应 `quant_ablation_proposed_1e-6.scp`，也有相同的 609 个样本。
- 发表正文写的是 550 张域外图片，但现存 609 条标签统计恰好复现表格数值。正文样本数与实验记录存在不一致，复现时应明确实际采用的集合。

## 如何关联到 checkpoint

1. 发表表格九项数值定位到旧 `_unift_12k` / `quant_proposed` 结果家族。
2. 研究目录 `infer.sh` 第 31、50 行明确使用 `_unift_12k/testset` 输出名；`config.yml` 第 91、125 行保留了相应 `paper_used/proposed/...stage2_unift` 模型路径。目录名、步数与结果家族一致。
3. 当前候选 G/E 与 `paper_used/proposed` 的文件哈希完全相同，前序核查已逐文件读取验证。
4. 原始 G/E 的哈希如下；精简发行版的哈希另列于 `release_models.json`，其 1,571 个张量已逐一验证一致。

| 文件 | 原始 SHA-256 |
| --- | --- |
| `G_12000.pth` | `574f5d4a764c3e72598b76eb267ee382d8d30da3bdfbc764943a1d176a960b15` |
| `E_12000.pth` | `c7f72e10d669c47b3a494d3f9a335bac89b41ef2a9c07ec16f3fe9fcdc440a90` |

最后的 `webui.model`、`webui_batch_quantitative.py` 和 `analysis_rebuttal_mm25.py` 指向的是较新的 1e-6 实验。它们证明后续实验被运行或分析过，却不能推翻已发表数值与旧结果一致的事实。

## 证据边界

- 这是对保留分类标签的重算，没有重新运行年龄、性别或情绪分类器。
- 原论文域外生成音频仍有 609 份；旧域内结果目录只剩目录结构和 SCP，无法对旧域内音频进行逐比特重生成核对。
- 历史生成没有保存 checkpoint 哈希清单，因此“某次进程实际读取的字节”不能由旧日志直接证明。模型归属是九项唯一结果指纹、实验路径、步数及现存 checkpoint 身份共同支持的高置信度结论。
- MCD 与主观 QMOS/CMOS 未重新计算，不能声称已逐项复现全部论文评测。

## 重算

在维护仓库中执行，原研究目录保持只读：

```bash
python scripts/trace_mm25_results.py \
  --research-root /path/to/Image2Voice \
  --output ../outputs/harmonivox-mm25-provenance/result_fingerprint.json
```

本轮报告保存于工作区 `outputs/harmonivox-mm25-provenance/`：包含论文 PDF、Table 2 页面图、逐结果文件哈希和统计、口径一致性检查及 notebook 证据。PDF 和机器相关报告不进入推理发行包。
