---
pipeline_tag: text-to-speech
library_name: pytorch
language:
- en
- zh
tags:
- harmonivox
- image-conditioned
- text-to-speech
---

# HarmoniVox: Painting Voices to Match the Avatar's Soul

ACM MM 2025 · [Paper](https://doi.org/10.1145/3746027.3755736) ·
[GitHub code](https://github.com/songtoy/HarmoniVox)

HarmoniVox generates speech conditioned on one image, text and an explicit
language. This repository contains the selected MM25 `paper-proposed` model
with its configurations and integrity metadata. Inference code is maintained
separately in the GitHub repository. The later 1e-6 experiment is not included. Use the provided CLI; this is a custom PyTorch model rather than a
Transformers `pipeline` or `AutoModel` checkpoint.

## Files

| File | Contents |
| --- | --- |
| `G_12000.pth` | Speech generator, 238,330,866 bytes |
| `E_12000.pth` | Image encoder, 3,941,418,226 bytes |
| `config.json` | Matching speech configuration, with training corpus paths removed |
| `qformer_config.json` | Q-Former architecture metadata, matching the inference architecture |
| `manifest.json` | Model profile, source/export hashes, verified tensor counts |
| `SHA256SUMS` | SHA-256 checksums of all other repository files |
| `README.md` | This model card |

Optimizer states have been removed without changing model precision or tensor
values. All 1,571 model tensors were verified against the source checkpoints.

## Download and run

Obtain the HarmoniVox inference code separately from the GitHub repository and
follow its README to install Python 3.9 and the pinned CPU or CUDA environment.
No inference scripts or source archive are included in this model repository.

From the `HarmoniVox/` code directory, replace `FULL_COMMIT_SHA` with the full
model commit SHA shown in the Hub's commit history, then download into a new
`../checkpoint/` directory and prepare the separate language resources:

```bash
python scripts/huggingface_release.py download --repo-id @REPO_ID@ --revision FULL_COMMIT_SHA --output-dir ../checkpoint
python scripts/prepare_assets.py --languages EN ZH
python inference.py --language EN --check-only --verify-weights
python inference.py --image /path/to/prompt.jpg --text "Hello, this is HarmoniVox." --language EN --device cuda --output outputs/en.wav
python inference.py --image /path/to/prompt.jpg --text "你好，欢迎使用语音合成。" --language ZH --device cuda --output outputs/zh.wav
```

Use `--device cpu` with the CPU installation. Existing checkpoint directories
and output WAV files are not overwritten. The model download tool verifies
all files before installing them. Alternatively, use a separately installed
`hf` CLI to download only this model repository's files:

```bash
hf download @REPO_ID@ --revision FULL_COMMIT_SHA --local-dir ../checkpoint
```

## Runtime resources

E contains the complete trained CLIP visual backbone and Q-Former. CLIP is
constructed from the pinned OpenCLIP package's `ViT-H-14` configuration;
Q-Former is constructed from `configs/qformer.json` in the code checkout.
Both are then strictly loaded from E. No original CLIP/XLM-RoBERTa weights
or XLM-R tokenizer files need to be downloaded.

The chosen language's BERT model remains separate: `microsoft/deberta-v3-large`
for English and `hfl/chinese-roberta-wwm-ext-large` for Chinese. English also
needs NLTK resources. `scripts/prepare_assets.py` prepares those versioned
assets and checks their file hashes. G includes the speech vocoder; no separate
vocoder, WavLM or discriminator weights are required for the public CLI.

## Validation and limits

English and Chinese short-sentence inference was verified on CPU and RTX A6000
CUDA using Python 3.9, PyTorch 2.3.0, Transformers 4.44.2 and OpenCLIP 2.32.0.
Output is mono PCM-16 WAV at 16 kHz. Removing the redundant original weight
loads preserved the corresponding CPU/CUDA baseline PCM exactly for the tested
inputs at seed 42. This execution regression is not a speech-quality benchmark
or a promise of identical output across different hardware/dependency versions.
Japanese is outside this release's validation scope.

The preserved experiment results match the nine published MM25 Table 2
classification metrics; the detailed evidence and remaining provenance gaps
are in the code repository's `docs/MM25_FINAL_MODEL.md`. Training-data details
and the earliest speech-base checkpoint ancestry remain incomplete in the
release model card. No training data, evaluation images or reference audio is
included. Project code and model licenses have not been selected.

## Citation and acknowledgements

```bibtex
@inproceedings{zhou2025harmonivox,
  title = {HarmoniVox: Painting Voices to Match the Avatar's Soul},
  author = {Zhou, Songtao and Qin, Xiaoyu and Zhou, Yixuan and Wang, Qixin and
            Jin, Zeyu and Wang, Zixuan and Wu, Zhiyong and Jia, Jia},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  year = {2025},
  doi = {10.1145/3746027.3755736}
}
```

We thank the authors and contributors of
[Bert-VITS2](https://github.com/fishaudio/Bert-VITS2),
[VITS](https://github.com/jaywalnut310/vits) and
[Salesforce LAVIS / BLIP-2](https://github.com/salesforce/LAVIS).
Original third-party copyright notices remain in the source files.
