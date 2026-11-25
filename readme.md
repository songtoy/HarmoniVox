# HarmoniVox

**HarmoniVox: Painting Voices to Match the Avatar's Soul** · ACM MM 2025

[Paper](https://doi.org/10.1145/3746027.3755736) · [Citation](#citation)

Image-conditioned text-to-speech using a CLIP visual encoder, Q-Former and a
Bert-VITS2-based speech synthesizer. This release preparation covers the
`qformer_textual_multi` inference route and its matching G/E checkpoints.

**Status:** local release candidate. Public checkpoint downloads and the code/model
licenses are pending. See the maintainer [release checklist](docs/RELEASE.md)
and [model card](docs/MODEL_CARD.md).

## Installation

Use Python 3.9 and an isolated environment. The CPU installation uses a complete
dependency lock; the public entry point is the CLI and does not require Gradio.
The isolated Linux CPU and CUDA installations and both EN/ZH synthesis paths
have been verified; see [the clean-install record](docs/CLEAN_INSTALL.md) and
[CUDA validation](docs/CUDA_VALIDATION.md). The first release validation covers
English and Chinese. The legacy Japanese path is outside this scope.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.0 setuptools==69.5.1 wheel==0.45.1
python -m pip install torch==2.3.0 torchaudio==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock-cpu.txt
```

Any Python 3.9 interpreter, including one provided by Conda, can create this
venv. Do not enable system site packages. `requirements.txt` points to the
main CLI dependencies in `requirements-inference.txt`; the two platform lock
files pin the complete validated CPU and CUDA environments.

For CUDA 11.8, use a separate environment and replace the two dependency-install
commands above with:

```bash
python -m pip install torch==2.3.0 torchaudio==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-lock-cu118.txt
```

The CPU lock includes CPU-specific wheel versions; use the cu118 lock for CUDA.
The CUDA path was verified on an RTX A6000 with PyTorch 2.3.0+cu118.
Use `--device cuda` with this installation.

## Model files

Keep weights outside the Git repository. The default layout is:

```text
workspace/
├── HarmoniVox/
├── checkpoint/
│   ├── G_12000.pth
│   ├── E_12000.pth
│   ├── config.json
│   └── manifest.json
└── pretrained_models/
    ├── nltk_data/
    └── bert/
        ├── deberta-v3-large/                      # EN
        ├── chinese-roberta-wwm-ext-large/         # ZH
        └── deberta-v2-large-japanese-char-wwm/     # JP, optional
```

English text resources are kept in `pretrained_models/nltk_data/`, including
the averaged perceptron tagger and CMU dictionary. Inference uses this explicit
location rather than relying on a pre-existing global NLTK installation.

The prepared model repository is
[`sonterzhuo/HarmoniVox`](https://huggingface.co/sonterzhuo/HarmoniVox); the files
have not been uploaded yet. After publication, replace `FULL_COMMIT_SHA` with
the full model commit SHA and download into a new `../checkpoint/` directory:

```bash
python scripts/huggingface_release.py download \
  --repo-id sonterzhuo/HarmoniVox \
  --revision FULL_COMMIT_SHA \
  --output-dir ../checkpoint
```

The download is verified before installation and will not overwrite an
existing checkpoint directory. If the selected files are already present,
continue with resource preparation and the CLI below. The matching speech
configuration is also bundled in `configs/harmonivox.json`.

Inference code is maintained separately in this code repository. The Hub
repository contains model weights, configurations and integrity metadata only;
see [the Hub preparation and download guide](docs/HUGGINGFACE.md).

Use the matching HarmoniVox configuration and checkpoint pair. Discriminator
weights (`D_*`, `DUR_*`, `WD_*`) and WavLM are unnecessary for this inference
route. E already contains the trained CLIP visual backbone and Q-Former. The
loader constructs CLIP from OpenCLIP's packaged `ViT-H-14` architecture and
Q-Former from the bundled `configs/qformer.json`, then strictly loads E.
**No original CLIP or XLM-RoBERTa weights, or XLM-R tokenizer files, are needed.**
The selected language's BERT weights are separate and remain required for text
features; they are not part of G/E.

The CLI defaults to the bundled `configs/harmonivox.json` and explicitly loads
`G_12000.pth` / `E_12000.pth`. It does not automatically switch to a later step.
Synthesis verifies their SHA-256 hashes against the selected profile in
`release_models.json`, so same-step ablation checkpoints cannot be substituted
silently. The default profile `paper-proposed` matches the original
`paper_used/proposed` weights, whose preserved results match all nine published
MM25 Table 2 classification metrics. The later 1e-6 experiment is separate,
under `experimental-proposed-1e-6`; it does not match the published final row.
See [the MM25 result audit](docs/MM25_FINAL_MODEL.md) and
[checkpoint lineage](docs/CHECKPOINT_LINEAGE.md).
Custom experiments require `--model-profile custom`, with explicit model/config
paths and, if needed, `--checkpoint-step`; this disables identity verification.

Prepare the language BERT models and English text resources from fixed upstream commits:

```bash
python scripts/prepare_assets.py --languages EN ZH
```

This downloads the required weights and tokenizer/configuration files, verifies
their contents against `configs/assets.json`, and writes
`pretrained_models/assets-manifest.json`. Existing matching files are reused.
Existing files with different content are reported rather than overwritten.

To reuse an existing model collection, add `--local-source /path/to/models`.
The supplied directory should contain the `bert/` tree shown above. Matching files are copied and verified; missing files
are downloaded. Checkpoints are prepared separately.

The pinned upstream sources are:

| Component | Upstream |
| --- | --- |
| English BERT features | [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) |
| Chinese BERT features | [hfl/chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) |
| Japanese BERT features | [ku-nlp/deberta-v2-large-japanese-char-wwm](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm) |

BERT assets are loaded only for the selected language. English needs
`spm.model`; Chinese and Japanese need `vocab.txt`. Their model directories
must also contain `config.json` and `pytorch_model.bin`.

Override asset locations when using another layout:

```bash
export HARMONIVOX_PRETRAINED_DIR=/path/to/pretrained_models
export HARMONIVOX_CHECKPOINT_DIR=/path/to/checkpoint
```

## CLI inference

Check required files without loading PyTorch or allocating model memory:

```bash
python inference.py --language EN --check-only
```

Add `--verify-weights` to this preflight to read and hash the selected weights.
Named model profiles are always verified before synthesis.

Synthesize speech from one image and a text prompt:

```bash
python inference.py \
  --image /path/to/prompt.jpg \
  --text "Hello, this is HarmoniVox." \
  --language EN \
  --device cpu \
  --output outputs/example.wav
```

Use `--device cpu` when CUDA is unavailable. Output is a mono PCM-16 WAV at the
sample rate in the model configuration (16 kHz for the local 12000-step pair).
Existing output files are rejected. Image conditioning determines the voice
style; this entry point has no speaker-selection or reference-audio control.

For custom checkpoint locations:

```bash
python inference.py \
  --model-dir /path/to/harmonivox/model \
  --config /path/to/harmonivox/config.json \
  --image /path/to/prompt.jpg --text "你好，世界。" --language ZH \
  --device cpu \
  --output outputs/example-zh.wav
```

The released model uses a deterministic duration predictor (`use_sdp=false`).
`--length-scale` controls duration and `--noise-scale` controls latent noise.
The legacy SDP ratio and duration-noise sliders have no effect on this model
and are not exposed by the release CLI.

`python inference.py --help` lists synthesis controls. `--seed` defaults to 42;
identical output across hardware and dependency versions is not guaranteed.

## Repository and release artifacts

| Path | Role |
| --- | --- |
| `inference.py` | Public CLI entry point |
| `infer_web_qformer_textual_multi.py` | Shared model loading and inference |
| `models.py`, `models_qformer_textual_multi.py` | Speech synthesizer and image encoder |
| `text/`, `monotonic_align/`, `blip2_models/` | Model support code and text processing |
| `scripts/export_inference.py` | Remove optimizer state, sanitize config, verify tensors, hash weights |
| `scripts/package_release.py` | Package explicitly listed inference source files |
| `scripts/prepare_assets.py`, `configs/assets.json` | Prepare and verify fixed base-model/text resources |
| `requirements-lock-cpu.txt`, `requirements-lock-cu118.txt` | Complete pinned CPU/CUDA inference environments |
| `scripts/check_repository.py`, `.github/workflows/ci.yml` | Source boundary and automated release checks |
| `scripts/huggingface_release.py`, `docs/HUGGINGFACE.md` | Prepare/verify the upload bundle, upload it, and download a fixed model revision |
| `CITATION.cff` | ACM MM 2025 paper citation |
| `release_models.json` | Exact original/exported checkpoint identities; keeps paper and revision models distinct |
| `docs/` | Release checklist, model card and third-party inventory |

This repository contains the inference source subset with a new Git history.
Research training, ablation, evaluation, legacy WebUI and unused vendored
annotators are excluded. The original research repository and its history are
preserved separately; see [repository scope](docs/REPOSITORY.md).
Model weights are distributed separately through Hugging Face; see
[model download and verification](docs/HUGGINGFACE.md).

## Citation

If you use HarmoniVox in your research, please cite:

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

## Acknowledgements

HarmoniVox builds on [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) for
multilingual text processing and speech synthesis. We thank the Bert-VITS2
authors and contributors for their open-source implementation.

We also acknowledge [VITS](https://github.com/jaywalnut310/vits) and
[Salesforce LAVIS / BLIP-2](https://github.com/salesforce/LAVIS), whose components
are used and adapted in this project. Existing third-party copyright notices
are retained in the source files.
