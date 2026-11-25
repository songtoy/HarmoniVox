# HarmoniVox model card — release draft

Paper: [HarmoniVox: Painting Voices to Match the Avatar's Soul](https://hcsi.cs.tsinghua.edu.cn/Paper/paper25/MM2025_Songtao%20Zhou_HarmoniVox_Painting%20Voices%20to%20Match%20the%20Avatar%E2%80%99s%20Soul.pdf), ACM MM 2025, DOI `10.1145/3746027.3755736`.

## Model and interface

- Route: `qformer_textual_multi` (the proposed route in the research workspace).
- Inputs: one RGB image, text and an explicit language (`EN`, `ZH`, `JP`).
- Output: mono speech; the local checkpoint configuration uses 16,000 Hz.
- Architecture: CLIP ViT-H/14 visual encoder, XLM-RoBERTa-initialized Q-Former,
  and Bert-VITS2-based synthesizer with image-conditioned style features.
- Local candidate: `G_12000.pth` + `E_12000.pth`, configuration format `2.3`.
  This format identifier is not a public release version.
- Verified identity: `paper-proposed`, byte-identical to the research workspace's
  `paper_used/proposed` files. The preserved results match all nine
  published MM25 Table 2 classification metrics. The later
  `experimental-proposed-1e-6` is a different experiment and does not match those
  published metrics. See [the result audit](MM25_FINAL_MODEL.md).
- The inference implementation uses a fixed internal speaker ID of zero.
  The exported config retains the architecture's `n_speakers` value and exposes
  only the UI alias `imager`; it omits the training corpus's speaker inventory.

Prepared Hub target: `sonterzhuo/HarmoniVox`. The upload bundle contains only model
weights, configurations and integrity metadata; code is maintained separately. The repository and
remote revision are not published by local preparation; see
[HUGGINGFACE.md](HUGGINGFACE.md).

## Files and integrity

The export contains `model/G_12000.pth`, `model/E_12000.pth`, `config.json`
and `manifest.json`. The manifest records source and exported SHA-256 hashes,
sizes and verified tensor counts. Exporting removes optimizer state without
changing tensor values or precision. Base-model and tokenizer assets are
separate downloads for the selected language; see [the README](../readme.md).
E includes the complete CLIP visual backbone and Q-Former parameters.
The loader builds their architectures from local configuration and strictly
loads E, so original CLIP/XLM-RoBERTa weights and XLM-R tokenizer assets are
not runtime dependencies. `configs/qformer.json` is bundled with the code.

## Validation

Validation results and remaining release gates are tracked in
[RELEASE.md](RELEASE.md). A successful short synthesis confirms execution,
not speech quality or reproduction of the paper's evaluation scores.

The first release validation covers English and Chinese on CPU and CUDA.
Japanese is a retained legacy code path, excluded from this validation scope
and from the first release support claim. See [CUDA validation](CUDA_VALIDATION.md).

## Licensing status

The project code and model licenses have not been selected. Intended use is
academic and noncommercial research; this records the release intent rather
than granting a specific license. Code origins are recorded in
[CODE_ORIGINS.md](CODE_ORIGINS.md).

## Information to complete before publication

- Public model repository, immutable revision, version, release date and contact.
- Training datasets, versions, splits, filtering, speaker consent and relevant terms.
- Relationship between this checkpoint and any pretrained speech checkpoint.
- Training hardware, procedure and reproducible evaluation setup.
- Language coverage and measured quality, limitations and failure cases.
- Project/model license, upstream notices and redistribution review.
- Optional BibTeX rendering; paper identity and authors are provided in `CITATION.cff`.
- Intended uses and model-specific restrictions, if any.

No training corpus, images, evaluation samples or original speaker list are
included in the inference artifact. Public example images and audio are still
to be selected and reviewed.
