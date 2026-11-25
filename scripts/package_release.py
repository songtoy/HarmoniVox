"""Build an inference-only source ZIP from an explicit file list."""

import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
# The inference working tree and source artifact share the same file boundary.
FILES = (
    ".gitignore", ".github/workflows/ci.yml", "CITATION.cff",
    "readme.md", "requirements.txt", "requirements-inference.txt",
    "requirements-lock-cpu.txt", "requirements-lock-cu118.txt", "config.py", "config.yml",
    "inference.py", "release_assets.py", "infer_web_qformer_textual_multi.py",
    "configs/harmonivox.json", "configs/qformer.json", "configs/assets.json", "release_models.json",
    "models_qformer_textual_multi.py", "models.py", "commons.py", "modules.py",
    "attentions.py", "transforms.py", "meta_modules.py", "mel_processing.py", "utils.py",
    "monotonic_align/__init__.py", "monotonic_align/core.py",
    "text/__init__.py", "text/symbols.py", "text/cleaner.py", "text/chinese.py",
    "text/chinese_bert.py", "text/tone_sandhi.py", "text/opencpop-strict.txt",
    "text/english.py", "text/english_bert_mock.py", "text/cmudict.rep",
    "text/japanese.py", "text/japanese_bert.py", "text/bert_utils.py",
    "blip2_models/__init__.py", "blip2_models/blip2.py", "blip2_models/base_model.py",
    "blip2_models/Qformer.py", "blip2_models/common/dist_utils.py",
    "scripts/export_inference.py", "scripts/package_release.py",
    "scripts/audit_checkpoints.py",
    "scripts/trace_mm25_results.py",
    "scripts/prepare_assets.py",
    "scripts/check_repository.py",
    "scripts/huggingface_release.py",
    "tests/test_release.py",
    "tests/test_huggingface.py",
    "docs/RELEASE.md", "docs/MODEL_CARD.md", "docs/THIRD_PARTY.md", "docs/CHECKPOINT_LINEAGE.md", "docs/MM25_FINAL_MODEL.md", "docs/CLEAN_INSTALL.md",
    "docs/CUDA_VALIDATION.md", "docs/REPOSITORY.md",
    "docs/CODE_ORIGINS.md",
    "docs/HUGGINGFACE.md", "docs/HF_MODEL_CARD.md",
)


def package(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    # Preserve project license/notice documents alongside the upstream texts.
    files = list(FILES) + [path.name for pattern in ("LICENSE*", "NOTICE*") for path in ROOT.glob(pattern) if path.is_file()]
    for name in files:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f"Release source is missing: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            archive.write(ROOT / name, f"HarmoniVox/{name}")
    print(f"Packaged {len(files)} files: {output}")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.output)


if __name__ == "__main__":
    main()
