"""Paths and asset checks shared by inference and release tools (standard library only)."""

import os
import re
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BERT_MODELS = {
    "EN": "deberta-v3-large",
    "ZH": "chinese-roberta-wwm-ext-large",
    "JP": "deberta-v2-large-japanese-char-wwm",
}


def pretrained_root():
    return Path(os.environ.get(
        "HARMONIVOX_PRETRAINED_DIR", REPO_ROOT.parent / "pretrained_models"
    )).expanduser().resolve()


def checkpoint_root():
    return Path(os.environ.get(
        "HARMONIVOX_CHECKPOINT_DIR", REPO_ROOT.parent / "checkpoint"
    )).expanduser().resolve()


def nltk_root():
    return Path(os.environ.get(
        "HARMONIVOX_NLTK_DIR", pretrained_root() / "nltk_data"
    )).expanduser().resolve()


def bert_path(language):
    return pretrained_root() / "bert" / BERT_MODELS[language]


def qformer_config_path():
    return REPO_ROOT / "configs" / "qformer.json"


def qformer_path():
    return Path(os.environ.get(
        "HARMONIVOX_XLM_ROBERTA_PATH", pretrained_root() / "xlm-roberta-base"
    )).expanduser().resolve()


def checkpoint_pair(model_dir, step=None):
    """Select the latest step only if both G and E exist at that step."""
    model_dir = Path(model_dir)
    steps = {}
    for prefix in ("G", "E"):
        steps[prefix] = {
            int(match.group(1)): path
            for path in model_dir.glob(f"{prefix}_*.pth")
            if (match := re.fullmatch(rf"{prefix}_(\d+)\.pth", path.name))
            and path.is_file()
        }
    if step is not None:
        if step not in steps["G"] or step not in steps["E"]:
            raise FileNotFoundError(f"Expected G_{step}.pth and E_{step}.pth in {model_dir}")
        return steps["G"][step], steps["E"][step]
    if not steps["G"] or steps["G"].keys() != steps["E"].keys():
        raise FileNotFoundError(
            f"Expected matched G_<step>.pth and E_<step>.pth files in {model_dir}; "
            f"found G steps {sorted(steps['G'])}, E steps {sorted(steps['E'])}."
        )
    step = max(steps["G"])
    return steps["G"][step], steps["E"][step]


def missing_assets(model_dir, config_path, languages, step=None):
    """Check local files; recognize Git LFS pointers as missing assets."""
    required = [Path(config_path), qformer_config_path()]
    try:
        required.extend(checkpoint_pair(model_dir, step=step))
    except FileNotFoundError as exc:
        errors = [str(exc)]
    else:
        errors = []
    for language in languages:
        model = bert_path(language)
        required.extend(model / name for name in ("config.json", "pytorch_model.bin"))
        required.append(model / ("spm.model" if language == "EN" else "vocab.txt"))
    if "EN" in languages:
        required.extend(nltk_root() / name for name in (
            "taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle",
            "corpora/cmudict/cmudict",
        ))
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty file: {path}")
        elif path.stat().st_size < 1024:
            if path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
                errors.append(f"Git LFS pointer; download the actual file: {path}")
    return errors


def release_models():
    return json.loads((REPO_ROOT / "release_models.json").read_text(encoding="utf-8"))


def verify_model_identity(model_dir, config_path, profile, step, registry=None):
    """Reject same-step checkpoints from a different experiment before inference."""
    registry = release_models() if registry is None else registry
    expected = registry["profiles"][profile]
    if step != expected["checkpoint_step"]:
        raise ValueError(f"Profile {profile} requires step {expected['checkpoint_step']}, got {step}")
    paths = [Path(config_path), *checkpoint_pair(model_dir, step=step)]
    result = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        allowed = expected["config_sha256"] if path == Path(config_path) else expected["checkpoints"][path.name]
        if actual not in allowed:
            raise ValueError(
                f"File does not belong to model profile {profile}: {path}\n"
                f"SHA-256: {actual}. Do not mix proposed, revised and ablation checkpoints."
            )
        result[str(path)] = actual
    return result
