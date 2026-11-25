"""Export matched inference checkpoints, sanitized config and SHA-256 manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_assets import checkpoint_pair, release_models


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inference_config(config):
    """Keep model architecture and audio settings; omit training corpus metadata."""
    data_fields = (
        "sampling_rate", "filter_length", "hop_length", "win_length", "n_mel_channels",
        "mel_fmin", "mel_fmax", "add_blank", "n_speakers", "max_wav_value",
    )
    return {
        "version": config.get("version", "2.3"),
        "train": {"segment_size": config["train"]["segment_size"], "seed": config["train"].get("seed", 42)},
        "data": {**{key: config["data"][key] for key in data_fields if key in config["data"]}, "spk2id": {"imager": 0}},
        "model": {key: value for key, value in config["model"].items() if key != "slm"},
    }


def export(model_dir, config_path, output_dir):
    import torch

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {output_dir}")
    pair = checkpoint_pair(model_dir)
    config = inference_config(json.loads(Path(config_path).read_text(encoding="utf-8")))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "harmonivox"
        (staging / "model").mkdir(parents=True)
        manifest = {"format": "harmonivox-inference-v1", "source_config_sha256": sha256(config_path), "files": []}
        for source in pair:
            print(f"Exporting {source.name}", flush=True)
            checkpoint = torch.load(str(source), map_location="cpu", weights_only=True, mmap=True)
            payload = {
                "model": checkpoint["model"], "optimizer": None,
                "iteration": checkpoint.get("iteration", 0),
                "learning_rate": checkpoint.get("learning_rate", 0),
            }
            destination = staging / "model" / source.name
            torch.save(payload, str(destination))
            reloaded = torch.load(str(destination), map_location="cpu", weights_only=True, mmap=True)
            if reloaded["model"].keys() != payload["model"].keys():
                raise RuntimeError(f"Exported parameter keys differ: {source.name}")
            for key, tensor in payload["model"].items():
                if not torch.equal(tensor, reloaded["model"][key]):
                    raise RuntimeError(f"Exported tensor differs: {source.name}: {key}")
            manifest["files"].append({
                "path": f"model/{source.name}", "bytes": destination.stat().st_size,
                "sha256": sha256(destination), "source_sha256": sha256(source),
                "source_bytes": source.stat().st_size, "tensors_verified": len(payload["model"]),
            })
            del reloaded, payload, checkpoint
        config_output = staging / "config.json"
        config_output.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["files"].append({"path": "config.json", "bytes": config_output.stat().st_size, "sha256": sha256(config_output)})
        source_hashes = {Path(item["path"]).name: item["source_sha256"] for item in manifest["files"] if "source_sha256" in item}
        matches = [
            name for name, profile in release_models()["profiles"].items()
            if all(source_hashes.get(filename) in hashes for filename, hashes in profile["checkpoints"].items())
        ]
        manifest["model_profile"] = matches[0] if len(matches) == 1 else "custom"
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output_dir)
    print(f"Verified inference export: {output_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New destination directory")
    args = parser.parse_args()
    export(args.model_dir, args.config, args.output_dir)


if __name__ == "__main__":
    main()
