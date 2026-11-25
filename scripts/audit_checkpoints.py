"""Inventory research checkpoints without executing checkpoint pickle code."""

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(path):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        payload = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
        state = payload["model"]
        groups = (payload.get("optimizer") or {}).get("param_groups", [])
        return {
            "iteration": payload.get("iteration"),
            "learning_rate": payload.get("learning_rate"),
            "tensor_count": len(state),
            "tensor_bytes": sum(value.numel() * value.element_size() for value in state.values()),
            "model_prefixes": sorted({key.split(".")[0] for key in state}),
            "optimizer_groups": [
                {"lr": group.get("lr"), "initial_lr": group.get("initial_lr"), "parameter_count": len(group["params"])}
                for group in groups
            ],
        }


def log_evidence(path):
    """Record loaded/saved checkpoint events, retaining original line numbers."""
    events = []
    configurations = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            message = line.rstrip().split("\tINFO\t")[-1]
            if "Loaded checkpoint" in message or "Saving model and optimizer state" in message:
                events.append({"line": line_number, "message": message})
            elif message.startswith("{'train':"):
                try:
                    config = ast.literal_eval(message)
                    configurations.append({
                        "line": line_number, "model_dir": config.get("model_dir"),
                        "training_files": config["data"].get("training_files"),
                        "learning_rate": config["train"].get("learning_rate"),
                        "epochs": config["train"].get("epochs"),
                    })
                except (ValueError, SyntaxError, KeyError):
                    continue
    return {"path": str(path), "checkpoint_events": events, "configurations": configurations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-ge", action="store_true", help="Read all G/E files to identify exact duplicates")
    args = parser.parse_args()
    research = args.research_root.expanduser().resolve()
    candidate = args.candidate_dir.expanduser().resolve()
    if not research.is_dir() or not candidate.is_dir():
        parser.error("research root and candidate directory must exist")
    paths = sorted(set(path.resolve() for root in (research, candidate) for path in root.rglob("*.pth")))
    if not paths:
        parser.error("no .pth files found")
    records = []
    hashes = defaultdict(list)
    for index, path in enumerate(paths, 1):
        record = {"path": str(path), "bytes": path.stat().st_size}
        try:
            record.update(metadata(path))
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        if args.hash_ge and path.name.startswith(("G_", "E_")):
            record["sha256"] = sha256(path)
            hashes[record["sha256"]].append(str(path))
        records.append(record)
        print(f"{index}/{len(paths)} {path.name}: {'unreadable' if 'error' in record else 'readable'}", flush=True)
    logs = [log_evidence(path) for path in sorted(research.rglob("train.log"))]
    aliases = [{"path": str(path), "target": str(path.resolve())} for path in sorted(research.rglob("*")) if path.is_symlink() and path.name in ("paper_used", "ablations_loss")]
    report = {
        "checkpoints": records, "logs": logs, "directory_aliases": aliases,
        "identical_files": [{"sha256": digest, "paths": values} for digest, values in hashes.items() if len(values) > 1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved checkpoint and log evidence: {args.output}")


if __name__ == "__main__":
    main()
