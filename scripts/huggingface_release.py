"""Prepare, verify, upload or download the matched HarmoniVox Hub release."""

import argparse
from copy import deepcopy
import errno
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_assets import REPO_ROOT, checkpoint_root, release_models
from scripts.export_inference import sha256
HUB_FILES = ("G_12000.pth", "E_12000.pth", "config.json", "qformer_config.json",
             "manifest.json", "README.md", "SHA256SUMS")
MODEL_FILES = {"G_12000.pth", "E_12000.pth", "config.json", "qformer_config.json"}


def validate_repo_id(repo_id):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id, flags=re.ASCII):
        raise ValueError("Use an explicit Hub model ID: account/model-name")
    return repo_id


def read_checksums(path):
    entries = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if not match or match[2] in entries:
            raise ValueError("Invalid or duplicate SHA256SUMS entry")
        entries[match[2]] = match[1]
    if set(entries) != set(HUB_FILES) - {"SHA256SUMS"}:
        raise ValueError("SHA256SUMS does not describe the complete release")
    return entries


def check(folder, registry=None):
    """Verify every upload byte and reject extra files or a different model."""
    folder = Path(folder).resolve()
    if not folder.is_dir() or {p.name for p in folder.iterdir()} != set(HUB_FILES):
        raise ValueError(f"Folder must contain exactly: {', '.join(HUB_FILES)}")
    for name in HUB_FILES:
        p = folder / name
        if p.is_symlink() or not p.is_file() or not p.stat().st_size:
            raise ValueError(f"Expected a nonempty regular file: {name}")
    expected = read_checksums(folder / "SHA256SUMS")
    hashes = {}
    for name in HUB_FILES:
        hashes[name] = sha256(folder / name)
        if name in expected and hashes[name] != expected[name]:
            raise ValueError(f"SHA-256 mismatch: {name}")
    manifest = json.loads((folder / "manifest.json").read_text())
    registry = release_models() if registry is None else registry
    profile = registry["default_profile"]
    if manifest.get("format") != "harmonivox-inference-v1" or manifest.get("model_profile") != profile:
        raise ValueError(f"Expected the selected inference export: {profile}")
    identity = registry["profiles"][profile]
    for name, digest in identity["exported_checkpoints"].items():
        if hashes.get(name) != digest:
            raise ValueError(f"Not the selected optimizer-free inference weight: {name}")
    if hashes["config.json"] not in identity["config_sha256"]:
        raise ValueError("Configuration does not belong to the selected model")
    if hashes["qformer_config.json"] != sha256(REPO_ROOT / "configs/qformer.json"):
        raise ValueError("Q-Former configuration does not match this inference code")
    records = manifest.get("files", [])
    if len(records) != len(MODEL_FILES) or {v["path"] for v in records} != MODEL_FILES:
        raise ValueError("Manifest must describe exactly the G/E and two configurations")
    for record in records:
        name = record["path"]
        if record["sha256"] != hashes[name] or record["bytes"] != (folder / name).stat().st_size:
            raise ValueError(f"Manifest mismatch: {name}")
    if "code_snapshot" in manifest:
        raise ValueError("The Hub release must contain model resources only, without a code snapshot")
    return {"model_profile": profile, "weight_bytes": sum((folder / n).stat().st_size
            for n in identity["exported_checkpoints"]), "files": [
                {"path": n, "bytes": (folder / n).stat().st_size, "sha256": hashes[n]}
                for n in HUB_FILES]}


def prepare(export_dir, output_dir, repo_id):
    """Flatten a verified inference export into a model-only upload folder."""
    export_dir, output_dir = Path(export_dir).resolve(), Path(output_dir).absolute()
    validate_repo_id(repo_id)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    manifest = deepcopy(json.loads((export_dir / "manifest.json").read_text()))
    if manifest.get("model_profile") != release_models()["default_profile"]:
        raise ValueError("Use the selected paper-proposed inference export")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".harmonivox-hub-", dir=output_dir.parent) as tmp:
        staging = Path(tmp) / "release"
        staging.mkdir()
        for item in manifest["files"]:
            source = (export_dir / item["path"]).resolve()
            if export_dir not in source.parents or source.name not in MODEL_FILES:
                raise ValueError("Unexpected exported model path")
            destination = staging / source.name
            try:
                os.link(source, destination)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copyfile(source, destination)
            item["path"] = destination.name
        qformer = staging / "qformer_config.json"
        shutil.copyfile(REPO_ROOT / "configs/qformer.json", qformer)
        manifest["files"].append({"path": qformer.name, "bytes": qformer.stat().st_size,
                                  "sha256": sha256(qformer)})
        manifest.pop("code_snapshot", None)
        manifest["hub_repo_id"] = repo_id
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        card = (REPO_ROOT / "docs/HF_MODEL_CARD.md").read_text()
        card = card.replace("@REPO_ID@", repo_id)
        (staging / "README.md").write_text(card)
        (staging / "SHA256SUMS").write_text("".join(
            f"{sha256(staging / name)}  {name}\n" for name in sorted(HUB_FILES) if name != "SHA256SUMS"))
        summary = check(staging)
        staging.rename(output_dir)
    return summary


def upload(folder, repo_id, dry_run=False, public=False, api=None):
    """Upload the checked allowlist; dry-run performs no network calls."""
    validate_repo_id(repo_id)
    report = check(folder)
    if dry_run:
        return {"dry_run": True, "repo_id": repo_id, "new_repository_private": not public, **report}
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=not public, exist_ok=True)
    parent = api.model_info(repo_id).sha
    commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(folder),
        path_in_repo="", allow_patterns=list(HUB_FILES), parent_commit=parent,
        commit_message="Release MM25 paper-proposed inference weights and configurations")
    receipt = {"repo_id": repo_id, "revision": commit.oid, "commit_url": str(commit.commit_url), **report}
    receipt_path = Path(folder).resolve().parent / f"harmonivox-upload-{commit.oid}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return {"receipt": str(receipt_path), **receipt}


def download(repo_id, revision, output_dir, local_files_only=False, snapshot_fn=None, registry=None):
    """Download model resources from one immutable revision, then verify and install."""
    validate_repo_id(repo_id)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("--revision must be the full 40-character model commit SHA")
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    if snapshot_fn is None:
        from huggingface_hub import snapshot_download
        snapshot_fn = snapshot_download
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".harmonivox-download-", dir=output_dir.parent) as tmp:
        received, clean = Path(tmp) / "received", Path(tmp) / "checkpoint"
        snapshot = snapshot_fn(repo_id=repo_id, repo_type="model", revision=revision,
                    local_dir=None if local_files_only else str(received),
                    allow_patterns=list(HUB_FILES), local_files_only=local_files_only)
        source_root = Path(snapshot).resolve() if snapshot is not None else received.resolve()
        clean.mkdir()
        for name in HUB_FILES:
            source = source_root / name
            if not source.is_file():
                raise ValueError(f"Missing actual downloaded file: {name}")
            if source_root == received.resolve():
                if source.is_symlink():
                    raise ValueError(f"Expected a regular downloaded file: {name}")
                shutil.move(str(source), str(clean / name))
            else:
                # HF's global offline cache uses symlinks to shared blobs.
                # Copy the bytes into staging; never move or alter cache entries.
                shutil.copyfile(source, clean / name)
        report = check(clean, registry=registry)
        clean.rename(output_dir)
    receipt = {"repo_id": repo_id, "revision": revision, "output_dir": str(output_dir), **report}
    receipt_path = output_dir.parent / f"{output_dir.name}-download-{revision}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return {"receipt": str(receipt_path), **receipt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Build a local upload folder; no remote changes")
    p.add_argument("--export-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repo-id", default=release_models()["hub_repo_id"])
    p = commands.add_parser("check", help="Verify an existing local upload folder")
    p.add_argument("--folder", type=Path, required=True)
    p = commands.add_parser("upload", help="Upload the verified folder using cached Hub credentials")
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--repo-id", default=release_models()["hub_repo_id"])
    p.add_argument("--dry-run", action="store_true", help="Validate and print the plan without accessing the Hub")
    p.add_argument("--public", action="store_true", help="Create a new public repo (existing visibility is unchanged)")
    p = commands.add_parser("download", help="Install verified model resources from a pinned commit")
    p.add_argument("--repo-id", default=release_models()["hub_repo_id"])
    p.add_argument("--revision", required=True)
    p.add_argument("--output-dir", type=Path, default=checkpoint_root())
    p.add_argument("--local-files-only", action="store_true")
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        result = globals()[command](**args)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
