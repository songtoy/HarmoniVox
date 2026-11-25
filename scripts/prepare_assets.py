"""Prepare versioned language BERT models and English text resources for inference."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import Request, urlopen
from urllib.parse import quote
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_assets import REPO_ROOT, pretrained_root


def integrity(path, expected):
    """Check LFS SHA-256 or the upstream Git blob identity for small metadata."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    if "sha256" in expected:
        valid = actual["bytes"] == expected["bytes"] and actual["sha256"] == expected["sha256"]
    else:
        data = path.read_bytes()
        blob = lambda value: hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()
        valid = blob(data) == expected["git_blob_sha1"]
        # Some existing JSON downloads differ only by their final newline.
        # Accept exactly this formatting difference; do not change token content.
        if not valid and path.suffix == ".json":
            valid = blob(data.rstrip(b"\r\n")) == expected["git_blob_sha1"]
    if not valid:
        raise ValueError(f"File differs from the pinned upstream asset: {path}")
    return actual


def local_copy(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".asset-import-", dir=destination.parent) as folder:
        temporary = Path(folder) / destination.name
        shutil.copyfile(source, temporary)
        result = integrity(temporary, expected)
        # Atomically refuse to overwrite a file that appeared during preparation.
        os.link(temporary, destination)
    return result


def download_file(url, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".asset-download-", dir=destination.parent) as folder:
        temporary = Path(folder) / destination.name
        with urlopen(Request(url, headers={"User-Agent": "HarmoniVox"}), timeout=60) as response, temporary.open("wb") as stream:
            received = 0
            next_update = 256 * 1024 * 1024
            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                stream.write(block)
                received += len(block)
                if received >= next_update:
                    print(f"    {destination.name}: {received / expected['bytes']:.0%}", flush=True)
                    next_update += 256 * 1024 * 1024
        result = integrity(temporary, expected)
        os.link(temporary, destination)
    return result


def prepare(manifest, destination, languages, local_sources):
    records = []
    for name, asset in manifest["assets"].items():
        if asset["languages"] and not set(asset["languages"]) & set(languages):
            continue
        print(f"Preparing {name}: {asset['repo_id']}@{asset['revision']}", flush=True)
        directory = destination / asset["directory"]
        for filename, expected in asset["files"].items():
            target = directory / filename
            origin = "existing"
            if target.is_file():
                result = integrity(target, expected)
            else:
                candidates = [root / asset["directory"] / filename for root in local_sources]
                source = next((path for path in candidates if path.is_file()), None)
                if source:
                    print(f"  Importing {asset['directory']}/{filename}", flush=True)
                    result = local_copy(source, target, expected)
                    origin = "local-copy"
                else:
                    print(f"  Downloading {asset['directory']}/{filename}", flush=True)
                    url = f"https://huggingface.co/{asset['repo_id']}/resolve/{asset['revision']}/{quote(filename)}?download=true"
                    result = download_file(url, target, expected)
                    origin = "download"
            records.append({
                "path": str(target.relative_to(destination)), "repo_id": asset["repo_id"],
                "revision": asset["revision"], "origin": origin, **result,
            })
    if "EN" in languages:
        nltk = manifest["nltk"]
        for resource in nltk["resources"]:
            archive_path = destination / "nltk_data" / resource["directory"] / (resource["name"] + ".zip")
            if archive_path.is_file():
                integrity(archive_path, resource)
            else:
                url = f"https://raw.githubusercontent.com/nltk/nltk_data/{nltk['revision']}/{resource['archive_path']}"
                print(f"Downloading English text resource: {resource['name']}", flush=True)
                download_file(url, archive_path, resource)
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.namelist():
                    path = Path(member)
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError(f"Invalid text-resource archive member: {member}")
                archive.extractall(archive_path.parent)
            records.append({
                "path": str(archive_path.relative_to(destination)),
                "repo_id": "nltk/nltk_data", "revision": nltk["revision"],
                **integrity(archive_path, resource),
            })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=pretrained_root())
    parser.add_argument("--languages", nargs="+", choices=("EN", "ZH", "JP"), default=("EN", "ZH"))
    parser.add_argument("--local-source", type=Path, action="append", default=[], help="Optional existing asset root to copy from; may be repeated")
    args = parser.parse_args()
    destination = args.assets_dir.expanduser().resolve()
    local_sources = [path.expanduser().resolve() for path in args.local_source]
    manifest = json.loads((REPO_ROOT / "configs/assets.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    records = prepare(manifest, destination, args.languages, local_sources)
    report = destination / "assets-manifest.json"
    existing = json.loads(report.read_text()).get("files", []) if report.is_file() else []
    valid_paths = {f"{asset['directory']}/{name}"
                   for asset in manifest["assets"].values() for name in asset["files"]}
    valid_paths.update(f"nltk_data/{item['directory']}/{item['name']}.zip"
                       for item in manifest["nltk"]["resources"])
    # Drop obsolete CLIP/XLM-R records while retaining other prepared languages.
    merged = {item["path"]: item for item in existing if item["path"] in valid_paths}
    merged.update({item["path"]: item for item in records})
    report.write_text(json.dumps({"format": manifest["format"], "files": list(merged.values())}, indent=2) + "\n")
    print(f"Prepared {len(records)} verified asset files in {destination}")


if __name__ == "__main__":
    main()
