import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.export_inference import sha256
from scripts.huggingface_release import (
    HUB_FILES, MODEL_FILES, check, download, prepare, upload,
)
from scripts.package_release import ROOT


def refresh_checksums(folder):
    (folder / "SHA256SUMS").write_text("".join(
        f"{sha256(folder / name)}  {name}\n" for name in sorted(HUB_FILES) if name != "SHA256SUMS"))


def fixture(folder):
    folder.mkdir()
    for name in ("G_12000.pth", "E_12000.pth"):
        (folder / name).write_bytes(("synthetic test weight " + name).encode())
    for name, source in (("config.json", "harmonivox.json"), ("qformer_config.json", "qformer.json")):
        shutil.copyfile(ROOT / "configs" / source, folder / name)
    (folder / "README.md").write_text("Synthetic local test model card.\n")
    manifest = {"format": "harmonivox-inference-v1", "model_profile": "paper-proposed",
        "files": [{"path": name, "bytes": (folder / name).stat().st_size,
                   "sha256": sha256(folder / name)} for name in sorted(MODEL_FILES)]}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    refresh_checksums(folder)
    return {"default_profile": "paper-proposed", "profiles": {"paper-proposed": {
        "config_sha256": [sha256(folder / "config.json")], "exported_checkpoints": {
            name: sha256(folder / name) for name in ("G_12000.pth", "E_12000.pth")}}}}


class HuggingFaceReleaseTests(unittest.TestCase):
    def test_bundle_rejects_relabelled_weights_extra_files_and_unsafe_checksums(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "upload"
            registry = fixture(folder)
            self.assertEqual(len(check(folder, registry=registry)["files"]), 7)
            extra = folder / ".env"
            extra.write_text("not for upload")
            with self.assertRaises(ValueError):
                check(folder, registry=registry)
            extra.unlink()
            source_zip = folder / "HarmoniVox-inference.zip"
            source_zip.write_bytes(b"Code must not be included in the Hub upload")
            with self.assertRaises(ValueError):
                check(folder, registry=registry)
            source_zip.unlink()
            sums = (folder / "SHA256SUMS").read_bytes()
            (folder / "SHA256SUMS").write_text("0" * 64 + "  ../outside\n")
            with self.assertRaises(ValueError):
                check(folder, registry=registry)
            (folder / "SHA256SUMS").write_bytes(sums)
            # Self-consistent remote checksums must not override the trusted model identity.
            (folder / "E_12000.pth").write_bytes(b"same-name ablation or training checkpoint")
            manifest = json.loads((folder / "manifest.json").read_text())
            for item in manifest["files"]:
                if item["path"] == "E_12000.pth":
                    item.update(bytes=(folder / item["path"]).stat().st_size,
                                sha256=sha256(folder / item["path"]))
            (folder / "manifest.json").write_text(json.dumps(manifest))
            refresh_checksums(folder)
            with self.assertRaisesRegex(ValueError, "selected optimizer-free"):
                check(folder, registry=registry)

    def test_download_installs_only_verified_data_and_preserves_existing_destinations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry = fixture(root / "remote")
            calls = []

            def snapshot(**kwargs):
                calls.append(kwargs)
                received = Path(kwargs["local_dir"])
                received.mkdir()
                for name in HUB_FILES:
                    shutil.copyfile(root / "remote" / name, received / name)
                (received / ".cache").mkdir()

            target = root / "checkpoint"
            with self.assertRaises(ValueError):
                download("sonterzhuo/HarmoniVox", "main", target,
                         snapshot_fn=snapshot, registry=registry)
            self.assertFalse(calls)
            report = download("sonterzhuo/HarmoniVox", "a" * 40, target,
                              snapshot_fn=snapshot, registry=registry)
            self.assertEqual(calls[0]["revision"], "a" * 40)
            self.assertEqual(set(calls[0]["allow_patterns"]), set(HUB_FILES))
            self.assertEqual({p.name for p in target.iterdir()}, set(HUB_FILES))
            self.assertFalse((target / ".cache").exists())
            self.assertEqual(report["revision"], "a" * 40)
            with self.assertRaises(FileExistsError):
                download("sonterzhuo/HarmoniVox", "a" * 40, target,
                         snapshot_fn=snapshot, registry=registry)
            self.assertEqual(len(calls), 1)

            def corrupted(**kwargs):
                snapshot(**kwargs)
                (Path(kwargs["local_dir"]) / "G_12000.pth").write_bytes(b"partial file")

            broken = root / "broken"
            with self.assertRaises(ValueError):
                download("sonterzhuo/HarmoniVox", "a" * 40, broken,
                         snapshot_fn=corrupted, registry=registry)
            self.assertFalse(broken.exists())
            self.assertFalse(list(root.glob(".harmonivox-download-*")))

            def cached_snapshot(**kwargs):
                self.assertTrue(kwargs["local_files_only"])
                self.assertIsNone(kwargs["local_dir"])
                return str(root / "remote")

            offline = root / "offline"
            download("sonterzhuo/HarmoniVox", "a" * 40, offline, local_files_only=True,
                     snapshot_fn=cached_snapshot, registry=registry)
            self.assertEqual({p.name for p in offline.iterdir()}, set(HUB_FILES))
            self.assertEqual({p.name for p in (root / "remote").iterdir()}, set(HUB_FILES))

    def test_upload_dry_run_is_offline_and_real_api_path_records_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "upload"
            registry = fixture(folder)
            api = Mock()
            api.model_info.return_value.sha = "a" * 40
            api.upload_folder.return_value.oid = "b" * 40
            api.upload_folder.return_value.commit_url = "https://huggingface.co/sonterzhuo/HarmoniVox/commit/" + "b" * 40
            with patch("scripts.huggingface_release.release_models", return_value=registry):
                plan = upload(folder, "sonterzhuo/HarmoniVox", dry_run=True, api=api)
                self.assertTrue(plan["dry_run"])
                self.assertFalse(api.mock_calls)
                receipt = upload(folder, "sonterzhuo/HarmoniVox", api=api)
            api.create_repo.assert_called_once_with(repo_id="sonterzhuo/HarmoniVox",
                repo_type="model", private=True, exist_ok=True)
            kwargs = api.upload_folder.call_args.kwargs
            self.assertEqual(set(kwargs["allow_patterns"]), set(HUB_FILES))
            self.assertNotIn("HarmoniVox-inference.zip", kwargs["allow_patterns"])
            self.assertEqual(kwargs["parent_commit"], "a" * 40)
            self.assertNotIn("delete_patterns", kwargs)
            self.assertEqual(receipt["revision"], "b" * 40)
            self.assertEqual(Path(receipt["receipt"]).parent, folder.parent)
            self.assertEqual({p.name for p in folder.iterdir()}, set(HUB_FILES))

    def test_prepare_rejects_wrong_profile_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "export").mkdir()
            (root / "export/manifest.json").write_text(json.dumps({"model_profile": "experimental-proposed-1e-6"}))
            with self.assertRaises(ValueError):
                prepare(root / "export", root / "output", "sonterzhuo/HarmoniVox")
            self.assertFalse((root / "output").exists())
            (root / "output").mkdir()
            with self.assertRaises(FileExistsError):
                prepare(root / "export", root / "output", "sonterzhuo/HarmoniVox")


if __name__ == "__main__":
    unittest.main()
