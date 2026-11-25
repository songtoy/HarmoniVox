import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from release_assets import checkpoint_pair, missing_assets, verify_model_identity
from scripts.export_inference import export, inference_config, sha256
from scripts.package_release import package
from scripts.prepare_assets import integrity, local_copy


class ReleaseTests(unittest.TestCase):
    def test_asset_import_validates_content_and_preserves_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"known model bytes")
            expected = {"bytes": source.stat().st_size, "sha256": sha256(source)}
            destination = root / "assets/model.bin"
            self.assertEqual(local_copy(source, destination, expected)["sha256"], expected["sha256"])
            source.write_bytes(b"different weights")
            with self.assertRaises(ValueError):
                integrity(source, expected)
            with self.assertRaises(FileExistsError):
                local_copy(destination, destination, expected)
            self.assertEqual(destination.read_bytes(), b"known model bytes")

    def test_metadata_newline_is_accepted_without_accepting_token_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            content = b'{"token": "[MASK]"}'
            expected = {"bytes": len(content), "git_blob_sha1": hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()}
            path.write_bytes(content + b"\n")
            integrity(path, expected)
            path.write_bytes(b'{"token": "[UNK]"}\n')
            with self.assertRaises(ValueError):
                integrity(path, expected)

    def test_same_step_ablation_or_mixed_pair_is_rejected_by_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in (("G_12000.pth", b"generator"), ("E_12000.pth", b"encoder"), ("config.json", b"{}")):
                (root / name).write_bytes(content)
            registry = {"profiles": {"paper": {"checkpoint_step": 12000, "config_sha256": [sha256(root / "config.json")], "checkpoints": {name: [sha256(root / name)] for name in ("G_12000.pth", "E_12000.pth")}}}}
            self.assertEqual(len(verify_model_identity(root, root / "config.json", "paper", 12000, registry)), 3)
            (root / "E_12000.pth").write_bytes(b"same-shape ablation encoder")
            with self.assertRaises(ValueError):
                verify_model_identity(root, root / "config.json", "paper", 12000, registry)

    def test_checkpoint_selection_is_numeric_and_rejects_unpaired_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("G_9.pth", "E_9.pth", "G_100.pth", "E_100.pth", "G_backup.pth"):
                (root / name).touch()
            self.assertEqual(checkpoint_pair(root), (root / "G_100.pth", root / "E_100.pth"))
            (root / "E_101.pth").touch()
            with self.assertRaises(FileNotFoundError):
                checkpoint_pair(root)
            self.assertEqual(checkpoint_pair(root, step=9), (root / "G_9.pth", root / "E_9.pth"))
            with self.assertRaises(FileNotFoundError):
                checkpoint_pair(root, step=101)

    def test_asset_check_reports_lfs_and_only_requires_selected_language(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=False):
            root = Path(directory)
            os.environ["HARMONIVOX_PRETRAINED_DIR"] = str(root)
            os.environ.pop("HARMONIVOX_CLIP_PATH", None)
            os.environ.pop("HARMONIVOX_XLM_ROBERTA_PATH", None)
            os.environ.pop("HARMONIVOX_NLTK_DIR", None)
            paths = ["config.json", "model/G_1.pth", "model/E_1.pth"]
            paths += [f"bert/deberta-v3-large/{name}" for name in ("config.json", "pytorch_model.bin", "spm.model")]
            paths += ["nltk_data/taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle", "nltk_data/corpora/cmudict/cmudict"]
            for name in paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test asset")
            self.assertEqual(missing_assets(root / "model", root / "config.json", ["EN"]), [])
            weight = root / "bert/deberta-v3-large/pytorch_model.bin"
            weight.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:example\n")
            self.assertTrue(any("Git LFS pointer" in error for error in missing_assets(root / "model", root / "config.json", ["EN"])))

    def test_cleaner_does_not_import_other_language_assets(self):
        result = subprocess.run([sys.executable, "-c", "import sys; import text.cleaner; assert 'text.japanese' not in sys.modules; assert 'text.english' not in sys.modules; assert 'text.chinese' not in sys.modules"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_help_does_not_need_model_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(ROOT / "inference.py"), "--help"], cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--check-only", result.stdout)
            self.assertIn("--checkpoint-step", result.stdout)
            self.assertNotIn("--sdp-ratio", result.stdout)
            self.assertNotIn("--noise-scale-w", result.stdout)

    def test_config_export_preserves_architecture_and_removes_corpus_metadata(self):
        original = {"train": {"segment_size": 16384, "seed": 42, "epochs": 300}, "data": {"sampling_rate": 16000, "n_speakers": 335, "image_dir": "/private/data", "spk2id": {"private-speaker": 0}}, "model": {"hidden_channels": 192, "slm": {"model": "/private/wavlm"}}, "version": "2.3"}
        exported = inference_config(original)
        self.assertEqual(exported["data"]["n_speakers"], 335)
        self.assertEqual(exported["model"]["hidden_channels"], 192)
        self.assertNotIn("private", json.dumps(exported))
        self.assertNotIn("epochs", exported["train"])
        self.assertEqual(original["data"]["spk2id"], {"private-speaker": 0})

    def test_source_archive_excludes_research_and_local_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "inference.zip"
            package(output)
            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
                self.assertIn("HarmoniVox/inference.py", names)
                self.assertIn("HarmoniVox/configs/qformer.json", names)
                self.assertFalse(any("/annotator/" in name or "/train_" in name or name.endswith((".pth", ".pickle", ".pyc")) for name in names))
                for name in names:
                    if name.endswith(".py"):
                        compile(archive.read(name), name, "exec")
            with self.assertRaises(FileExistsError):
                package(output)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for export validation")
    def test_export_keeps_tensors_and_strips_optimizer(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            expected = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
            for prefix in ("G", "E"):
                torch.save({"model": expected, "optimizer": {"state": torch.ones(4)}, "iteration": 31, "learning_rate": 1e-6}, model / f"{prefix}_12000.pth")
            config = root / "original.json"
            config.write_text(json.dumps({"train": {"segment_size": 16384}, "data": {"sampling_rate": 16000}, "model": {"hidden_channels": 192}}))
            output = root / "export"
            manifest = export(model, config, output)
            for prefix in ("G", "E"):
                payload = torch.load(output / "model" / f"{prefix}_12000.pth", weights_only=True)
                self.assertIsNone(payload["optimizer"])
                self.assertTrue(torch.equal(payload["model"]["weight"], expected["weight"]))
            for item in manifest["files"]:
                self.assertEqual(item["sha256"], sha256(output / item["path"]))
            with self.assertRaises(FileExistsError):
                export(model, config, output)

    @unittest.skipUnless(importlib.util.find_spec("open_clip"), "Full inference dependencies are required")
    def test_image_encoder_builds_from_config_without_pretrained_models_or_tokenizers(self):
        import torch
        from types import SimpleNamespace
        from models_qformer_textual_multi import Blip2Qformer

        class Vision(torch.nn.Module):
            def forward(self, image):
                return torch.ones(image.shape[0], 1280), torch.ones(image.shape[0], 2, 1280)

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"hidden_size": 8, "intermediate_size": 16,
                "num_attention_heads": 2, "num_hidden_layers": 2, "vocab_size": 13,
                "max_position_embeddings": 16, "type_vocab_size": 1}))
            with patch("transformers.XLMRobertaModel.from_pretrained", side_effect=AssertionError("No XLM-R weights")), \
                 patch("transformers.AutoTokenizer.from_pretrained", side_effect=AssertionError("No XLM-R tokenizer")), \
                 patch("transformers.BertConfig.from_pretrained", side_effect=AssertionError("Use local config JSON")):
                model = Blip2Qformer(SimpleNamespace(visual=Vision()), num_query_token=2,
                    audio_embed_dim=4, embed_dim=4, qformer_config=config).eval()
                with torch.inference_mode():
                    features = model(torch.zeros(1, 3, 2, 2))
                self.assertEqual(tuple(features.shape), (1, 4))
                self.assertTrue(torch.isfinite(features).all())
                self.assertIsNone(model.tokenizer)

    @unittest.skipUnless(importlib.util.find_spec("open_clip"), "Full inference dependencies are required")
    def test_inference_rejects_incomplete_checkpoint(self):
        import torch
        from infer_web_qformer_textual_multi import load_inference_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "G_1.pth"
            torch.save({"model": {"weight": torch.ones(2, 3)}}, path)
            with self.assertRaises(RuntimeError):
                load_inference_checkpoint(path, torch.nn.Linear(3, 2))


if __name__ == "__main__":
    unittest.main()
