"""Match preserved experiment results to the published MM25 Table 2/3 numbers.

Uses only the standard library. Does not regenerate classification labels or
modify research data. Source PDF is linked in the report, not downloaded here.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


PAPER_URL = "https://hcsi.cs.tsinghua.edu.cn/Paper/paper25/MM2025_Songtao%20Zhou_HarmoniVox_Painting%20Voices%20to%20Match%20the%20Avatar%E2%80%99s%20Soul.pdf"
ATTRIBUTES = ("age", "gender", "pitch", "energy", "emotion", "speed")
EXPECTED_INTRA = {
    "proposed": (56.99, 97.85, 65.59, 65.59, 58.06, 96.77),
    "baseline": (53.76, 100.00, 64.52, 54.84, 44.09, 88.17),
    "without_descriptions": (58.06, 98.92, 63.44, 64.52, 50.54, 96.77),
}
EXPECTED_OUT = {
    "proposed": (23.81, 90.64, 40.39),
    "baseline": (27.91, 90.80, 38.42),
    "without_descriptions": (28.90, 89.98, 37.60),
    "mead": (24.63, 82.10, 35.14),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_age(label):
    # Exact normalization used in analysis_rebuttal_mm25.py.
    return {"Elderly adult": "Elderly", "Young adult": "Young", "Young Adult": "Young"}.get(label, label)


def image_key(path):
    # All preserved out-of-domain samples have category/name.{jpg,wav} keys.
    return "/".join(Path(path).with_suffix(".jpg").parts[-2:])


def out_of_domain(path, truth):
    seen = set()
    correct = Counter()
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        audio, age, gender, emotion = line.split("\t")
        key = image_key(audio)
        if key in seen:
            raise ValueError(f"Duplicate sample in {path}: {key}")
        seen.add(key)
        gt = truth[key]
        correct["age"] += normalize_age(age) == normalize_age(gt["age"]["label"])
        correct["gender"] += gender == gt["gender"]["label"]
        correct["emotion"] += emotion == gt["emotion"]["label"]
        count += 1
    return {
        "path": str(path), "sha256": sha256(path), "samples": count,
        "sample_key_sha256": hashlib.sha256("\n".join(sorted(seen)).encode()).hexdigest(),
        "correct": dict(correct),
        "percent": {key: 100 * correct[key] / count for key in ("age", "gender", "emotion")},
    }


def intra_domain(path, truth_lines, filtered):
    truth = {}
    for fields in truth_lines:
        sample = Path(fields[0]).stem.split("_", 3)[-1]
        if sample in truth:
            raise ValueError(f"Duplicate ground truth: {sample}")
        truth[sample] = fields[1:]
    groups = defaultdict(dict)
    extra = Counter()
    generated_rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        filename = Path(fields[0]).stem
        if filename.startswith("gt_audio_"):
            extra["ground_truth_rows"] += 1
            continue
        match = re.fullmatch(r"gen_audio_\d+_(.+)_(False|True)", filename)
        if not match:
            raise ValueError(f"Unexpected prediction filename: {filename}")
        sample, mode = match.groups()
        if mode in groups[sample]:
            raise ValueError(f"Duplicate sample/mode in {path}: {sample}/{mode}")
        groups[sample][mode] = fields[1:]
        generated_rows.append((sample, mode, fields[1:]))
    selected = sorted(filtered & truth.keys() & groups.keys())
    if len(selected) != len(filtered):
        raise ValueError(f"Filtered set is incomplete in {path}")
    if any(set(groups[key]) != {"False", "True"} for key in selected):
        raise ValueError(f"Missing a prediction mode in {path}")
    correct = {key: 0 for key in ATTRIBUTES}
    for sample in selected:
        for index, key in enumerate(ATTRIBUTES):
            correct[key] += any(row[index] == truth[sample][index] for row in groups[sample].values())
    # Independently replay the original positional pairing for these candidates.
    # This catches silent ordering drift, rather than assuming zip() is safe.
    replay = Counter()
    replay_count = 0
    for index, fields in enumerate(truth_lines):
        sample = Path(fields[0]).stem.split("_", 3)[-1]
        first, second = generated_rows[2 * index:2 * index + 2]
        if first[0] != sample or second[0] != sample:
            raise ValueError(f"Original positional pairing is misaligned in {path}: {sample}")
        if sample not in filtered:
            continue
        replay_count += 1
        for offset, key in enumerate(ATTRIBUTES):
            replay[key] += fields[offset + 1] in (first[2][offset], second[2][offset])
    if replay_count != len(selected) or dict(replay) != correct:
        raise ValueError(f"Keyed scoring disagrees with original positional scoring: {path}")
    return {
        "path": str(path), "sha256": sha256(path), "samples": len(selected),
        "generated_samples": len(groups), "generated_rows": len(generated_rows),
        "ignored_rows": dict(extra), "correct": correct,
        "percent": {key: 100 * value / len(selected) for key, value in correct.items()},
        "protocol": "either of the two False/True predictions is correct; 93-sample filter",
        "legacy_positional_replay_agrees": True,
    }


def matches(computed, expected):
    return all(round(value, 2) == target for value, target in zip(computed, expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True, help="Image2Voice directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.research_root.expanduser().resolve()
    evaluation = root / "Evaluation"
    repo = root / "Bert-VITS2-master"
    metadata_path = repo / "data/real_eval/real_eval.json"
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    truth = {image_key(path): value for path, value in raw.items()}
    if len(truth) != len(raw):
        raise ValueError("Ground-truth image keys are not unique")
    out_results = {path.name: out_of_domain(path, truth) for path in sorted((evaluation / "rebuttal_results").glob("quant*.scp"))}
    gt_path = evaluation / "testset_ground_truth_speed.scp"
    filter_path = evaluation / "testset_filtered_all_acc.json"
    truth_lines = [line.split("\t") for line in gt_path.read_text().splitlines()]
    filtered = set(json.loads(filter_path.read_text()))
    files = {
        "paper_proposed": evaluation / "FINAL_eval_scps/eval_emotional_dp_bn_gan_norm_330_qformer_textual_iam_iac_stage1_wo_ia_add_ia_stage2_unift_12k.speed.scp",
        "paper_baseline": evaluation / "FINAL_eval_scps/eval_emotional_dp_bn_gan_norm_qformer_stage1_uni.speed.scp",
        "paper_without_descriptions": evaluation / "FINAL_eval_scps/eval_emotional_dp_bn_gan_norm_qformer_iac_iam_stage2_uni.speed.scp",
        "experiment_proposed_1e6": evaluation / "rebuttal_results/intra-domain_quant_proposed_1e-6.scp",
        "experiment_without_c_1e6": evaluation / "rebuttal_results/intra-domain_quant_proposed_1e-6_wo_c.scp",
        "experiment_without_m_1e6": evaluation / "rebuttal_results/intra-domain_quant_proposed_1e-6_wo_m.scp",
    }
    intra_results = {key: intra_domain(path, truth_lines, filtered) for key, path in files.items()}
    main_intra = EXPECTED_INTRA["proposed"]
    main_out = EXPECTED_OUT["proposed"]
    matching_intra = [key for key, result in intra_results.items() if matches(list(result["percent"].values()), main_intra)]
    matching_out = [key for key, result in out_results.items() if matches(list(result["percent"].values()), main_out)]
    anchors = {}
    for key, expected in EXPECTED_INTRA.items():
        anchors[f"intra_{key}"] = matches(list(intra_results[f"paper_{key}"]["percent"].values()), expected)
    for key, name in (("proposed", "quant_proposed.scp"), ("baseline", "quant_baseline.scp"), ("without_descriptions", "quant_proposed_wo_text.scp"), ("mead", "quant_mead.scp")):
        anchors[f"out_{key}"] = matches(list(out_results[name]["percent"].values()), EXPECTED_OUT[key])
    if not all(anchors.values()):
        raise ValueError(f"Control rows do not reproduce the published table: {anchors}")
    result = {
        "paper_url": PAPER_URL, "paper_table_pages": [7, 8],
        "ground_truth_sha256": {str(path): sha256(path) for path in (metadata_path, gt_path, filter_path)},
        "paper_expected_intra": dict(zip(ATTRIBUTES, main_intra)),
        "paper_expected_out": dict(zip(("age", "gender", "emotion"), main_out)),
        "out_of_domain": out_results, "intra_domain": intra_results,
        "matching_published_proposed_intra": matching_intra,
        "matching_published_proposed_out": matching_out,
        "control_rows_match": anchors,
        "conclusion": "Published Table 2/3 matches paper_used/proposed family, not the later 1e-6 experiment",
        "limits": [
            "Replays preserved classification labels; does not rerun the audio classifiers.",
            "Original intra-domain audio files are absent; no bitwise regeneration proof.",
            "Historical generation did not record checkpoint hashes; checkpoint linkage also uses experiment paths and prior checkpoint identity audit.",
            "MCD and subjective MOS scores are not recomputed.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Published proposed matching intra-domain result:", matching_intra)
    print("Published proposed matching out-of-domain result:", matching_out)
    print("All baseline / no-description / MEAD control rows match:", all(anchors.values()))
    for name in ("paper_proposed", "experiment_proposed_1e6"):
        print(name, ", ".join(f"{key}={value:.2f}" for key, value in intra_results[name]["percent"].items()))
    for name in ("quant_proposed.scp", "quant_ablation_proposed_1e-6.scp"):
        print(name, ", ".join(f"{key}={value:.2f}" for key, value in out_results[name]["percent"].items()))
    print("Report:", args.output)


if __name__ == "__main__":
    main()
