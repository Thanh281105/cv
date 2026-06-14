from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viground.constants import HARD_PAIRS_REPO_PATH, SEED, STANDARD_REPO_PATH
from viground.data import extract_pair_rows, load_dataset_file, row_id
from viground.io_utils import ensure_dir, read_jsonl, write_json
from viground.metrics import iou, prediction_is_valid


MODELS = ("base", "random_ft", "hardpair_ft")


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            predictions[str(record["id"])] = record["prediction"]
    return predictions


def standard_values(rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    values = {}
    for row in rows:
        pred = predictions.get(row_id(row))
        score = 0.0
        if prediction_is_valid(pred):
            score = iou(pred["pixel"][0], row["bbox_xyxy"])
        values[row_id(row)] = {"miou": score, "acc05": float(score >= 0.5)}
    return values


def hard_pair_values(pairs: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    values = {}
    for pair in pairs:
        sample_a, sample_b = extract_pair_rows(pair)
        pred_a = predictions.get(row_id(sample_a))
        pred_b = predictions.get(row_id(sample_b))
        valid_a = prediction_is_valid(pred_a)
        valid_b = prediction_is_valid(pred_b)
        pair_acc = 0.0
        wrong_instance = 0.0
        collapse = 0.0
        pair_miou = 0.0
        if valid_a and valid_b:
            box_a = pred_a["pixel"][0]
            box_b = pred_b["pixel"][0]
            gt_a = sample_a["bbox_xyxy"]
            gt_b = sample_b["bbox_xyxy"]
            own_a = iou(box_a, gt_a)
            own_b = iou(box_b, gt_b)
            pair_acc = float(own_a >= 0.5 and own_b >= 0.5)
            pair_miou = min(own_a, own_b)
            same_image = Path(sample_a["file_name"]).name == Path(sample_b["file_name"]).name
            if same_image:
                wrong_a = iou(box_a, gt_b) > own_a
                wrong_b = iou(box_b, gt_a) > own_b
                wrong_instance = (float(wrong_a) + float(wrong_b)) / 2
                collapse = float(iou(box_a, box_b) >= 0.7 and iou(gt_a, gt_b) < 0.3)
        values[str(pair.get("pair_id"))] = {
            "pair_accuracy": pair_acc,
            "pair_miou": pair_miou,
            "wrong_instance": wrong_instance,
            "same_box_collapse": collapse,
        }
    return values


def paired_bootstrap(a: np.ndarray, b: np.ndarray, seed: int, resamples: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    observed = float(np.mean(b - a))
    if len(a) == 0:
        return {"difference": 0.0, "ci95": [0.0, 0.0]}
    indices = np.arange(len(a))
    diffs = np.empty(resamples, dtype=float)
    for idx in range(resamples):
        sample = rng.choice(indices, size=len(indices), replace=True)
        diffs[idx] = np.mean(b[sample] - a[sample])
    return {
        "difference": round(observed, 6),
        "ci95": [round(float(np.percentile(diffs, 2.5)), 6), round(float(np.percentile(diffs, 97.5)), 6)],
    }


def mcnemar_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    discordant_ab = int(np.sum(a_bool & ~b_bool))
    discordant_ba = int(np.sum(~a_bool & b_bool))
    total = discordant_ab + discordant_ba
    if total == 0:
        return 1.0
    return float(binomtest(min(discordant_ab, discordant_ba), total, 0.5).pvalue)


def compare(
    label: str,
    metric: str,
    old_values: dict[str, dict[str, float]],
    new_values: dict[str, dict[str, float]],
    resamples: int,
    binary: bool,
) -> dict[str, Any]:
    keys = sorted(set(old_values) & set(new_values))
    old = np.array([old_values[key][metric] for key in keys], dtype=float)
    new = np.array([new_values[key][metric] for key in keys], dtype=float)
    result = paired_bootstrap(old, new, SEED, resamples)
    return {
        "comparison": label,
        "metric": metric,
        "n": len(keys),
        "absolute_difference": result["difference"],
        "ci95": result["ci95"],
        "p_value": round(mcnemar_pvalue(old, new), 6) if binary else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired bootstrap and McNemar statistics from real predictions.")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--local-dir", default=".hf_cache/viground")
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    prediction_paths = {model: artifact_root / "evaluation" / model / "predictions.jsonl" for model in MODELS}
    missing = [str(path) for path in prediction_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prediction files: " + ", ".join(missing))

    standard_rows = read_jsonl(load_dataset_file(STANDARD_REPO_PATH, args.local_dir))
    hard_pairs = read_jsonl(load_dataset_file(HARD_PAIRS_REPO_PATH, args.local_dir))
    vi_standard = [row for row in standard_rows if row.get("language") == "vi"]
    predictions = {model: load_predictions(path) for model, path in prediction_paths.items()}
    standard = {model: standard_values(vi_standard, predictions[model]) for model in MODELS}
    hard = {model: hard_pair_values(hard_pairs, predictions[model]) for model in MODELS}

    effects = [
        compare("Random-FT versus Base", "acc05", standard["base"], standard["random_ft"], args.resamples, True),
        compare("HardPair-FT versus Base", "acc05", standard["base"], standard["hardpair_ft"], args.resamples, True),
        compare("HardPair-FT versus Random-FT", "pair_accuracy", hard["random_ft"], hard["hardpair_ft"], args.resamples, True),
        compare("HardPair-FT versus Random-FT", "wrong_instance", hard["random_ft"], hard["hardpair_ft"], args.resamples, False),
        compare("HardPair-FT versus Random-FT", "same_box_collapse", hard["random_ft"], hard["hardpair_ft"], args.resamples, True),
    ]
    output_dir = ensure_dir(artifact_root / "statistics")
    write_json(output_dir / "paired_effects.json", {"resamples": args.resamples, "effects": effects})
    with (output_dir / "paired_effects.csv").open("w", encoding="utf-8") as file:
        file.write("comparison,metric,n,absolute_difference,ci95_low,ci95_high,p_value\n")
        for item in effects:
            low, high = item["ci95"]
            file.write(
                f"{item['comparison']},{item['metric']},{item['n']},{item['absolute_difference']},{low},{high},{item['p_value']}\n"
            )
    print("Wrote statistics:", output_dir)


if __name__ == "__main__":
    main()

