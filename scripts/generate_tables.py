from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viground.constants import BASE_PILOT_RESULT
from viground.io_utils import ensure_dir


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def table_dataset(audit: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not audit:
        return []
    counts = audit["counts"]
    unique_images = audit.get("unique_images", {})
    return [
        {"split": "train_random", "samples": counts["random_train_rows"], "pairs": "", "unique_images": unique_images.get("random_train", ""), "unique_categories": audit["category_counts"]["random_train_unique_categories"], "language": "vi", "translation_valid_rate": audit["translation_valid_rates"]["random_train"]},
        {"split": "train_hardpair", "samples": counts["hardpair_train_rows"], "pairs": counts["hardpair_train_pairs"], "unique_images": unique_images.get("hardpair_train", ""), "unique_categories": audit["category_counts"]["hardpair_train_unique_categories"], "language": "vi", "translation_valid_rate": audit["translation_valid_rates"]["hardpair_train"]},
        {"split": "standard_benchmark", "samples": counts["standard_rows"], "pairs": "", "unique_images": unique_images.get("standard", ""), "unique_categories": audit["category_counts"]["standard_unique_categories"], "language": "en,vi", "translation_valid_rate": audit["translation_valid_rates"]["standard"]},
        {"split": "hard_pair_benchmark", "samples": counts["benchmark_hard_pairs"] * 2, "pairs": counts["benchmark_hard_pairs"], "unique_images": unique_images.get("benchmark_hard_pairs", ""), "unique_categories": audit["category_counts"].get("benchmark_hard_pair_unique_categories", ""), "language": "en,vi", "translation_valid_rate": ""},
    ]


def table_pilot() -> list[dict[str, Any]]:
    return [
        {"split": "English", **BASE_PILOT_RESULT["english"]},
        {"split": "Vietnamese", **BASE_PILOT_RESULT["vietnamese"]},
        {"split": "Vietnamese hard pairs", **BASE_PILOT_RESULT["hard_pairs"]},
    ]


def table_standard(artifact_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in ("base", "random_ft", "hardpair_ft"):
        metrics = load_json(artifact_root / "evaluation" / model / "standard_metrics.json")
        if not metrics:
            continue
        for language_key, language in (("english", "en"), ("vietnamese", "vi")):
            item = metrics[language_key]
            rows.append(
                {
                    "model": model,
                    "language": language,
                    "n": item["n"],
                    "mIoU": item["mIoU"],
                    "Acc@0.5": item["Acc@0.5"],
                    "Acc@0.75": item["Acc@0.75"],
                    "parse_failure": item["Parse Fail"],
                    "multi_box_failure": item["Multi-box"],
                    "grounding_gap": metrics["vietnamese_grounding_gap"] if language == "vi" else "",
                }
            )
    return rows


def table_hard_pair(artifact_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in ("base", "random_ft", "hardpair_ft"):
        metrics = load_json(artifact_root / "evaluation" / model / "hard_pair_metrics.json")
        if not metrics:
            continue
        for language in ("vi", "en"):
            item = metrics[language]
            rows.append(
                {
                    "model": model,
                    "language": language,
                    "pairs": item["pairs"],
                    "query_acc05": item["Query Acc@0.5"],
                    "pair_accuracy": item["Pair Accuracy"],
                    "pair_miou": item["Pair mIoU"],
                    "wrong_instance": item["Wrong-Instance"],
                    "same_box_collapse": item["Same-Box Collapse"],
                    "parse_failure": item["Parse Fail"],
                }
            )
    return rows


def table_training(artifact_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in ("random_ft", "hardpair_ft"):
        prepared = load_json(artifact_root / "training" / model / "prepared_training_data.json") or {}
        trainable = load_json(artifact_root / "training" / model / "trainable_parameters.json") or {}
        rows.append(
            {
                "model": model,
                "train_samples": prepared.get("rows", ""),
                "trainable_parameters": trainable.get("trainable_parameters", ""),
                "optimizer_steps": "",
                "training_loss": "",
                "runtime": "",
                "peak_vram": "",
                "checkpoint_size": "",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate required ViGround result tables from available real artifacts.")
    parser.add_argument("--artifact-root", default="artifacts")
    args = parser.parse_args()
    artifact_root = Path(args.artifact_root)
    table_dir = ensure_dir(artifact_root / "tables")

    audit = load_json(artifact_root / "dataset_audit" / "dataset_integrity_report.json")
    write_csv(table_dir / "table1_dataset.csv", table_dataset(audit), ["split", "samples", "pairs", "unique_images", "unique_categories", "language", "translation_valid_rate"])
    write_csv(table_dir / "table2_base_pilot.csv", table_pilot(), ["split", "mIoU", "Acc@0.5", "Acc@0.75", "Parse Fail", "Multi-box", "Pair Accuracy", "Pair mIoU", "Wrong-Instance", "Same-Box Collapse", "Runtime Error", "Cross-image Pairs"])
    write_csv(table_dir / "table3_full_standard_benchmark.csv", table_standard(artifact_root), ["model", "language", "n", "mIoU", "Acc@0.5", "Acc@0.75", "parse_failure", "multi_box_failure", "grounding_gap"])
    write_csv(table_dir / "table4_full_hard_pair_benchmark.csv", table_hard_pair(artifact_root), ["model", "language", "pairs", "query_acc05", "pair_accuracy", "pair_miou", "wrong_instance", "same_box_collapse", "parse_failure"])
    effects = load_json(artifact_root / "statistics" / "paired_effects.json")
    effect_rows = effects["effects"] if effects else []
    write_csv(table_dir / "table5_paired_effects.csv", effect_rows, ["comparison", "metric", "n", "absolute_difference", "ci95", "p_value"])
    write_csv(table_dir / "table6_training.csv", table_training(artifact_root), ["model", "train_samples", "trainable_parameters", "optimizer_steps", "training_loss", "runtime", "peak_vram", "checkpoint_size"])
    print("Wrote tables:", table_dir)


if __name__ == "__main__":
    main()
