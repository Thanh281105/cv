from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from tqdm.auto import tqdm

from scripts.evaluate_baselines import (
    ImageResolver,
    box_token,
    ensure_images_extracted,
    flatten_hard_pairs,
    make_hard_pair_language_view,
    parse_qwen_box,
)
from viground.constants import HARD_PAIRS_REPO_PATH, IMAGES_ZIP_REPO_PATH, STANDARD_REPO_PATH
from viground.data import choose_alignment_key, load_dataset_file, row_id
from viground.io_utils import ensure_dir, read_jsonl, write_json
from viground.metrics import evaluate_hard_pairs, evaluate_standard, latency_stats


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            predictions[str(record["id"])] = record["prediction"]
    return predictions


def write_predictions(path: Path, predictions: dict[str, dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for identifier, prediction in predictions.items():
            file.write(json.dumps({"id": identifier, "prediction": prediction}, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair Qwen2.5-VL parsed boxes by reinterpreting pixel-coordinate outputs.")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--work-dir", default=".hf_cache/viground_baselines")
    parser.add_argument("--model-key", default="qwen2_5_vl_3b")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    output_dir = artifact_root / "evaluation" / args.model_key
    prediction_path = output_dir / "predictions.jsonl"
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)

    work_dir = ensure_dir(args.work_dir)
    data_dir = ensure_dir(work_dir / "dataset")
    images_dir = ensure_dir(data_dir / "images_extracted")
    standard_path = load_dataset_file(STANDARD_REPO_PATH, data_dir)
    hard_pairs_path = load_dataset_file(HARD_PAIRS_REPO_PATH, data_dir)
    images_zip_path = load_dataset_file(IMAGES_ZIP_REPO_PATH, data_dir)
    ensure_images_extracted(images_zip_path, images_dir)

    standard_rows = read_jsonl(standard_path)
    hard_pairs = read_jsonl(hard_pairs_path)
    en_rows = [row for row in standard_rows if row.get("language") == "en"]
    vi_rows = [row for row in standard_rows if row.get("language") == "vi"]
    alignment_key, overlap = choose_alignment_key(en_rows, vi_rows)
    en_rows = sorted(en_rows, key=lambda row: str(row[alignment_key]))
    vi_rows = sorted(vi_rows, key=lambda row: str(row[alignment_key]))
    hard_pairs_vi = make_hard_pair_language_view(hard_pairs, "vi")
    hard_pairs_en = make_hard_pair_language_view(hard_pairs, "en")

    row_index = {}
    for block in (en_rows, vi_rows, flatten_hard_pairs(hard_pairs_vi), flatten_hard_pairs(hard_pairs_en)):
        for row in block:
            row_index[row_id(row)] = row

    resolver = ImageResolver(images_dir)
    predictions = read_predictions(prediction_path)
    backup_path = output_dir / "predictions.before_qwen_repair.jsonl"
    if not backup_path.exists():
        shutil.copy2(prediction_path, backup_path)

    changed = 0
    still_failed = 0
    for identifier, prediction in tqdm(predictions.items(), desc="Repairing Qwen predictions"):
        row = row_index.get(identifier)
        if row is None:
            continue
        raw = str(prediction.get("model_raw") or prediction.get("raw") or "")
        image_path = resolver.resolve(row)
        with Image.open(image_path) as image:
            normalized, pixel, parse_ok = parse_qwen_box(raw, image.width, image.height)
        old_norm = prediction.get("norm")
        prediction["norm"] = normalized
        prediction["pixel"] = pixel
        prediction["parse_ok"] = parse_ok
        if parse_ok and normalized:
            prediction["raw"] = box_token(normalized[0])
        if old_norm != normalized:
            changed += 1
        if not parse_ok:
            still_failed += 1

    write_predictions(prediction_path, predictions)
    standard_metrics = {
        "model": args.model_key,
        "english": evaluate_standard(en_rows, predictions),
        "vietnamese": evaluate_standard(vi_rows, predictions),
    }
    standard_metrics["vietnamese_grounding_gap"] = round(
        standard_metrics["english"]["Acc@0.5"] - standard_metrics["vietnamese"]["Acc@0.5"],
        4,
    )
    hard_pair_metrics = {
        "model": args.model_key,
        "vi": evaluate_hard_pairs(hard_pairs_vi, predictions),
        "en": evaluate_hard_pairs(hard_pairs_en, predictions),
    }
    failure_cases = [
        {"id": identifier, "prediction": prediction}
        for identifier, prediction in predictions.items()
        if prediction.get("error") or not prediction.get("parse_ok") or len(prediction.get("pixel", [])) != 1
    ]
    write_json(output_dir / "standard_metrics.json", standard_metrics)
    write_json(output_dir / "hard_pair_metrics.json", hard_pair_metrics)
    write_json(output_dir / "latency_stats.json", latency_stats(predictions))
    with (output_dir / "failure_cases.jsonl").open("w", encoding="utf-8") as file:
        for record in failure_cases:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(
        output_dir / "qwen_repair_report.json",
        {
            "changed_predictions": changed,
            "still_parse_failed": still_failed,
            "backup": str(backup_path),
            "standard_metrics": standard_metrics,
            "hard_pair_metrics": hard_pair_metrics,
        },
    )
    print("Repaired Qwen predictions:", output_dir)
    print("Changed predictions:", changed)
    print("Still parse failed:", still_failed)


if __name__ == "__main__":
    main()
