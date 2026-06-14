from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from .constants import (
    DATASET_REVISION,
    EXPECTED_COUNTS,
    HARD_PAIRS_REPO_PATH,
    HF_DATASET_ID,
    IMAGE_EXTENSIONS,
    IMAGES_ZIP_REPO_PATH,
    LOCANY_HARDPAIR_TRAIN_REPO_PATH,
    LOCANY_RANDOM_TRAIN_REPO_PATH,
    MODEL_REVISION,
    REQUIRED_REPO_FILES,
    STANDARD_REPO_PATH,
    TRAIN_HARDPAIR_PAIRS_REPO_PATH,
    TRAIN_HARDPAIR_REPO_PATH,
    TRAIN_RANDOM_REPO_PATH,
)
from .io_utils import capture_environment, ensure_dir, hf_token, read_jsonl, write_json, write_text


ALIGNMENT_KEY_CANDIDATES = (
    "sample_id",
    "source_sample_id",
    "source_id",
    "expression_id",
    "ref_id",
)


def row_id(row: dict[str, Any]) -> str:
    benchmark_id = row.get("benchmark_id")
    if benchmark_id is not None:
        return str(benchmark_id)
    sample_id = row.get("sample_id")
    if sample_id is None:
        raise KeyError(f"Row has neither benchmark_id nor sample_id: {sorted(row)}")
    language = row.get("language")
    if language:
        return f"{sample_id}::{language}"
    return str(sample_id)


def extract_pair_rows(pair: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    for key_a, key_b in (("sample_a", "sample_b"), ("a", "b"), ("left", "right")):
        a = pair.get(key_a)
        b = pair.get(key_b)
        if isinstance(a, dict) and isinstance(b, dict):
            return a, b
    raise ValueError(f"Unsupported hard-pair schema: {sorted(pair)}")


def choose_alignment_key(en_rows: list[dict[str, Any]], vi_rows: list[dict[str, Any]]) -> tuple[str, int]:
    best_key = ""
    best_overlap = 0
    for key in ALIGNMENT_KEY_CANDIDATES:
        en_values = {str(row[key]) for row in en_rows if row.get(key) is not None}
        vi_values = {str(row[key]) for row in vi_rows if row.get(key) is not None}
        overlap = len(en_values & vi_values)
        if overlap > best_overlap:
            best_key = key
            best_overlap = overlap
    if not best_key:
        raise RuntimeError("Could not align English and Vietnamese benchmark rows")
    return best_key, best_overlap


def load_dataset_file(repo_path: str, local_dir: str | Path) -> Path:
    path = hf_hub_download(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        filename=repo_path,
        local_dir=str(local_dir),
        token=hf_token(),
    )
    return Path(path)


def repository_manifest() -> dict[str, Any]:
    api = HfApi(token=hf_token())
    info = api.dataset_info(
        repo_id=HF_DATASET_ID,
        revision=DATASET_REVISION,
        files_metadata=True,
    )
    files = []
    for sibling in sorted(info.siblings, key=lambda item: item.rfilename):
        files.append(
            {
                "path": sibling.rfilename,
                "size": getattr(sibling, "size", None),
                "blob_id": getattr(sibling, "blob_id", None),
                "lfs": getattr(sibling, "lfs", None),
            }
        )
    return {
        "dataset_id": HF_DATASET_ID,
        "requested_revision": DATASET_REVISION,
        "resolved_revision": info.sha,
        "files": files,
    }


def _image_candidates(row: dict[str, Any]) -> list[str]:
    candidates = []
    for key in ("image", "file_name", "image_path"):
        value = row.get(key)
        if value:
            candidates.append(str(value))
    return candidates


def _normalize_zip_name(name: str) -> str:
    return name.replace("\\", "/").lstrip("./")


def _build_zip_image_index(zip_path: Path) -> tuple[set[str], dict[str, list[str]]]:
    relative_index: set[str] = set()
    basename_index: dict[str, list[str]] = defaultdict(list)
    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            normalized = _normalize_zip_name(info.filename)
            if Path(normalized).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative_index.add(normalized)
            basename_index[Path(normalized).name].append(normalized)
    return relative_index, basename_index


def _resolve_image(candidates: list[str], relative_index: set[str], basename_index: dict[str, list[str]]) -> str | None:
    for candidate in candidates:
        normalized = _normalize_zip_name(candidate)
        if normalized in relative_index:
            return normalized
        parts = normalized.split("/")
        for start in range(1, len(parts)):
            suffix = "/".join(parts[start:])
            if suffix in relative_index:
                return suffix
        basename = Path(normalized).name
        matches = basename_index.get(basename, [])
        if len(matches) == 1:
            return matches[0]
    return None


def _collect_referenced_rows(
    standard_rows: list[dict[str, Any]],
    benchmark_pairs: list[dict[str, Any]],
    random_train_rows: list[dict[str, Any]],
    hardpair_train_rows: list[dict[str, Any]],
    locany_random_rows: list[dict[str, Any]],
    locany_hardpair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    rows.extend(standard_rows)
    rows.extend(random_train_rows)
    rows.extend(hardpair_train_rows)
    rows.extend(locany_random_rows)
    rows.extend(locany_hardpair_rows)
    for pair in benchmark_pairs:
        a, b = extract_pair_rows(pair)
        rows.extend([a, b])
    return rows


def _count_check(name: str, actual: int) -> dict[str, Any]:
    expected = EXPECTED_COUNTS[name]
    return {"name": name, "actual": actual, "expected": expected, "ok": actual == expected}


def audit_dataset(
    artifact_root: str | Path = "artifacts",
    local_dir: str | Path = ".hf_cache/viground",
    verify_images: bool = True,
) -> dict[str, Any]:
    artifact_root = Path(artifact_root)
    local_dir = Path(local_dir)
    frozen_dir = ensure_dir(artifact_root / "frozen_revisions")
    audit_dir = ensure_dir(artifact_root / "dataset_audit")

    manifest = repository_manifest()
    repo_files = {item["path"] for item in manifest["files"]}
    missing_repo_files = [path for path in REQUIRED_REPO_FILES if path not in repo_files]

    write_text(frozen_dir / "dataset_revision.txt", manifest["resolved_revision"] + "\n")
    write_text(frozen_dir / "model_revision.txt", MODEL_REVISION + "\n")
    write_json(frozen_dir / "environment.json", capture_environment())
    write_json(frozen_dir / "repository_file_manifest.json", manifest)

    paths = {
        "standard": load_dataset_file(STANDARD_REPO_PATH, local_dir),
        "benchmark_hard_pairs": load_dataset_file(HARD_PAIRS_REPO_PATH, local_dir),
        "random_train": load_dataset_file(TRAIN_RANDOM_REPO_PATH, local_dir),
        "hardpair_train": load_dataset_file(TRAIN_HARDPAIR_REPO_PATH, local_dir),
        "hardpair_train_pairs": load_dataset_file(TRAIN_HARDPAIR_PAIRS_REPO_PATH, local_dir),
        "locany_random_train": load_dataset_file(LOCANY_RANDOM_TRAIN_REPO_PATH, local_dir),
        "locany_hardpair_train": load_dataset_file(LOCANY_HARDPAIR_TRAIN_REPO_PATH, local_dir),
    }

    standard_rows = read_jsonl(paths["standard"])
    benchmark_pairs = read_jsonl(paths["benchmark_hard_pairs"])
    random_train_rows = read_jsonl(paths["random_train"])
    hardpair_train_rows = read_jsonl(paths["hardpair_train"])
    hardpair_train_pairs = read_jsonl(paths["hardpair_train_pairs"])
    locany_random_rows = read_jsonl(paths["locany_random_train"])
    locany_hardpair_rows = read_jsonl(paths["locany_hardpair_train"])

    language_counts = Counter(str(row.get("language")) for row in standard_rows)
    count_checks = [
        _count_check("random_ft_train_samples", len(random_train_rows)),
        _count_check("hardpair_ft_train_samples", len(hardpair_train_rows)),
        _count_check("hardpair_ft_train_pairs", len(hardpair_train_pairs)),
        _count_check("standard_benchmark_rows", len(standard_rows)),
        _count_check("standard_benchmark_english_rows", language_counts.get("en", 0)),
        _count_check("standard_benchmark_vietnamese_rows", language_counts.get("vi", 0)),
        _count_check("hard_pair_benchmark_pairs", len(benchmark_pairs)),
    ]

    image_audit: dict[str, Any] = {"verified": False}
    warnings = []
    if verify_images:
        images_zip = load_dataset_file(IMAGES_ZIP_REPO_PATH, local_dir)
        relative_index, basename_index = _build_zip_image_index(images_zip)
        referenced_rows = _collect_referenced_rows(
            standard_rows,
            benchmark_pairs,
            random_train_rows,
            hardpair_train_rows,
            locany_random_rows,
            locany_hardpair_rows,
        )
        missing_images = []
        for row in referenced_rows:
            candidates = _image_candidates(row)
            if candidates and _resolve_image(candidates, relative_index, basename_index) is None:
                missing_images.append({"id": row.get("sample_id") or row.get("benchmark_id"), "candidates": candidates})

        image_count = len(relative_index)
        image_count_check = _count_check("fixed_bundle_images", image_count)
        if not image_count_check["ok"]:
            warnings.append(
                {
                    "name": "fixed_bundle_images",
                    "message": "Image bundle count differs from the plan text; all referenced images are still required to resolve.",
                    "actual": image_count_check["actual"],
                    "expected": image_count_check["expected"],
                }
            )
        image_audit = {
            "verified": True,
            "zip_path": str(images_zip),
            "image_count": image_count,
            "count_check": image_count_check,
            "referenced_rows_checked": len(referenced_rows),
            "missing_references_count": len(missing_images),
            "missing_reference_examples": missing_images[:20],
        }

    standard_categories = {str(row.get("category_name")) for row in standard_rows}
    random_categories = {str(row.get("category_name")) for row in random_train_rows}
    hardpair_categories = {str(row.get("category_name")) for row in hardpair_train_rows}
    benchmark_pair_samples = []
    for pair in benchmark_pairs:
        a, b = extract_pair_rows(pair)
        benchmark_pair_samples.extend([a, b])

    report = {
        "dataset_id": HF_DATASET_ID,
        "requested_revision": DATASET_REVISION,
        "resolved_revision": manifest["resolved_revision"],
        "model_revision": MODEL_REVISION,
        "required_files_missing": missing_repo_files,
        "counts": {
            "standard_rows": len(standard_rows),
            "standard_language_counts": dict(language_counts),
            "benchmark_hard_pairs": len(benchmark_pairs),
            "random_train_rows": len(random_train_rows),
            "hardpair_train_rows": len(hardpair_train_rows),
            "hardpair_train_pairs": len(hardpair_train_pairs),
            "locany_random_train_rows": len(locany_random_rows),
            "locany_hardpair_train_rows": len(locany_hardpair_rows),
        },
        "category_counts": {
            "standard_unique_categories": len(standard_categories),
            "random_train_unique_categories": len(random_categories),
            "hardpair_train_unique_categories": len(hardpair_categories),
            "benchmark_hard_pair_unique_categories": len({str(pair.get("category_name")) for pair in benchmark_pairs}),
        },
        "unique_images": {
            "standard": len(_unique_image_names(standard_rows)),
            "random_train": len(_unique_image_names(random_train_rows)),
            "hardpair_train": len(_unique_image_names(hardpair_train_rows)),
            "benchmark_hard_pairs": len(_unique_image_names(benchmark_pair_samples)),
        },
        "translation_valid_rates": {
            "standard": _translation_valid_rate(standard_rows),
            "random_train": _translation_valid_rate(random_train_rows),
            "hardpair_train": _translation_valid_rate(hardpair_train_rows),
        },
        "count_checks": count_checks,
        "image_audit": image_audit,
        "warnings": warnings,
    }
    report["ok"] = (
        not missing_repo_files
        and all(check["ok"] for check in count_checks)
        and (not verify_images or image_audit["missing_references_count"] == 0)
    )
    write_json(audit_dir / "dataset_integrity_report.json", report)
    if not report["ok"]:
        raise RuntimeError("Dataset audit failed; see artifacts/dataset_audit/dataset_integrity_report.json")
    return report


def _translation_valid_rate(rows: list[dict[str, Any]]) -> float | None:
    values = [row.get("translation_valid") for row in rows if row.get("translation_valid") is not None]
    if not values:
        return None
    return round(sum(bool(value) for value in values) / len(values), 6)


def _unique_image_names(rows: list[dict[str, Any]]) -> set[str]:
    names = set()
    for row in rows:
        candidates = _image_candidates(row)
        if candidates:
            names.add(Path(candidates[0]).name)
    return names
