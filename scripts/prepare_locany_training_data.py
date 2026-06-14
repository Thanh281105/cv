from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm.auto import tqdm

from viground.constants import (
    IMAGE_EXTENSIONS,
    IMAGES_ZIP_REPO_PATH,
    LOCANY_HARDPAIR_TRAIN_REPO_PATH,
    LOCANY_RANDOM_TRAIN_REPO_PATH,
)
from viground.data import load_dataset_file
from viground.io_utils import ensure_dir, read_jsonl, write_json


TRAIN_FILES = {
    "random_ft": LOCANY_RANDOM_TRAIN_REPO_PATH,
    "hardpair_ft": LOCANY_HARDPAIR_TRAIN_REPO_PATH,
}


def extract_images(zip_path: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extract_complete"
    if marker.exists():
        return
    if extract_dir.exists():
        for child in extract_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    ensure_dir(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in tqdm(archive.infolist(), desc="Extracting images"):
            archive.extract(member, extract_dir)
    marker.touch()


def build_image_index(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    relative_index = {}
    basename_index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root).as_posix().lstrip("./")
        relative_index[relative] = path
        basename_index.setdefault(path.name, []).append(path)
    return relative_index, basename_index


def resolve_image(raw_name: str, root: Path, relative_index: dict[str, Path], basename_index: dict[str, list[Path]]) -> Path | None:
    normalized = raw_name.replace("\\", "/").lstrip("./")
    if normalized in relative_index:
        return relative_index[normalized]
    parts = normalized.split("/")
    for start in range(1, len(parts)):
        suffix = "/".join(parts[start:])
        if suffix in relative_index:
            return relative_index[suffix]
    matches = basename_index.get(Path(normalized).name, [])
    if len(matches) == 1:
        return matches[0]
    direct = root / normalized
    return direct if direct.exists() else None


def write_resolved_jsonl(rows: list[dict[str, Any]], images_dir: Path, output_path: Path) -> dict[str, Any]:
    relative_index, basename_index = build_image_index(images_dir)
    missing = []
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            image = row.get("image")
            if not image:
                missing.append({"sample_id": row.get("sample_id"), "reason": "missing image field"})
                continue
            resolved = resolve_image(str(image), images_dir, relative_index, basename_index)
            if resolved is None:
                missing.append({"sample_id": row.get("sample_id"), "image": image})
                continue
            item = dict(row)
            item["image"] = resolved.relative_to(images_dir).as_posix()
            file.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    return {"input_rows": len(rows), "missing_rows": missing, "output_path": str(output_path)}


def prepare_training_data(kind: str, work_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    work_dir = Path(work_dir)
    output_dir = Path(output_dir)
    data_dir = ensure_dir(work_dir / "dataset")
    images_dir = ensure_dir(data_dir / "images_extracted")
    train_path = load_dataset_file(TRAIN_FILES[kind], data_dir)
    images_zip = load_dataset_file(IMAGES_ZIP_REPO_PATH, data_dir)
    extract_images(images_zip, images_dir)
    rows = read_jsonl(train_path)
    resolved_train_path = output_dir / f"{kind}_locateanything_train.resolved.jsonl"
    resolve_report = write_resolved_jsonl(rows, images_dir, resolved_train_path)
    if resolve_report["missing_rows"]:
        write_json(output_dir / "skipped_samples.json", resolve_report["missing_rows"])
        raise RuntimeError(f"Missing images for {len(resolve_report['missing_rows'])} training rows")

    meta_path = output_dir / f"{kind}_eagle_meta.json"
    meta = {
        kind: {
            "root": str(images_dir.resolve()),
            "annotation": [str(resolved_train_path.resolve())],
            "repeat_time": 1.0,
            "data_augment": False,
        }
    }
    write_json(meta_path, meta)
    report = {
        "kind": kind,
        "train_repo_path": TRAIN_FILES[kind],
        "rows": len(rows),
        "images_dir": str(images_dir),
        "resolved_train_path": str(resolved_train_path),
        "eagle_meta_path": str(meta_path),
    }
    write_json(output_dir / "prepared_training_data.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LocateAnything JSONL and Eagle meta_path for training.")
    parser.add_argument("--kind", choices=sorted(TRAIN_FILES), required=True)
    parser.add_argument("--work-dir", default=".hf_cache/viground_train")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or f"artifacts/training/{args.kind}"
    report = prepare_training_data(args.kind, args.work_dir, output_dir)
    print("Prepared training data:", report["eagle_meta_path"])


if __name__ == "__main__":
    main()
