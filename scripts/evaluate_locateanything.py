from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from viground.constants import (
    DATASET_REVISION,
    DO_SAMPLE,
    GENERATION_MODE,
    HARD_PAIRS_REPO_PATH,
    HF_DATASET_ID,
    IMAGE_EXTENSIONS,
    IMAGES_ZIP_REPO_PATH,
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_OUTPUT_DIRS,
    MODEL_REVISION,
    PROMPT_TEMPLATE,
    SEED,
    STANDARD_REPO_PATH,
)
from viground.data import choose_alignment_key, extract_pair_rows, load_dataset_file, row_id
from viground.io_utils import append_jsonl, capture_environment, ensure_dir, hf_token, read_jsonl, write_json
from viground.metrics import evaluate_hard_pairs, evaluate_standard, latency_stats, parse_boxes


class ImageResolver:
    def __init__(self, extract_dir: str | Path):
        self.extract_dir = Path(extract_dir)
        self.relative_index: dict[str, Path] = {}
        self.basename_index: dict[str, list[Path]] = {}
        for path in self.extract_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative_key = path.relative_to(self.extract_dir).as_posix().lstrip("./")
            self.relative_index[relative_key] = path
            self.basename_index.setdefault(path.name, []).append(path)

    def resolve(self, row: dict[str, Any]) -> Path:
        candidates = [str(row[key]) for key in ("file_name", "image", "image_path") if row.get(key)]
        for candidate in candidates:
            normalized = candidate.replace("\\", "/").lstrip("./")
            exact = self.relative_index.get(normalized)
            if exact is not None:
                return exact
            parts = normalized.split("/")
            for start in range(1, len(parts)):
                suffix = "/".join(parts[start:])
                match = self.relative_index.get(suffix)
                if match is not None:
                    return match
            basename = Path(normalized).name
            matches = self.basename_index.get(basename, [])
            if len(matches) == 1:
                return matches[0]
        raise FileNotFoundError(f"Image not found for row {row.get('sample_id') or row.get('benchmark_id')}: {candidates}")


class LocateAnythingBenchmarkWorker:
    def __init__(self, model_path: str | Path, adapter_path: str | None, device: str, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model = self.model.to(device).eval()

    @staticmethod
    def build_prompt(query: str) -> str:
        return PROMPT_TEMPLATE.format(query=query)

    def extract_answer(self, response: Any) -> str:
        candidate = response[0] if isinstance(response, tuple) else response
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, (list, tuple)):
            if not candidate:
                return ""
            if isinstance(candidate[0], str):
                return candidate[0]
            candidate = candidate[0]
        if torch.is_tensor(candidate):
            if candidate.ndim == 1:
                candidate = candidate.unsqueeze(0)
            return self.tokenizer.batch_decode(candidate, skip_special_tokens=False)[0]
        return str(candidate)

    @torch.inference_mode()
    def predict(self, image: Image.Image, query: str) -> dict[str, Any]:
        image = image.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.build_prompt(query)},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)
        generation_kwargs = {
            "pixel_values": inputs["pixel_values"].to(self.dtype),
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "image_grid_hws": inputs.get("image_grid_hws", None),
            "tokenizer": self.tokenizer,
            "max_new_tokens": MAX_NEW_TOKENS,
            "use_cache": True,
            "generation_mode": GENERATION_MODE,
            "do_sample": DO_SAMPLE,
            "verbose": False,
        }
        started = time.perf_counter()
        response = self.model.generate(**generation_kwargs)
        latency_ms = (time.perf_counter() - started) * 1000
        return {"answer": self.extract_answer(response), "latency_ms": latency_ms}


def load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    predictions: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return predictions
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            predictions[str(record["id"])] = record["prediction"]
    return predictions


def row_query(row: dict[str, Any]) -> str:
    return str(row.get("query") or row.get("expression_vi") or row.get("expression_en"))


def predict_one(worker: LocateAnythingBenchmarkWorker, resolver: ImageResolver, row: dict[str, Any]) -> dict[str, Any]:
    try:
        image_path = resolver.resolve(row)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        prediction = worker.predict(image=image, query=row_query(row))
        normalized, pixel, parse_ok = parse_boxes(prediction["answer"], image.width, image.height)
        return {
            "raw": prediction["answer"],
            "norm": normalized,
            "pixel": pixel,
            "parse_ok": parse_ok,
            "latency_ms": prediction["latency_ms"],
            "image_width": image.width,
            "image_height": image.height,
            "error": None,
        }
    except Exception as exc:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "raw": "",
            "norm": [],
            "pixel": [],
            "parse_ok": False,
            "latency_ms": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def predict_rows(
    worker: LocateAnythingBenchmarkWorker,
    resolver: ImageResolver,
    rows: list[dict[str, Any]],
    checkpoint_path: str | Path,
    description: str,
    retry_errors: bool,
) -> dict[str, dict[str, Any]]:
    predictions = load_predictions(checkpoint_path)
    pending = []
    for row in rows:
        identifier = row_id(row)
        old_prediction = predictions.get(identifier)
        should_retry = retry_errors and old_prediction is not None and old_prediction.get("error")
        if old_prediction is None or should_retry:
            pending.append(row)
    print(f"{description}: total={len(rows)}, available={len(predictions)}, pending={len(pending)}")
    for row in tqdm(pending, desc=description):
        identifier = row_id(row)
        prediction = predict_one(worker, resolver, row)
        predictions[identifier] = prediction
        append_jsonl(checkpoint_path, {"id": identifier, "prediction": prediction})
    return predictions


def ensure_images_extracted(zip_path: Path, extract_dir: Path) -> None:
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


def make_hard_pair_language_view(pairs: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    if language == "vi":
        return pairs
    converted = []
    for pair in pairs:
        item = dict(pair)
        sample_a, sample_b = extract_pair_rows(pair)
        new_a = dict(sample_a)
        new_b = dict(sample_b)
        for sample in (new_a, new_b):
            sample["language"] = "en"
            sample["query"] = sample.get("expression_en")
            if sample.get("sample_id"):
                sample["benchmark_id"] = f"{sample['sample_id']}:hardpair_en"
        item["sample_a"] = new_a
        item["sample_b"] = new_b
        converted.append(item)
    return converted


def flatten_hard_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for pair in pairs:
        for row in extract_pair_rows(pair):
            identifier = row_id(row)
            if identifier not in seen:
                seen.add(identifier)
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LocateAnything evaluation on the pinned ViGround benchmark.")
    parser.add_argument("--model-key", choices=["base", "random_ft", "hardpair_ft"], required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--work-dir", default=".hf_cache/viground_eval")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--limit-standard", type=int, default=None, help="Debug limit per standard language.")
    parser.add_argument("--limit-hard-pairs", type=int, default=None, help="Debug limit per hard-pair language.")
    args = parser.parse_args()

    random.seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for final LocateAnything evaluation.")

    output_dir = ensure_dir(Path(args.artifact_root) / "evaluation" / args.model_key)
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
    if args.limit_standard:
        en_rows = en_rows[: args.limit_standard]
        vi_rows = vi_rows[: args.limit_standard]
    if args.limit_hard_pairs:
        hard_pairs = hard_pairs[: args.limit_hard_pairs]

    hard_pairs_vi = make_hard_pair_language_view(hard_pairs, "vi")
    hard_pairs_en = make_hard_pair_language_view(hard_pairs, "en")
    all_rows = []
    seen = set()
    for block in (en_rows, vi_rows, flatten_hard_pairs(hard_pairs_vi), flatten_hard_pairs(hard_pairs_en)):
        for row in block:
            identifier = row_id(row)
            if identifier not in seen:
                seen.add(identifier)
                all_rows.append(row)

    model_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        token=hf_token(),
        local_dir=str(work_dir / "LocateAnything-3B"),
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    worker = LocateAnythingBenchmarkWorker(model_path, args.adapter_path, "cuda", dtype)
    resolver = ImageResolver(images_dir)

    prediction_path = output_dir / "predictions.jsonl"
    predictions = predict_rows(worker, resolver, all_rows, prediction_path, args.model_key, args.retry_errors)

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
        output_dir / "run_manifest.json",
        {
            "model_key": args.model_key,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "adapter_path": args.adapter_path,
            "dataset_id": HF_DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "seed": SEED,
            "generation_mode": GENERATION_MODE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": DO_SAMPLE,
            "standard_counts": {"en": len(en_rows), "vi": len(vi_rows), "alignment_overlap": overlap},
            "hard_pair_counts": {"pairs": len(hard_pairs), "vi_requests": len(flatten_hard_pairs(hard_pairs_vi)), "en_requests": len(flatten_hard_pairs(hard_pairs_en))},
            "environment": capture_environment(),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        },
    )
    print("Evaluation complete:", output_dir)


if __name__ == "__main__":
    main()
