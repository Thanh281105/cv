from __future__ import annotations

import argparse
import gc
import json
import random
import re
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
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from viground.constants import (
    DATASET_REVISION,
    HARD_PAIRS_REPO_PATH,
    HF_DATASET_ID,
    IMAGE_EXTENSIONS,
    IMAGES_ZIP_REPO_PATH,
    SEED,
    STANDARD_REPO_PATH,
)
from viground.data import choose_alignment_key, extract_pair_rows, load_dataset_file, row_id
from viground.io_utils import append_jsonl, capture_environment, ensure_dir, hf_token, read_jsonl, write_json
from viground.metrics import evaluate_hard_pairs, evaluate_standard, latency_stats, parse_boxes


MODEL_CONFIGS = {
    "grounding_dino": {
        "model_id": "IDEA-Research/grounding-dino-base",
        "model_family": "grounding_dino",
    },
    "qwen2_5_vl_3b": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_family": "qwen2_5_vl",
    },
}


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
            matches = self.basename_index.get(Path(normalized).name, [])
            if len(matches) == 1:
                return matches[0]
        raise FileNotFoundError(f"Image not found for row {row.get('sample_id') or row.get('benchmark_id')}: {candidates}")


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


def row_query(row: dict[str, Any]) -> str:
    return str(row.get("query") or row.get("expression_vi") or row.get("expression_en"))


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


def normalized_from_pixel(box: list[float], width: int, height: int) -> list[int]:
    return [
        int(round(max(0.0, min(box[0] / width * 1000, 1000.0)))),
        int(round(max(0.0, min(box[1] / height * 1000, 1000.0)))),
        int(round(max(0.0, min(box[2] / width * 1000, 1000.0)))),
        int(round(max(0.0, min(box[3] / height * 1000, 1000.0)))),
    ]


def pixel_from_normalized(box: list[float], width: int, height: int) -> list[float]:
    return [
        max(0.0, min(box[0] / 1000 * width, float(width))),
        max(0.0, min(box[1] / 1000 * height, float(height))),
        max(0.0, min(box[2] / 1000 * width, float(width))),
        max(0.0, min(box[3] / 1000 * height, float(height))),
    ]


def box_token(box: list[int]) -> str:
    return f"<box><{box[0]}><{box[1]}><{box[2]}><{box[3]}></box>"


class GroundingDinoWorker:
    def __init__(self, model_path: str | Path, device: str, box_threshold: float, text_threshold: float):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(device).eval()

    @staticmethod
    def format_query(query: str) -> str:
        text = query.strip().lower()
        return text if text.endswith(".") else f"{text}."

    @torch.inference_mode()
    def predict(self, image: Image.Image, query: str) -> dict[str, Any]:
        image = image.convert("RGB")
        text = self.format_query(query)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        started = time.perf_counter()
        outputs = self.model(**inputs)
        latency_ms = (time.perf_counter() - started) * 1000
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        labels = results.get("labels", results.get("text_labels", []))
        candidates = []
        for index, box in enumerate(boxes):
            pixel = [float(value) for value in box.tolist()]
            score = float(scores[index].item() if hasattr(scores[index], "item") else scores[index])
            label = str(labels[index]) if index < len(labels) else text
            candidates.append({"box": pixel, "score": score, "label": label})
        candidates.sort(key=lambda item: item["score"], reverse=True)
        if not candidates:
            return {
                "raw": "",
                "norm": [],
                "pixel": [],
                "parse_ok": False,
                "latency_ms": latency_ms,
                "error": None,
                "candidates": [],
            }
        best = candidates[0]["box"]
        normalized = normalized_from_pixel(best, image.width, image.height)
        return {
            "raw": box_token(normalized),
            "norm": [normalized],
            "pixel": [best],
            "parse_ok": True,
            "latency_ms": latency_ms,
            "error": None,
            "candidates": candidates[:20],
        }


BOX_PATTERNS = (
    re.compile(r'"(?:bbox|bbox_2d|box|bounding_box)"\s*:\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]', re.IGNORECASE),
    re.compile(r'"(?:bbox|bbox_2d|box|bounding_box)"\s*:\s*\[\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]\s*,\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]\s*\]', re.IGNORECASE),
    re.compile(r'\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)\s*,\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)'),
    re.compile(r'\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]'),
)


def parse_qwen_box(text: str, width: int, height: int) -> tuple[list[list[int]], list[list[float]], bool]:
    normalized, pixel, parse_ok = parse_boxes(text, width, height)
    if parse_ok:
        return normalized, pixel, parse_ok

    values = None
    for pattern in BOX_PATTERNS:
        match = pattern.search(text or "")
        if match:
            values = [float(match.group(index)) for index in range(1, 5)]
            break
    if values is None:
        return [], [], False
    x1, y1, x2, y2 = values
    if max(values) <= 1.0:
        values = [value * 1000 for value in values]
        x1, y1, x2, y2 = values
    elif x2 <= width and y2 <= height:
        pixel = [x1, y1, x2, y2]
        if pixel[2] <= pixel[0] or pixel[3] <= pixel[1]:
            return [], [], False
        normalized = normalized_from_pixel(pixel, width, height)
        return [normalized], [pixel], True
    if x2 <= x1 or y2 <= y1 or any(value < 0 or value > 1000 for value in values):
        return [], [], False
    pixel = pixel_from_normalized(values, width, height)
    if pixel[2] <= pixel[0] or pixel[3] <= pixel[1]:
        return [], [], False
    return [[int(round(value)) for value in values]], [pixel], True


class QwenVlWorker:
    def __init__(self, model_path: str | Path, device: str, dtype: torch.dtype, max_new_tokens: int, min_pixels: int, max_pixels: int):
        try:
            from qwen_vl_utils import process_vision_info
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError("Qwen2.5-VL evaluation requires `pip install qwen-vl-utils==0.0.8`.") from exc

        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device).eval()

    @staticmethod
    def build_prompt(query: str) -> str:
        return (
            "Locate exactly one object matching this referring expression:\n"
            f"{query}\n\n"
            "Return only one bounding box and no other text. "
            "Prefer this exact format: <box><x1><y1><x2><y2></box>. "
            "If you cannot use that format, return valid JSON: {\"bbox_2d\":[x1,y1,x2,y2]}. "
            "Coordinates must be integers normalized from 0 to 1000, where "
            "(0,0) is the top-left image corner and (1000,1000) is the bottom-right."
        )

    @torch.inference_mode()
    def predict(self, image_path: Path, image: Image.Image, query: str) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path.resolve().as_uri()},
                    {"type": "text", "text": self.build_prompt(query)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        started = time.perf_counter()
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        latency_ms = (time.perf_counter() - started) * 1000
        generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        normalized, pixel, parse_ok = parse_qwen_box(answer, image.width, image.height)
        raw = box_token(normalized[0]) if parse_ok and normalized else answer
        return {
            "raw": raw,
            "model_raw": answer,
            "norm": normalized,
            "pixel": pixel,
            "parse_ok": parse_ok,
            "latency_ms": latency_ms,
            "error": None,
        }


def predict_one(worker: Any, resolver: ImageResolver, row: dict[str, Any], model_family: str) -> dict[str, Any]:
    try:
        image_path = resolver.resolve(row)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if model_family == "qwen2_5_vl":
            return worker.predict(image_path=image_path, image=image, query=row_query(row))
        return worker.predict(image=image, query=row_query(row))
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
    worker: Any,
    resolver: ImageResolver,
    rows: list[dict[str, Any]],
    checkpoint_path: str | Path,
    model_key: str,
    model_family: str,
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
    print(f"{model_key}: total={len(rows)}, available={len(predictions)}, pending={len(pending)}")
    for row in tqdm(pending, desc=model_key):
        identifier = row_id(row)
        prediction = predict_one(worker, resolver, row, model_family)
        predictions[identifier] = prediction
        append_jsonl(checkpoint_path, {"id": identifier, "prediction": prediction})
    return predictions


def validate_box_token_smoke(predictions: dict[str, dict[str, Any]], output_dir: Path) -> None:
    failures = []
    for identifier, prediction in predictions.items():
        raw = str(prediction.get("raw") or "")
        if prediction.get("error") or not prediction.get("parse_ok") or "<box>" not in raw or "</box>" not in raw:
            failures.append(
                {
                    "id": identifier,
                    "raw": raw,
                    "error": prediction.get("error"),
                    "parse_ok": prediction.get("parse_ok"),
                }
            )
    report = {
        "checked": len(predictions),
        "passed": len(predictions) - len(failures),
        "failed": len(failures),
        "failures": failures[:50],
    }
    write_json(output_dir / "smoke_box_format_check.json", report)
    if failures:
        raise RuntimeError(f"Smoke box-format check failed for {len(failures)} predictions. See {output_dir / 'smoke_box_format_check.json'}")


def build_worker(args: argparse.Namespace, model_path: str) -> Any:
    family = MODEL_CONFIGS[args.model_key]["model_family"]
    if family == "grounding_dino":
        return GroundingDinoWorker(
            model_path=model_path,
            device=args.device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if args.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return QwenVlWorker(
        model_path=model_path,
        device=args.device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.qwen_min_pixels,
        max_pixels=args.qwen_max_pixels,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run additional ViGround baselines on the pinned benchmark.")
    parser.add_argument("--model-key", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--model-id", default=None, help="Override the default Hugging Face model id.")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--work-dir", default=".hf_cache/viground_baselines")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--limit-standard", type=int, default=None, help="Debug limit per standard language.")
    parser.add_argument("--limit-hard-pairs", type=int, default=None, help="Debug limit per hard-pair language.")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--qwen-min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--qwen-max-pixels", type=int, default=2048 * 28 * 28)
    parser.add_argument("--smoke-check", action="store_true", help="Fail if any prediction does not produce a LocateAnything-style <box> token.")
    args = parser.parse_args()

    random.seed(SEED)
    if args.device == "cuda" and not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Use --device cpu --allow-cpu only for small smoke tests.")

    config = MODEL_CONFIGS[args.model_key]
    model_id = args.model_id or config["model_id"]
    model_family = config["model_family"]
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
        repo_id=model_id,
        revision=args.model_revision,
        token=hf_token(),
        local_dir=str(work_dir / model_id.replace("/", "__")),
    )
    worker = build_worker(args, model_path)
    resolver = ImageResolver(images_dir)

    prediction_path = output_dir / "predictions.jsonl"
    predictions = predict_rows(worker, resolver, all_rows, prediction_path, args.model_key, model_family, args.retry_errors)
    if args.smoke_check:
        validate_box_token_smoke(predictions, output_dir)

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
            "model_id": model_id,
            "model_revision": args.model_revision,
            "resolved_model_path": str(model_path),
            "model_family": model_family,
            "dataset_id": HF_DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "seed": SEED,
            "standard_counts": {"en": len(en_rows), "vi": len(vi_rows), "alignment_overlap": overlap},
            "hard_pair_counts": {
                "pairs": len(hard_pairs),
                "vi_requests": len(flatten_hard_pairs(hard_pairs_vi)),
                "en_requests": len(flatten_hard_pairs(hard_pairs_en)),
            },
            "grounding_dino": {"box_threshold": args.box_threshold, "text_threshold": args.text_threshold},
            "qwen2_5_vl": {
                "max_new_tokens": args.max_new_tokens,
                "dtype": args.dtype,
                "min_pixels": args.qwen_min_pixels,
                "max_pixels": args.qwen_max_pixels,
                "max_visual_tokens": args.qwen_max_pixels // (28 * 28),
            },
            "raw_box_format": "<box><x1><y1><x2><y2></box>",
            "smoke_check": args.smoke_check,
            "environment": capture_environment(),
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        },
    )
    print("Evaluation complete:", output_dir)


if __name__ == "__main__":
    main()
