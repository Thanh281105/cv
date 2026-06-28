from __future__ import annotations

import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Any

from .data import extract_pair_rows, row_id

BOX_TOKEN_RE = re.compile(
    r"<box>\s*"
    r"<\s*(\d+(?:\.\d+)?)\s*>\s*"
    r"<\s*(\d+(?:\.\d+)?)\s*>\s*"
    r"<\s*(\d+(?:\.\d+)?)\s*>\s*"
    r"<\s*(\d+(?:\.\d+)?)\s*>\s*"
    r"</box>",
    re.IGNORECASE,
)

BOX_COMMA_RE = re.compile(
    r"<box>\s*\(?\s*"
    r"([+-]?\d+(?:\.\d+)?)\s*,\s*"
    r"([+-]?\d+(?:\.\d+)?)\s*,\s*"
    r"([+-]?\d+(?:\.\d+)?)\s*,\s*"
    r"([+-]?\d+(?:\.\d+)?)\s*"
    r"\)?\s*</box>",
    re.IGNORECASE,
)


def parse_boxes(text: str | None, image_width: int, image_height: int) -> tuple[list[list[int]], list[list[float]], bool]:
    text = text or ""
    values: list[list[float]] = []
    for regex in (BOX_TOKEN_RE, BOX_COMMA_RE):
        for match in regex.finditer(text):
            values.append([float(match.group(index)) for index in range(1, 5)])

    unique_values = []
    seen = set()
    for box in values:
        key = tuple(round(value, 6) for value in box)
        if key not in seen:
            seen.add(key)
            unique_values.append(box)

    normalized_boxes = []
    pixel_boxes = []
    for x1, y1, x2, y2 in unique_values:
        coordinates = (x1, y1, x2, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        if any(value < 0 or value > 1000 for value in coordinates):
            continue
        normalized_boxes.append([int(round(value)) for value in coordinates])
        pixel = [
            max(0.0, min(x1 / 1000 * image_width, float(image_width))),
            max(0.0, min(y1 / 1000 * image_height, float(image_height))),
            max(0.0, min(x2 / 1000 * image_width, float(image_width))),
            max(0.0, min(y2 / 1000 * image_height, float(image_height))),
        ]
        if pixel[2] > pixel[0] and pixel[3] > pixel[1]:
            pixel_boxes.append(pixel)
    return normalized_boxes, pixel_boxes, bool(pixel_boxes)


def iou(box_a: list[float], box_b: list[float]) -> float:
    intersection_x1 = max(box_a[0], box_b[0])
    intersection_y1 = max(box_a[1], box_b[1])
    intersection_x2 = min(box_a[2], box_b[2])
    intersection_y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, intersection_x2 - intersection_x1) * max(0.0, intersection_y2 - intersection_y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def prediction_is_valid(prediction: dict[str, Any] | None) -> bool:
    return bool(
        prediction
        and not prediction.get("error")
        and prediction.get("parse_ok")
        and prediction.get("pixel")
        and len(prediction["pixel"]) == 1
    )


def evaluate_standard(rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    iou_values = []
    parse_failures = 0
    multi_box_outputs = 0
    runtime_errors = 0
    latencies = []

    for row in rows:
        prediction = predictions.get(row_id(row))
        if prediction and prediction.get("latency_ms"):
            latencies.append(float(prediction["latency_ms"]))
        if prediction and prediction.get("error"):
            runtime_errors += 1
        if not prediction_is_valid(prediction):
            parse_failures += 1
            if prediction and len(prediction.get("pixel", [])) > 1:
                multi_box_outputs += 1
            iou_values.append(0.0)
            continue
        iou_values.append(iou(prediction["pixel"][0], row["bbox_xyxy"]))

    count = len(rows)
    return {
        "n": count,
        "mIoU": _rate_mean(iou_values),
        "Acc@0.5": _rate(sum(value >= 0.5 for value in iou_values), count),
        "Acc@0.75": _rate(sum(value >= 0.75 for value in iou_values), count),
        "Parse Fail": _rate(parse_failures, count),
        "Multi-box": _rate(multi_box_outputs, count),
        "Runtime Error": _rate(runtime_errors, count),
        "mean_latency_ms": _rate_mean(latencies),
        "median_latency_ms": round(median(latencies), 4) if latencies else 0.0,
    }


def evaluate_hard_pairs(pairs: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    query_ious = []
    pair_correctness = []
    pair_min_ious = []
    wrong_instance_flags = []
    collapse_flags = []
    parse_failures = 0
    multi_box_outputs = 0
    runtime_errors = 0
    cross_image_pairs = 0
    latencies = []

    for pair in pairs:
        sample_a, sample_b = extract_pair_rows(pair)
        prediction_a = predictions.get(row_id(sample_a))
        prediction_b = predictions.get(row_id(sample_b))
        for prediction in (prediction_a, prediction_b):
            if prediction and prediction.get("latency_ms"):
                latencies.append(float(prediction["latency_ms"]))
            if prediction and prediction.get("error"):
                runtime_errors += 1
            if not prediction_is_valid(prediction):
                parse_failures += 1
                if prediction and len(prediction.get("pixel", [])) > 1:
                    multi_box_outputs += 1

        valid_a = prediction_is_valid(prediction_a)
        valid_b = prediction_is_valid(prediction_b)
        if not valid_a:
            query_ious.append(0.0)
        if not valid_b:
            query_ious.append(0.0)
        if not (valid_a and valid_b):
            pair_correctness.append(False)
            pair_min_ious.append(0.0)
            wrong_instance_flags.extend([False, False])
            collapse_flags.append(False)
            continue

        predicted_a = prediction_a["pixel"][0]
        predicted_b = prediction_b["pixel"][0]
        ground_truth_a = sample_a["bbox_xyxy"]
        ground_truth_b = sample_b["bbox_xyxy"]
        own_iou_a = iou(predicted_a, ground_truth_a)
        own_iou_b = iou(predicted_b, ground_truth_b)
        query_ious.extend([own_iou_a, own_iou_b])
        pair_correctness.append(own_iou_a >= 0.5 and own_iou_b >= 0.5)
        pair_min_ious.append(min(own_iou_a, own_iou_b))

        same_image = Path(sample_a["file_name"]).name == Path(sample_b["file_name"]).name
        if same_image:
            wrong_instance_flags.append(iou(predicted_a, ground_truth_b) > own_iou_a)
            wrong_instance_flags.append(iou(predicted_b, ground_truth_a) > own_iou_b)
            collapse_flags.append(iou(predicted_a, predicted_b) >= 0.7 and iou(ground_truth_a, ground_truth_b) < 0.3)
        else:
            cross_image_pairs += 1
            wrong_instance_flags.extend([False, False])
            collapse_flags.append(False)

    pair_count = len(pairs)
    query_count = pair_count * 2
    return {
        "pairs": pair_count,
        "queries": query_count,
        "Query Acc@0.5": _rate(sum(value >= 0.5 for value in query_ious), query_count),
        "Query mIoU": _rate_mean(query_ious),
        "Pair Accuracy": _rate(sum(pair_correctness), pair_count),
        "Pair mIoU": _rate_mean(pair_min_ious),
        "Wrong-Instance": _rate(sum(wrong_instance_flags), len(wrong_instance_flags)),
        "Same-Box Collapse": _rate(sum(collapse_flags), pair_count),
        "Parse Fail": _rate(parse_failures, query_count),
        "Multi-box": _rate(multi_box_outputs, query_count),
        "Runtime Error": _rate(runtime_errors, query_count),
        "Cross-image Pairs": cross_image_pairs,
        "mean_latency_ms": _rate_mean(latencies),
        "median_latency_ms": round(median(latencies), 4) if latencies else 0.0,
    }


def latency_stats(predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(item["latency_ms"]) for item in predictions.values() if item.get("latency_ms")]
    if not latencies:
        return {"n": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "n": len(latencies),
        "mean_ms": round(mean(latencies), 4),
        "median_ms": round(median(latencies), 4),
        "p95_ms": round(ordered[p95_index], 4),
    }


def _rate(numerator: int | float, denominator: int) -> float:
    return round(float(numerator) / denominator, 4) if denominator else 0.0


def _rate_mean(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0

