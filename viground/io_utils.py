from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def capture_environment() -> dict[str, Any]:
    packages = [
        "accelerate",
        "datasets",
        "decord",
        "huggingface_hub",
        "lmdb",
        "numpy",
        "opencv-python-headless",
        "peft",
        "Pillow",
        "safetensors",
        "torch",
        "transformers",
    ]
    env: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {package: package_version(package) for package in packages},
    }
    try:
        import torch

        env["torch_cuda_available"] = torch.cuda.is_available()
        env["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            env["gpu"] = {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_vram_bytes": props.total_memory,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
    except Exception as exc:
        env["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    return env

