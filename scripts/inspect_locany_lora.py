from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from viground.constants import MODEL_ID, MODEL_REVISION
from viground.io_utils import ensure_dir, hf_token, write_json


ATTENTION_TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
FORBIDDEN_TRAINABLE_MARKERS = (
    "vision_model",
    "mlp1",
    "multi_modal_projector",
    "gate_proj",
    "down_proj",
    "up_proj",
    "embed_tokens",
    "lm_head",
)


def module_category(name: str) -> str:
    if "lora_" in name:
        return "lora"
    if "vision_model" in name:
        return "vision"
    if "mlp1" in name or "projector" in name:
        return "projector"
    if "language_model" in name:
        return "language_model_base"
    return "other"


def inspect_model(model_path: str, output_path: str | Path, dtype: torch.dtype) -> dict[str, Any]:
    from eaglevl.model.locany.configuration_locateanything import LocateAnythingConfig
    from eaglevl.model.locany.modeling_locateanything import LocateAnythingForConditionalGeneration
    from eaglevl.train.constants import (
        BOX_END_TOKEN,
        BOX_START_TOKEN,
        IMG_CONTEXT_TOKEN,
        NULL_TOKEN,
        REF_END_TOKEN,
        REF_START_TOKEN,
        TEXT_MASK_TOKEN,
        number_tokens_list,
        special_tokens_list,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
    tokenizer.add_tokens(special_tokens_list + number_tokens_list, special_tokens=True)
    config = LocateAnythingConfig.from_pretrained(model_path)
    config._attn_implementation = "sdpa"
    config._attn_implementation_autoset = False
    config.text_config._attn_implementation = "sdpa"
    config.text_config._attn_implementation_autoset = False
    config.vision_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation_autoset = False
    config.image_token_index = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    config.text_mask_token_id = tokenizer.convert_tokens_to_ids(TEXT_MASK_TOKEN)
    config.none_token_id = tokenizer.convert_tokens_to_ids(NULL_TOKEN)
    config.box_start_token_id = tokenizer.convert_tokens_to_ids(BOX_START_TOKEN)
    config.box_end_token_id = tokenizer.convert_tokens_to_ids(BOX_END_TOKEN)
    config.ref_start_token_id = tokenizer.convert_tokens_to_ids(REF_START_TOKEN)
    config.ref_end_token_id = tokenizer.convert_tokens_to_ids(REF_END_TOKEN)

    model = LocateAnythingForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        config=config,
        attn_implementation="sdpa",
    )
    for parameter in model.vision_model.parameters():
        parameter.requires_grad = False
    for parameter in model.language_model.parameters():
        parameter.requires_grad = False
    if hasattr(model, "mlp1"):
        for parameter in model.mlp1.parameters():
            parameter.requires_grad = False
    model.wrap_llm_lora(r=8, lora_alpha=16, lora_dropout=0.05)

    parameters = []
    trainable_names = []
    for name, parameter in model.named_parameters():
        item = {
            "name": name,
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "requires_grad": bool(parameter.requires_grad),
            "module_category": module_category(name),
        }
        parameters.append(item)
        if parameter.requires_grad:
            trainable_names.append(name)

    bad_trainable = [
        name
        for name in trainable_names
        if "lora_" not in name or any(marker in name for marker in FORBIDDEN_TRAINABLE_MARKERS if marker not in ("language_model",))
    ]
    missing_attention = [
        suffix
        for suffix in ATTENTION_TARGET_SUFFIXES
        if not any(suffix in name and "lora_" in name for name in trainable_names)
    ]
    manifest = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": list(ATTENTION_TARGET_SUFFIXES)},
        "total_parameters": sum(item["numel"] for item in parameters),
        "trainable_parameters": sum(item["numel"] for item in parameters if item["requires_grad"]),
        "trainable_names": trainable_names,
        "bad_trainable": bad_trainable,
        "missing_attention_targets": missing_attention,
        "parameters": parameters,
    }
    write_json(output_path, manifest)
    if not trainable_names:
        raise RuntimeError("No trainable LoRA parameters found")
    if bad_trainable:
        raise RuntimeError(f"Forbidden trainable parameters found: {bad_trainable[:10]}")
    if missing_attention:
        raise RuntimeError(f"Missing attention LoRA targets: {missing_attention}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect LocateAnything attention-only LoRA trainable parameters.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--work-dir", default=".hf_cache/viground_train")
    parser.add_argument("--output", default="artifacts/training/trainable_parameters.json")
    args = parser.parse_args()

    model_path = args.model_path
    if model_path is None:
        model_path = snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            token=hf_token(),
            local_dir=str(Path(args.work_dir) / "LocateAnything-3B"),
        )
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    ensure_dir(Path(args.output).parent)
    manifest = inspect_model(str(model_path), args.output, dtype)
    print("Trainable parameters:", manifest["trainable_parameters"])


if __name__ == "__main__":
    main()
