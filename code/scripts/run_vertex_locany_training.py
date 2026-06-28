from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from huggingface_hub import snapshot_download

from scripts.patch_eagle_attention_lora import patch_file
from scripts.prepare_locany_training_data import prepare_training_data
from viground.constants import DATASET_REVISION, MODEL_ID, MODEL_REVISION, SEED
from viground.io_utils import capture_environment, ensure_dir, hf_token, write_json, write_text


MODE_STEPS = {
    "smoke": {"max_steps": 2, "save_steps": 2},
    "systems-test": {"max_steps": 50, "save_steps": 25},
    "full": {"max_steps": None, "save_steps": 125},
}


def bool_arg(value: bool) -> str:
    return "True" if value else "False"


def build_command(
    eagle_root: Path,
    model_path: str,
    meta_path: str,
    output_dir: Path,
    mode: str,
    bf16: bool,
) -> list[str]:
    mode_cfg = MODE_STEPS[mode]
    command = [
        "torchrun",
        "--nproc_per_node=1",
        str(eagle_root / "eaglevl" / "train" / "locany_finetune_magi_stream.py"),
        "--model_name_or_path",
        model_path,
        "--meta_path",
        meta_path,
        "--output_dir",
        str(output_dir),
        "--overwrite_output_dir",
        "True",
        "--do_train",
        "True",
        "--num_train_epochs",
        "1",
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "8",
        "--learning_rate",
        "1e-4",
        "--weight_decay",
        "0.01",
        "--warmup_ratio",
        "0.05",
        "--lr_scheduler_type",
        "cosine",
        "--max_grad_norm",
        "1.0",
        "--seed",
        str(SEED),
        "--data_seed",
        str(SEED),
        "--dataloader_num_workers",
        "4",
        "--save_strategy",
        "steps",
        "--save_steps",
        str(mode_cfg["save_steps"]),
        "--save_total_limit",
        "4",
        "--logging_strategy",
        "steps",
        "--logging_first_step",
        "True",
        "--logging_steps",
        "1",
        "--report_to",
        "none",
        "--remove_unused_columns",
        "False",
        "--bf16",
        bool_arg(bf16),
        "--fp16",
        bool_arg(not bf16),
        "--attn_implementation",
        "sdpa",
        "--causal_attn",
        "False",
        "--block_size",
        "6",
        "--grad_checkpoint",
        "True",
        "--max_seq_length",
        "4096",
        "--max_num_tokens_per_sample",
        "4096",
        "--max_num_tokens",
        "4096",
        "--packing_buffer_size",
        "8",
        "--freeze_backbone",
        "True",
        "--freeze_llm",
        "True",
        "--freeze_mlp",
        "True",
        "--use_backbone_lora",
        "0",
        "--use_llm_lora",
        "8",
        "--unfreeze_lm_head",
        "False",
    ]
    if mode_cfg["max_steps"] is not None:
        command.extend(["--max_steps", str(mode_cfg["max_steps"])])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ViGround LocateAnything LoRA training inside the Vertex container.")
    parser.add_argument("--kind", choices=["random_ft", "hardpair_ft"], required=True)
    parser.add_argument("--mode", choices=sorted(MODE_STEPS), default="full")
    parser.add_argument("--work-dir", default=".hf_cache/viground_train")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--eagle-root", default=os.environ.get("EAGLE_ROOT", "/opt/Eagle/Embodied"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.dry_run:
        raise RuntimeError("CUDA is required for LocateAnything training.")

    training_dir = ensure_dir(Path(args.artifact_root) / "training" / args.kind)
    checkpoints_dir = ensure_dir(training_dir / "checkpoints")
    eagle_root = Path(args.eagle_root)
    patched = patch_file(eagle_root / "eaglevl" / "model" / "locany" / "modeling_locateanything.py")
    data_report = prepare_training_data(args.kind, args.work_dir, training_dir)

    model_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        token=hf_token(),
        local_dir=str(Path(args.work_dir) / "LocateAnything-3B"),
    )
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    command = build_command(
        eagle_root=eagle_root,
        model_path=str(model_path),
        meta_path=data_report["eagle_meta_path"],
        output_dir=checkpoints_dir,
        mode=args.mode,
        bf16=bf16,
    )
    if args.resume_from_checkpoint:
        command.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])

    resolved_config = {
        "kind": args.kind,
        "mode": args.mode,
        "dataset_revision": DATASET_REVISION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": str(model_path),
        "eagle_root": str(eagle_root),
        "patches_applied": patched,
        "bf16": bf16,
        "training_output_dir": str(checkpoints_dir),
        "command": command,
        "data": data_report,
    }
    write_json(training_dir / "environment.json", capture_environment())
    write_json(training_dir / "resolved_config.json", resolved_config)
    write_text(training_dir / "resolved_config.yaml", "\n".join(f"{key}: {value}" for key, value in resolved_config.items()) + "\n")
    print("Preflight:")
    print("  dataset_revision:", DATASET_REVISION)
    print("  model_revision:", MODEL_REVISION)
    print("  bf16:", bf16)
    print("  output:", checkpoints_dir)
    print("Command:")
    print(" ".join(command))
    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
