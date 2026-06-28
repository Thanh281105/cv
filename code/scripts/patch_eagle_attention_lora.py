from __future__ import annotations

import argparse
import os
from pathlib import Path


OLD_LORA_TARGETS = """target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                            'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']"""
NEW_LORA_TARGETS = """target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']"""

OLD_VISION_ATTN = "config.vision_config._attn_implementation = 'flash_attention_2'"
NEW_VISION_ATTN = "config.vision_config._attn_implementation = model_args.attn_implementation"


def _backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".viground.bak")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def patch_model_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    changes = []
    if OLD_LORA_TARGETS in text:
        text = text.replace(OLD_LORA_TARGETS, NEW_LORA_TARGETS, 1)
        changes.append("llm_lora_attention_only")
    elif NEW_LORA_TARGETS not in text:
        raise RuntimeError("Could not find the expected wrap_llm_lora target_modules block")

    if changes:
        _backup_once(path)
        path.write_text(text, encoding="utf-8")
    return changes


def patch_train_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    changes = []
    if OLD_VISION_ATTN in text:
        text = text.replace(OLD_VISION_ATTN, NEW_VISION_ATTN, 1)
        changes.append("vision_attn_uses_model_args")
    elif NEW_VISION_ATTN not in text:
        raise RuntimeError("Could not find the expected vision attention assignment")
    if changes:
        _backup_once(path)
        path.write_text(text, encoding="utf-8")
    return changes


def patch_file(path: Path) -> list[str]:
    return patch_model_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch Eagle LocateAnything for attention-only LLM LoRA on L4.")
    parser.add_argument("--eagle-root", default=os.environ.get("EAGLE_ROOT", "/opt/Eagle/Embodied"))
    args = parser.parse_args()

    model_target = Path(args.eagle_root) / "eaglevl" / "model" / "locany" / "modeling_locateanything.py"
    train_target = Path(args.eagle_root) / "eaglevl" / "train" / "locany_finetune_magi_stream.py"
    for target in (model_target, train_target):
        if not target.exists():
            raise FileNotFoundError(target)
    changes = patch_model_file(model_target) + patch_train_file(train_target)
    print("Patched:", model_target)
    print("Patched:", train_target)
    print("Changes:", changes or ["already_patched"])


if __name__ == "__main__":
    main()
