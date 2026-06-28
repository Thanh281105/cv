from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import snapshot_download

from viground.io_utils import ensure_dir, hf_token


DEFAULT_RESULTS_REPO = "thanhhoangnvbg/viground-contrast-results"


def copy_tree_contents(source: Path, target: Path) -> None:
    ensure_dir(target)
    for child in source.iterdir():
        destination = target / child.name
        if child.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync published ViGround result artifacts from Hugging Face.")
    parser.add_argument("--repo-id", default=DEFAULT_RESULTS_REPO)
    parser.add_argument("--revision", default="aaf0b47bb503be3e9889697b5f5e1943e5c95540")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--work-dir", default=".hf_cache/viground_results")
    parser.add_argument(
        "--include",
        nargs="+",
        default=[
            "evaluation/base/*",
            "evaluation/random_ft/*",
            "evaluation/hardpair_ft/*",
            "statistics/*",
            "tables/*",
            "training/random_ft/*",
            "training/hardpair_ft/*",
        ],
    )
    args = parser.parse_args()

    snapshot_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            token=hf_token(),
            local_dir=args.work_dir,
            allow_patterns=args.include,
        )
    )
    artifact_root = ensure_dir(args.artifact_root)
    for folder in ("evaluation", "statistics", "tables", "training"):
        source = snapshot_path / folder
        if source.exists():
            copy_tree_contents(source, artifact_root / folder)
    print("Synced Hugging Face results into:", artifact_root)


if __name__ == "__main__":
    main()
