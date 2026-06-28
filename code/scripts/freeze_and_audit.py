from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viground.data import audit_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze revisions and audit the fixed ViGround-Contrast dataset.")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--local-dir", default=".hf_cache/viground")
    parser.add_argument(
        "--skip-image-verify",
        action="store_true",
        help="Skip downloading and checking data/images.zip. Use only for quick local checks.",
    )
    args = parser.parse_args()

    report = audit_dataset(
        artifact_root=args.artifact_root,
        local_dir=args.local_dir,
        verify_images=not args.skip_image_verify,
    )
    print("Dataset audit passed")
    print("Resolved dataset revision:", report["resolved_revision"])
    print("Counts:", report["counts"])
    if report["image_audit"].get("verified"):
        print("Images in fixed bundle:", report["image_audit"]["image_count"])


if __name__ == "__main__":
    main()
