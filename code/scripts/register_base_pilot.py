from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viground.constants import BASE_PILOT_RESULT
from viground.io_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the fixed LocateAnything base-model pilot result.")
    parser.add_argument("--output", default="results/base_pilot/pilot_result.json")
    args = parser.parse_args()

    output = Path(args.output)
    write_json(output, BASE_PILOT_RESULT)
    print(f"Wrote pilot result: {output}")


if __name__ == "__main__":
    main()
