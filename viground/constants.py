from __future__ import annotations

HF_DATASET_ID = "thanhhoangnvbg/viground-contrast-data"
DATASET_REVISION = "eb7ddb1069de0c6e7a6277cf46b3d94d0c9cd9ee"

MODEL_ID = "nvidia/LocateAnything-3B"
MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"

SEED = 42

STANDARD_REPO_PATH = "data/benchmark/standard.jsonl"
HARD_PAIRS_REPO_PATH = "data/benchmark/hard_pairs.jsonl"
IMAGES_ZIP_REPO_PATH = "data/images.zip"
TRAIN_RANDOM_REPO_PATH = "data/train_random/train_samples.jsonl"
TRAIN_HARDPAIR_REPO_PATH = "data/train_hardpair/train_samples.jsonl"
TRAIN_HARDPAIR_PAIRS_REPO_PATH = "data/train_hardpair/hard_pairs.jsonl"
LOCANY_RANDOM_TRAIN_REPO_PATH = "data/train_random/locateanything_train.jsonl"
LOCANY_HARDPAIR_TRAIN_REPO_PATH = "data/train_hardpair/locateanything_train.jsonl"
LOCANY_RANDOM_RECIPE_REPO_PATH = "configs/locany_random_recipe.json"
LOCANY_HARDPAIR_RECIPE_REPO_PATH = "configs/locany_hardpair_recipe.json"

REQUIRED_REPO_FILES = (
    STANDARD_REPO_PATH,
    HARD_PAIRS_REPO_PATH,
    IMAGES_ZIP_REPO_PATH,
    TRAIN_RANDOM_REPO_PATH,
    TRAIN_HARDPAIR_REPO_PATH,
    TRAIN_HARDPAIR_PAIRS_REPO_PATH,
    LOCANY_RANDOM_TRAIN_REPO_PATH,
    LOCANY_HARDPAIR_TRAIN_REPO_PATH,
    LOCANY_RANDOM_RECIPE_REPO_PATH,
    LOCANY_HARDPAIR_RECIPE_REPO_PATH,
)

EXPECTED_COUNTS = {
    "random_ft_train_samples": 4000,
    "hardpair_ft_train_samples": 4000,
    "hardpair_ft_train_pairs": 2000,
    "standard_benchmark_rows": 2000,
    "standard_benchmark_english_rows": 1000,
    "standard_benchmark_vietnamese_rows": 1000,
    "hard_pair_benchmark_pairs": 500,
    "fixed_bundle_images": 5024,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

GENERATION_MODE = "hybrid"
MAX_NEW_TOKENS = 2048
DO_SAMPLE = False

PROMPT_TEMPLATE = (
    "Locate a single instance that matches the following description: {query}."
)

MODEL_OUTPUT_DIRS = {
    "base": "artifacts/evaluation/base",
    "random_ft": "artifacts/evaluation/random_ft",
    "hardpair_ft": "artifacts/evaluation/hardpair_ft",
}

ADAPTER_REPOS = {
    "random_ft": "thanhhoangnvbg/ViGround-Contrast-Random-FT",
    "hardpair_ft": "thanhhoangnvbg/ViGround-Contrast-HardPair-FT",
}

BASE_PILOT_RESULT = {
    "result_type": "pilot",
    "note": "Preliminary 200 EN, 200 VI, 200 VI hard-pair pilot; not full benchmark.",
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "dataset_id": HF_DATASET_ID,
    "dataset_revision": DATASET_REVISION,
    "gpu": "Tesla T4",
    "generation_mode": GENERATION_MODE,
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": DO_SAMPLE,
    "seed": SEED,
    "standard_english_samples": 200,
    "standard_vietnamese_samples": 200,
    "vietnamese_hard_pairs": 200,
    "unique_inference_requests": 800,
    "english": {
        "mIoU": 0.8371,
        "Acc@0.5": 0.9100,
        "Acc@0.75": 0.8250,
        "Parse Fail": 0.0400,
        "Multi-box": 0.0400,
    },
    "vietnamese": {
        "mIoU": 0.6867,
        "Acc@0.5": 0.7300,
        "Acc@0.75": 0.6200,
        "Parse Fail": 0.0750,
        "Multi-box": 0.0650,
    },
    "vietnamese_grounding_gap": 0.1800,
    "hard_pairs": {
        "Pair Accuracy": 0.5200,
        "Pair mIoU": 0.4948,
        "Wrong-Instance": 0.1250,
        "Same-Box Collapse": 0.1150,
        "Parse Fail": 0.0450,
        "Runtime Error": 0.0000,
        "Cross-image Pairs": 0,
    },
}

