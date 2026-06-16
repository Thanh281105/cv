# ViGround-Contrast

Repo này chứa pipeline thí nghiệm ViGround-Contrast cho mô hình `nvidia/LocateAnything-3B`. Mục tiêu là kiểm tra liệu fine-tuning bằng các cặp khó tiếng Việt có giảm khoảng cách grounding đa ngôn ngữ và cải thiện phân biệt các instance cùng loại tốt hơn fine-tuning tiếng Việt ngẫu nhiên hay không.

## Nội dung chính

- Audit dataset cố định trên Hugging Face và lưu manifest/revision.
- Chuẩn bị dữ liệu huấn luyện LocateAnything/Eagle cho hai nhánh `random_ft` và `hardpair_ft`.
- Chạy LoRA fine-tuning cho LocateAnything-3B trong môi trường Vertex/Eagle.
- Đánh giá `base`, `random_ft`, `hardpair_ft` trên standard benchmark và hard-pair benchmark.
- Tính thống kê paired bootstrap/McNemar và sinh bảng kết quả CSV.

## Revisions cố định

Các script đang pin dataset/model revision trong `viground/constants.py`:

| Thành phần | Giá trị |
| --- | --- |
| Dataset | `thanhhoangnvbg/viground-contrast-data` |
| Dataset revision | `eb7ddb1069de0c6e7a6277cf46b3d94d0c9cd9ee` |
| Base model | `nvidia/LocateAnything-3B` |
| Model revision | `c32291ca5e996f5a7a485845b4f57a233936bba0` |
| Seed | `42` |

Không dùng `revision="main"` cho final runs.

## Hugging Face artifacts

Các artifact đã publish trên Hugging Face:

| Loại | Repo | SHA đã kiểm tra | Nội dung |
| --- | --- | --- | --- |
| Dataset | [`thanhhoangnvbg/viground-contrast-data`](https://huggingface.co/datasets/thanhhoangnvbg/viground-contrast-data) | `eb7ddb1069de0c6e7a6277cf46b3d94d0c9cd9ee` | Dataset train/eval, image bundle, config, runbook |
| Results | [`thanhhoangnvbg/viground-contrast-results`](https://huggingface.co/datasets/thanhhoangnvbg/viground-contrast-results) | `aaf0b47bb503be3e9889697b5f5e1943e5c95540` | Evaluation outputs, tables, statistics, logs |
| Model | [`thanhhoangnvbg/viground-random-ft-locany`](https://huggingface.co/thanhhoangnvbg/viground-random-ft-locany) | `41d5abf5d0244fe3fb6a492f15f16684d230fdad` | Random-FT LocateAnything-3B checkpoint |
| Model | [`thanhhoangnvbg/viground-hardpair-ft-locany`](https://huggingface.co/thanhhoangnvbg/viground-hardpair-ft-locany) | `261e88d6e5cce52a63b4ba65d9562bc3c80392fd` | HardPair-FT LocateAnything-3B checkpoint |

Tải artifact nhỏ từ results repo:

```bash
hf download thanhhoangnvbg/viground-contrast-results \
  --repo-type dataset \
  --include "tables/*.csv" "statistics/*" \
  --local-dir artifacts/hf_results
```

Tải checkpoint đã fine-tune từ Hub:

```bash
hf download thanhhoangnvbg/viground-random-ft-locany \
  --local-dir artifacts/models/viground-random-ft-locany

hf download thanhhoangnvbg/viground-hardpair-ft-locany \
  --local-dir artifacts/models/viground-hardpair-ft-locany
```

## Cấu trúc repo

```text
.
├── configs/
│   └── viground_experiment.yaml
├── docker/
│   └── vertex/Dockerfile
├── scripts/
│   ├── freeze_and_audit.py
│   ├── prepare_locany_training_data.py
│   ├── run_vertex_locany_training.py
│   ├── evaluate_locateanything.py
│   ├── compute_statistics.py
│   ├── generate_tables.py
│   └── register_base_pilot.py
├── viground/
│   ├── constants.py
│   ├── data.py
│   ├── io_utils.py
│   └── metrics.py
├── fine-tune-llm.ipynb
├── test_base_model.ipynb
├── plan.md
└── requirements-locany.txt
```

Các thư mục `.hf_cache/`, `artifacts/` và `results/` là output/cache cục bộ và được ignore bởi git.

## Cài đặt

Yêu cầu khuyến nghị:

- Python 3.10+.
- GPU CUDA cho training/evaluation LocateAnything.
- Hugging Face token nếu dataset/model yêu cầu quyền truy cập.
- Môi trường Eagle/LocateAnything khi chạy training qua `scripts/run_vertex_locany_training.py`.

Tạo môi trường và cài dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-locany.txt
```

Trên Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-locany.txt
```

Nếu cần token Hugging Face:

```bash
export HF_TOKEN=your_token_here
```

PowerShell:

```powershell
$env:HF_TOKEN="your_token_here"
```

## Quy trình chạy

### 1. Audit dataset và freeze revisions

```bash
python scripts/freeze_and_audit.py
```

Lệnh này tải/kiểm tra dataset đã pin, xác nhận số lượng mẫu và tạo các artifact:

```text
artifacts/frozen_revisions/dataset_revision.txt
artifacts/frozen_revisions/model_revision.txt
artifacts/frozen_revisions/environment.json
artifacts/frozen_revisions/repository_file_manifest.json
artifacts/dataset_audit/dataset_integrity_report.json
```

Để kiểm tra nhanh mà không verify ảnh:

```bash
python scripts/freeze_and_audit.py --skip-image-verify
```

### 2. Ghi nhận pilot result của base model

```bash
python scripts/register_base_pilot.py
```

Output mặc định:

```text
results/base_pilot/pilot_result.json
```

Đây là kết quả pilot đã định nghĩa trong repo, không phải full benchmark.

### 3. Chuẩn bị dữ liệu training

```bash
python scripts/prepare_locany_training_data.py --kind random_ft
python scripts/prepare_locany_training_data.py --kind hardpair_ft
```

Output mặc định nằm trong:

```text
artifacts/training/random_ft/
artifacts/training/hardpair_ft/
```

### 4. Chạy LoRA training

Script training giả định đang chạy trong container/máy có Eagle source tree. Mặc định `EAGLE_ROOT=/opt/Eagle/Embodied`; có thể override bằng biến môi trường hoặc argument.

Dry run để xem lệnh training đã resolve:

```bash
python scripts/run_vertex_locany_training.py --kind random_ft --mode smoke --dry-run
python scripts/run_vertex_locany_training.py --kind hardpair_ft --mode smoke --dry-run
```

Chạy smoke test:

```bash
python scripts/run_vertex_locany_training.py --kind random_ft --mode smoke
python scripts/run_vertex_locany_training.py --kind hardpair_ft --mode smoke
```

Chạy full training:

```bash
python scripts/run_vertex_locany_training.py --kind random_ft --mode full
python scripts/run_vertex_locany_training.py --kind hardpair_ft --mode full
```

Các mode có sẵn:

| Mode | Mục đích |
| --- | --- |
| `smoke` | Kiểm tra nhanh 2 steps |
| `systems-test` | Kiểm tra hệ thống 50 steps |
| `full` | Chạy full 1 epoch |

### 5. Evaluate

Evaluate base model:

```bash
python scripts/evaluate_locateanything.py --model-key base
```

Evaluate model đã fine-tune:

```bash
python scripts/evaluate_locateanything.py --model-key random_ft --adapter-path artifacts/training/random_ft/checkpoints/<checkpoint-or-adapter-dir>
python scripts/evaluate_locateanything.py --model-key hardpair_ft --adapter-path artifacts/training/hardpair_ft/checkpoints/<checkpoint-or-adapter-dir>
```

Chạy debug với giới hạn mẫu:

```bash
python scripts/evaluate_locateanything.py --model-key base --limit-standard 10 --limit-hard-pairs 10
```

Output evaluation mặc định:

```text
artifacts/evaluation/base/
artifacts/evaluation/random_ft/
artifacts/evaluation/hardpair_ft/
```

### 6. Tính thống kê và sinh bảng

Sau khi có đủ `predictions.jsonl` cho cả ba model:

```bash
python scripts/compute_statistics.py
python scripts/generate_tables.py
```

Bảng CSV được ghi vào:

```text
artifacts/tables/
```

## Kết quả final đã publish

Các số liệu dưới đây lấy từ `thanhhoangnvbg/viground-contrast-results`.

Standard benchmark:

| Model | Language | n | mIoU | Acc@0.5 | Acc@0.75 | Parse fail | Multi-box | VI gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | en | 1000 | 0.8027 | 0.870 | 0.792 | 0.080 | 0.080 |  |
| Base | vi | 1000 | 0.6627 | 0.704 | 0.627 | 0.059 | 0.056 | 0.166 |
| Random-FT | en | 1000 | 0.7779 | 0.842 | 0.770 | 0.108 | 0.108 |  |
| Random-FT | vi | 1000 | 0.6758 | 0.719 | 0.643 | 0.067 | 0.060 | 0.123 |
| HardPair-FT | en | 1000 | 0.7715 | 0.834 | 0.762 | 0.114 | 0.114 |  |
| HardPair-FT | vi | 1000 | 0.6774 | 0.720 | 0.648 | 0.069 | 0.062 | 0.114 |

Vietnamese hard-pair benchmark:

| Model | Pairs | Query Acc@0.5 | Pair Accuracy | Pair mIoU | Wrong-instance | Same-box collapse | Parse fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 500 | 0.672 | 0.504 | 0.4869 | 0.119 | 0.124 | 0.031 |
| Random-FT | 500 | 0.706 | 0.564 | 0.5317 | 0.101 | 0.086 | 0.036 |
| HardPair-FT | 500 | 0.708 | 0.564 | 0.5317 | 0.095 | 0.082 | 0.035 |

Paired effects:

| Comparison | Metric | n | Difference | 95% CI | p-value |
| --- | --- | ---: | ---: | --- | ---: |
| Random-FT versus Base | `acc05` | 1000 | 0.015 | `[-0.002, 0.032]` | 0.114660 |
| HardPair-FT versus Base | `acc05` | 1000 | 0.016 | `[-0.001, 0.033]` | 0.080507 |
| HardPair-FT versus Random-FT | `pair_accuracy` | 500 | 0.000 | `[-0.016, 0.016]` | 1.000000 |
| HardPair-FT versus Random-FT | `wrong_instance` | 500 | -0.006 | `[-0.016, 0.004]` |  |
| HardPair-FT versus Random-FT | `same_box_collapse` | 500 | -0.004 | `[-0.020, 0.010]` | 0.790527 |

Tóm tắt: cả hai mô hình fine-tuned cải thiện Vietnamese standard grounding so với base. HardPair-FT giữ `Pair Accuracy` bằng Random-FT trên hard pairs, nhưng giảm nhẹ `Wrong-Instance` và `Same-Box Collapse`. English performance giảm nhẹ sau Vietnamese fine-tuning, nên cần báo cáo trade-off này.

## Metrics cần báo cáo

Standard benchmark:

- `mIoU`
- `Acc@0.5`
- `Acc@0.75`
- `Parse Fail`
- `Multi-box`
- Vietnamese grounding gap

Hard-pair benchmark:

- `Pair Accuracy`
- `Pair mIoU`
- `Wrong-Instance`
- `Same-Box Collapse`
- `Parse Fail`

Các so sánh chính:

- `Random-FT` so với `Base` trên Vietnamese `Acc@0.5`.
- `HardPair-FT` so với `Base` trên Vietnamese `Acc@0.5`.
- `HardPair-FT` so với `Random-FT` trên `Pair Accuracy`, `Wrong-Instance`, `Same-Box Collapse`.
- English retention sau fine-tuning.

## Ghi chú tái lập

- Dataset được coi là nguồn dữ liệu cố định; không regenerate, retranslate, remine hoặc sửa dataset trong workflow final.
- Nếu thiếu file hoặc ảnh được reference, workflow nên fail thay vì tự sửa dữ liệu.
- Training của `random_ft` và `hardpair_ft` chỉ nên khác dữ liệu training; các cấu hình còn lại cần giữ giống nhau.
- Các notebook `fine-tune-llm.ipynb` và `test_base_model.ipynb` dùng cho exploration/kiểm thử, còn pipeline chính nằm trong `scripts/`.
