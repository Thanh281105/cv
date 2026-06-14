# Final Research Plan: ViGround-Contrast

## 1. Objective

Complete the ViGround-Contrast research experiment using the already prepared Hugging Face dataset and the verified LocateAnything-3B base-model pilot.

The core research question is:

> Can targeted Vietnamese hard-pair adaptation reduce the cross-lingual grounding gap and improve same-category instance discrimination more effectively than matched random Vietnamese fine-tuning?

Do not regenerate, retranslate, remine, repair, or otherwise modify the dataset. The current Hugging Face snapshot is the fixed source of truth.

---

## 2. Fixed Revisions

Use these exact revisions for every experiment:

```text
Dataset:
thanhhoangnvbg/viground-contrast-data

Dataset revision:
eb7ddb1069de0c6e7a6277cf46b3d94d0c9cd9ee

Base model:
nvidia/LocateAnything-3B

Model revision:
c32291ca5e996f5a7a485845b4f57a233936bba0

Random seed:
42
```

Never use `revision="main"` during final runs.

At startup, save:

```text
dataset_revision.txt
model_revision.txt
environment.json
repository_file_manifest.json
dataset_integrity_report.json
```

The integrity report must record the actual remote counts. Expected counts from the current experiment are:

```text
Random-FT training samples:       4,000
HardPair-FT training samples:     4,000
HardPair-FT training pairs:       2,000
Standard benchmark rows:          2,000
  English:                        1,000
  Vietnamese:                     1,000
Hard-pair benchmark pairs:          500
Images in fixed bundle:           5,024
```

Fail immediately if required files or referenced images are missing. Do not repair the remote dataset during the training workflow.

---

## 3. Established Base-Model Pilot

Treat the following as preliminary empirical evidence already obtained from the fixed dataset snapshot.

The pilot used:

```text
GPU: Tesla T4
Standard English samples: 200
Standard Vietnamese samples: 200
Vietnamese hard pairs: 200
Unique inference requests: 800
Generation mode: hybrid
Sampling: disabled
Seed: 42
```

Observed base metrics:

```text
English:
mIoU                0.8371
Acc@0.5             0.9100
Acc@0.75            0.8250
Parse failure       0.0400
Multi-box failure   0.0400

Vietnamese:
mIoU                0.6867
Acc@0.5             0.7300
Acc@0.75            0.6200
Parse failure       0.0750
Multi-box failure   0.0650

Vietnamese Grounding Gap:
Acc@0.5 gap         0.1800

Vietnamese hard pairs:
Pair Accuracy       0.5200
Pair mIoU           0.4948
Wrong-instance      0.1250
Same-box collapse   0.1150
Parse failure       0.0450
Runtime error       0.0000
Cross-image pairs   0
```

Import these results into:

```text
results/base_pilot/pilot_result.json
```

Clearly label them as pilot results, not final full-benchmark results.

---

## 4. Research Hypotheses

Evaluate the following hypotheses.

### H1: Vietnamese adaptation

Both Vietnamese fine-tuned models improve Vietnamese standard grounding over the base model.

Primary metric:

```text
Vietnamese Acc@IoU 0.5
```

Secondary metrics:

```text
Vietnamese mIoU
Vietnamese Acc@0.75
Parse failure rate
Multi-box failure rate
```

### H2: Hard-pair advantage

HardPair-FT outperforms Random-FT on same-category instance discrimination.

Primary hard-pair metrics:

```text
Pair Accuracy
Wrong-instance Rate
Same-box Collapse Rate
```

Expected direction:

```text
HardPair-FT Pair Accuracy       > Random-FT
HardPair-FT Wrong-instance      < Random-FT
HardPair-FT Same-box Collapse   < Random-FT
```

### H3: Generalization trade-off

HardPair-FT should not gain hard-pair performance only by severely damaging ordinary Vietnamese grounding.

Report:

```text
HardPair-FT standard VI Acc@0.5
minus
Random-FT standard VI Acc@0.5
```

A standard Vietnamese drop larger than 2 percentage points must be discussed as a trade-off.

### H4: English retention

Vietnamese adaptation should not catastrophically damage English grounding.

Report:

```text
Base English Acc@0.5
Random-FT English Acc@0.5
HardPair-FT English Acc@0.5
```

An English drop larger than 3 percentage points must be explicitly discussed.

---

## 5. Experimental Models

Evaluate exactly three models:

```text
1. Base
   nvidia/LocateAnything-3B
   No fine-tuning

2. Random-FT
   Base plus PEFT trained on 4,000 matched random Vietnamese samples

3. HardPair-FT
   Base plus identical PEFT method trained on 2,000 hard pairs
   Total 4,000 Vietnamese samples
```

Random-FT and HardPair-FT must differ only in training data.

They must share:

```text
base checkpoint
model revision
trainable modules
optimizer
learning rate
epochs
effective batch size
seed
precision
image processing
prompt template
output format
checkpoint policy
evaluation runtime
```

Do not alter one training configuration after seeing the other model’s results.

---

## 6. Adaptation Method

Use parameter-efficient fine-tuning rather than full-model fine-tuning.

### Primary method

Apply LoRA to the Qwen language-model attention projections:

```text
q_proj
k_proj
v_proj
o_proj
```

Configuration:

```text
LoRA rank:       8
LoRA alpha:      16
LoRA dropout:    0.05
Bias:            none
```

Freeze:

```text
MoonViT vision encoder
multimodal projector
PBD output heads
base Qwen weights
token embeddings
```

Only LoRA parameters may receive gradients.

Scientific motivation:

```text
Adapt Vietnamese query interpretation and cross-modal attention
while preserving the pretrained visual localization and PBD output system.
```

Do not use a generic text-only SFT trainer. Preserve the official LocateAnything multimodal and multi-token prediction loss path.

Use the NVIDIA/Eagle LocateAnything training entry point as the foundation:

```text
eaglevl/train/locany_finetune_magi_stream.py
```

Patch it minimally to inject PEFT LoRA while preserving its native forward pass and loss.

Before training, generate:

```text
trainable_parameters.json
```

This file must include:

```text
parameter name
shape
number of elements
requires_grad
module category
```

Assertions:

```text
At least one LoRA parameter is trainable.
All LoRA parameters receive non-zero finite gradients.
No frozen visual, projector, PBD, or base LLM parameter receives gradients.
Random-FT and HardPair-FT have identical trainable parameter manifests.
```

Do not silently fall back to projector-only training, full fine-tuning, or QLoRA. Stop and report incompatibility if native LoRA integration cannot preserve the LocateAnything training loss.

---

## 7. Vertex AI L4 Configuration

Run both jobs on Vertex AI using one NVIDIA L4 GPU.

Preferred worker:

```text
machine_type: g2-standard-8
accelerator_type: NVIDIA_L4
accelerator_count: 1
replica_count: 1
```

Use a custom training container with persistent output storage in Google Cloud Storage.

Required environment:

```text
Linux
PyTorch compatible with the installed CUDA driver
transformers==4.57.1
accelerate
peft
safetensors
huggingface_hub
Pillow==11.1.0
opencv-python-headless==4.11.0.86
decord==0.6.0
lmdb==1.7.5
```

Use:

```text
attn_implementation: sdpa
causal_attn: false
block_size: 6
grad_checkpoint: true
```

Do not install or use Magi Attention on L4.

Use BF16 only when:

```python
torch.cuda.is_bf16_supported()
```

Otherwise use FP16.

Preflight must print:

```text
GPU name
compute capability
VRAM
CUDA version
PyTorch version
Transformers version
dataset revision
model revision
trainable parameter count
maximum allocated VRAM during dry run
```

---

## 8. Training Hyperparameters

Use the same configuration for both models:

```text
epochs:                         1
per_device_train_batch_size:   1
gradient_accumulation_steps:   8
effective batch size:          8
learning_rate:                 1e-4
weight_decay:                  0.01
warmup_ratio:                  0.05
scheduler:                     cosine
max_grad_norm:                 1.0
seed:                          42
data_seed:                     42
dataloader_num_workers:        4
data augmentation:             false
sampling/repeat_time:          1.0
```

L4 sequence settings:

```text
max_seq_length:                4096
max_num_tokens_per_sample:     4096
max_num_tokens:                4096
packing_buffer_size:           8
```

Before the full job, calculate token-length statistics for both training sets:

```text
minimum
median
p90
p95
p99
maximum
number exceeding 4096
```

The two datasets must use the same truncation and filtering policy.

No training row may be silently skipped. Log all skipped sample IDs and reasons.

Save checkpoints at approximately:

```text
25%
50%
75%
100%
```

The final research model is the fixed one-epoch checkpoint. Do not select a checkpoint using test benchmark performance.

Checkpointing is for recovery and diagnostics only.

---

## 9. Training Smoke Gate

Before running either full job, run a two-batch training smoke test.

Pass criteria:

```text
loss is finite
loss.backward succeeds
LoRA gradient norm is finite and non-zero
frozen parameter gradients are absent
optimizer.step succeeds
checkpoint save succeeds
checkpoint reload succeeds
one inference after reload succeeds
exactly one valid box is returned
```

Then run 50 optimizer steps on Random-FT as a systems test.

This is not a hyperparameter search. Do not evaluate research benchmark metrics at step 50.

After the systems test passes, run the two complete training jobs with identical settings.

---

## 10. Evaluation Protocol

Convert the verified notebook evaluation logic into reproducible scripts.

Use:

```text
generation_mode: hybrid
max_new_tokens: 2048
do_sample: false
seed: 42
single-instance prompt only
```

Prompt:

```text
Locate a single instance that matches the following description: {query}.
```

Prediction rules:

```text
zero valid boxes      -> task failure
exactly one box       -> valid prediction
more than one box     -> task/format failure
```

Never select the first box from a multi-box output.

Use the same parser, coordinate conversion, IoU implementation, and image processor for all three models.

### Full standard benchmark

Evaluate every model on:

```text
1,000 English rows
1,000 aligned Vietnamese rows
```

Metrics:

```text
mIoU
Acc@0.5
Acc@0.75
Parse Failure Rate
Multi-box Failure Rate
Runtime Error Rate
Mean and median latency
Peak VRAM
```

Compute for every model:

```text
Vietnamese Grounding Gap
= English Acc@0.5 - Vietnamese Acc@0.5
```

Compute:

```text
Gap Closure
= Base Grounding Gap - Fine-tuned Grounding Gap
```

### Full hard-pair benchmark

Evaluate all 500 fixed hard pairs in Vietnamese:

```text
1,000 Vietnamese query requests per model
```

Additionally evaluate the same 500 pairs in English by creating an evaluation-only language view from `expression_en`.

Do not modify the dataset files. Construct the English view in memory.

This yields:

```text
1,000 Vietnamese hard-pair queries per model
1,000 English hard-pair queries per model
```

Hard-pair metrics:

```text
Query Acc@0.5
Query mIoU
Pair Accuracy
Pair mIoU using min(IoU_A, IoU_B)
Wrong-instance Rate
Same-box Collapse Rate
Parse Failure Rate
Multi-box Failure Rate
Runtime Error Rate
```

Wrong-instance definition:

```text
The prediction overlaps the paired distractor more than the target.
```

Same-box collapse definition:

```text
Predicted boxes for A and B have IoU >= 0.7
while the two ground-truth boxes have IoU < 0.3.
```

Total final inference budget:

```text
Standard:          2,000 requests per model
Hard pairs EN:     1,000 requests per model
Hard pairs VI:     1,000 requests per model

Total:             4,000 requests per model
Three models:     12,000 requests
```

Checkpoint predictions after every request so evaluation can resume without repetition.

---

## 11. Error Analysis

Break down results by:

```text
language
source split
category
bounding-box size
sentence length
spatial versus non-spatial
discriminative group
```

Discriminative groups should include, when available:

```text
left/right
top/bottom
front/behind
middle/center
nearest/farthest
standing/sitting
color
size
action
ordinal
relation
```

Produce transition analyses on aligned samples:

```text
Base wrong -> Random correct
Base wrong -> HardPair correct
Random wrong -> HardPair correct
Base correct -> fine-tuned wrong
```

For hard pairs, identify:

```text
Base same-box collapse resolved by HardPair-FT
Base wrong-instance resolved by HardPair-FT
Random-FT succeeds but HardPair-FT fails
HardPair-FT succeeds but Random-FT fails
```

Save at least 24 qualitative visualizations:

```text
6 Vietnamese gap examples
6 wrong-instance examples
6 same-box-collapse examples
6 comparative success/failure examples
```

Each visualization must contain:

```text
image
English expression
Vietnamese expression
ground-truth target
paired distractor when applicable
Base prediction
Random-FT prediction
HardPair-FT prediction
IoU values
```

---

## 12. Statistical Analysis

All model comparisons are paired because the models are evaluated on identical samples.

Use 10,000 paired bootstrap resamples.

For standard metrics, resample aligned source expressions.

For hard-pair metrics, resample entire pairs, never individual pair members.

Report 95% confidence intervals for:

```text
Acc@0.5
mIoU
Vietnamese Grounding Gap
Pair Accuracy
Wrong-instance Rate
Same-box Collapse Rate
```

Primary comparisons:

```text
Random-FT versus Base on Vietnamese Acc@0.5
HardPair-FT versus Base on Vietnamese Acc@0.5
HardPair-FT versus Random-FT on Pair Accuracy
HardPair-FT versus Random-FT on Wrong-instance Rate
HardPair-FT versus Random-FT on Same-box Collapse Rate
```

Also run McNemar tests for paired binary correctness outcomes.

Do not claim superiority when the confidence interval for the paired difference includes zero. Report the direction and uncertainty honestly.

---

## 13. Required Final Tables

Generate these tables automatically.

### Table 1: Dataset

```text
split
samples
pairs
unique images
unique categories
language
translation-valid rate
```

### Table 2: Base pilot

Use the already completed 200 EN, 200 VI, and 200 hard-pair pilot.

### Table 3: Full standard benchmark

```text
model
language
n
mIoU
Acc@0.5
Acc@0.75
parse failure
multi-box failure
grounding gap
```

### Table 4: Full hard-pair benchmark

```text
model
language
pairs
query Acc@0.5
Pair Accuracy
Pair mIoU
Wrong-instance
Same-box Collapse
parse failure
```

### Table 5: Paired effects

```text
comparison
metric
absolute difference
95% CI
p-value
```

### Table 6: Training

```text
model
train samples
trainable parameters
optimizer steps
training loss
runtime
peak VRAM
checkpoint size
```

---

## 14. Required Figures

Generate:

```text
1. English versus Vietnamese Acc@0.5 for all models
2. Vietnamese Grounding Gap before and after adaptation
3. Pair Accuracy for Base, Random-FT, HardPair-FT
4. Wrong-instance and Same-box Collapse rates
5. Per-discriminative-group performance
6. Training-loss curves for Random-FT and HardPair-FT
7. Qualitative comparison grid
```

Use identical axes and scales when comparing models.

---

## 15. Output Structure

Use:

```text
artifacts/
├── frozen_revisions/
├── dataset_audit/
├── training/
│   ├── random_ft/
│   └── hardpair_ft/
├── evaluation/
│   ├── base/
│   ├── random_ft/
│   └── hardpair_ft/
├── statistics/
├── figures/
├── tables/
├── qualitative/
└── report/
```

Each model evaluation directory must include:

```text
predictions.jsonl
standard_metrics.json
hard_pair_metrics.json
latency_stats.json
failure_cases.jsonl
run_manifest.json
```

Each training directory must include:

```text
resolved_config.yaml
environment.json
trainable_parameters.json
training_log.jsonl
trainer_state.json
checkpoints/
final_adapter/
```

Upload final adapters privately to:

```text
thanhhoangnvbg/ViGround-Contrast-Random-FT
thanhhoangnvbg/ViGround-Contrast-HardPair-FT
```

Retain NVIDIA model licensing and attribution information.

---

## 16. Research Conclusions

The final conclusion must be selected based on results, not written in advance.

Possible valid outcomes include:

### Outcome A

HardPair-FT improves Vietnamese standard grounding and clearly outperforms Random-FT on hard-pair metrics.

Conclusion:

```text
Targeted hard-pair adaptation improves both Vietnamese grounding
and same-category semantic discrimination.
```

### Outcome B

Random-FT and HardPair-FT both improve Vietnamese grounding, but HardPair-FT only improves pair metrics.

Conclusion:

```text
General Vietnamese adaptation closes the language gap,
while hard-pair selection provides an additional discrimination benefit.
```

### Outcome C

HardPair-FT improves pair metrics but slightly hurts standard grounding.

Conclusion:

```text
Hard-pair adaptation introduces a measurable specialization trade-off.
```

### Outcome D

Base remains best overall.

Conclusion:

```text
Small machine-translated adaptation does not reliably outperform
the strong foundation model, although hard-pair selection may reduce
specific instance-confusion errors.
```

Do not hide negative results.

---

## 17. Execution Order

Execute in this order:

```text
1. Freeze and audit exact HF and model revisions
2. Convert notebook logic into reproducible evaluation scripts
3. Register base pilot results
4. Build Vertex AI custom container
5. Run two-batch training smoke test
6. Run 50-step systems test
7. Train Random-FT for exactly one epoch
8. Train HardPair-FT for exactly one epoch
9. Reload both adapters and run inference smoke tests
10. Run full Base evaluation on the pinned benchmark
11. Run full Random-FT evaluation
12. Run full HardPair-FT evaluation
13. Compute paired statistics
14. Generate tables, figures, and qualitative examples
15. Write the final research report
```

Do not modify the dataset during these stages.

Do not substitute simulated metrics for real model predictions.

Do not claim completion until all three models have real predictions on the same pinned benchmark and all required artifacts are generated.
