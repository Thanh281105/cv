Dưới đây là thứ tự chạy trên VM. Chạy từng block, nếu block gate báo SMOKE FAILED thì dừng, đừng train full.
1. Set env
cd ~/viground-final

export PYTHONPATH=$PWD:/opt/Eagle/Embodied:$PYTHONPATH
export EAGLE_ROOT=/opt/Eagle/Embodied
export LAUNCHER=pytorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
2. Patch config cuối: 2048, lr thấp
python3 - <<'PY'
from pathlib import Path
import re

p = Path("scripts/run_vertex_locany_training.py")
text = p.read_text()
bak = p.with_suffix(".py.before_final_2048_retrain.bak")
if not bak.exists():
    bak.write_text(text)

def replace_pair(flag, value):
    global text
    pat = re.compile(rf'("--{re.escape(flag)}",\s*\n\s*)("[^"]*")')
    text, n = pat.subn(rf'\1"{value}"', text, count=1)
    print(flag, "->", value, "patched" if n else "WARN missing")

replace_pair("learning_rate", "2e-5")
replace_pair("warmup_ratio", "0.1")
replace_pair("max_seq_length", "2048")
replace_pair("max_num_tokens_per_sample", "2048")
replace_pair("max_num_tokens", "2048")

text = re.sub(
    r'("full":\s*\{\s*"max_steps":\s*)None',
    r'\g<1>500',
    text,
)
text = re.sub(
    r"('full':\s*\{\s*'max_steps':\s*)None",
    r'\g<1>500',
    text,
)

if 'args.mode == "full" and "--max_steps" not in command' not in text:
    text = text.replace(
        "    subprocess.run(command, check=True)\n",
        '    if args.mode == "full" and "--max_steps" not in command:\n'
        '        command.extend(["--max_steps", "500"])\n\n'
        "    subprocess.run(command, check=True)\n",
    )

p.write_text(text)
print("final retrain config ready")
PY
3. Tạo smoke checker
smoke_eval () {
  root="$1"
  model="$2"

  rm -rf "$root/evaluation/$model"

  python3 scripts/evaluate_locateanything.py \
    --artifact-root "$root" \
    --model-key "$model" \
    --limit-standard 5 \
    --limit-hard-pairs 2

  python3 - "$root" "$model" <<'PY'
import json, sys
from pathlib import Path

root, model = sys.argv[1], sys.argv[2]
rows = [json.loads(x) for x in Path(f"{root}/evaluation/{model}/predictions.jsonl").open()]
ok = sum(bool(r["prediction"].get("parse_ok")) for r in rows)
print(model, "parse_ok", ok, "/", len(rows))
for r in rows[:5]:
    print(repr(r["prediction"].get("raw", "")))

if ok == 0:
    raise SystemExit("SMOKE FAILED: no valid boxes")
if all(str(r["prediction"].get("raw", "")).endswith("<box>(") for r in rows[:5]):
    raise SystemExit("SMOKE FAILED: generation still stops at <box>(")
PY
}
4. Gate trước full
GATE=artifacts_gate_final_2048
rm -rf "$GATE"

python3 scripts/run_vertex_locany_training.py --kind random_ft --mode smoke --artifact-root "$GATE"
smoke_eval "$GATE" random_ft

rm -rf "$GATE"
python3 scripts/run_vertex_locany_training.py --kind hardpair_ft --mode smoke --artifact-root "$GATE"
smoke_eval "$GATE" hardpair_ft

rm -rf "$GATE"
python3 scripts/run_vertex_locany_training.py --kind random_ft --mode systems-test --artifact-root "$GATE"
smoke_eval "$GATE" random_ft
5. Nếu cả 3 gate pass, train full
FINAL=artifacts_final_retrain_2048
rm -rf "$FINAL"

python3 scripts/run_vertex_locany_training.py --kind random_ft --mode full --artifact-root "$FINAL"
smoke_eval "$FINAL" random_ft

python3 scripts/run_vertex_locany_training.py --kind hardpair_ft --mode full --artifact-root "$FINAL"
smoke_eval "$FINAL" hardpair_ft
6. Nếu full smoke pass, benchmark
python3 scripts/evaluate_locateanything.py --artifact-root "$FINAL" --model-key base
python3 scripts/evaluate_locateanything.py --artifact-root "$FINAL" --model-key random_ft
python3 scripts/evaluate_locateanything.py --artifact-root "$FINAL" --model-key hardpair_ft

python3 scripts/compute_statistics.py --artifact-root "$FINAL"
python3 scripts/generate_tables.py --artifact-root "$FINAL"
7. Xem kết quả
cat "$FINAL/tables/table3_full_standard_benchmark.csv"
cat "$FINAL/tables/table4_full_hard_pair_benchmark.csv"
cat "$FINAL/tables/table5_paired_effects.csv"
Chốt: nếu gate vẫn ra '<ref>...</ref><box>(', dừng ngay. Nếu gate ra box hợp lệ, mới đáng train full.