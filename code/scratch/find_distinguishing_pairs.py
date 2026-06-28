import json
from pathlib import Path

eval_dir = Path(r"c:\Users\Admin\Desktop\Computer Vision\final\code\artifacts\evaluation")
models = ["base", "random_ft", "hardpair_ft", "grounding_dino", "qwen2_5_vl_3b"]
hp_path = r"c:\Users\Admin\Desktop\Computer Vision\final\code\.hf_cache\viground\data\benchmark\hard_pairs.jsonl"

def compute_iou(box1, box2):
    if not box1 or not box2:
        return 0.0
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    xi1, yi1 = max(x11, x21), max(y11, y21)
    xi2, yi2 = min(x12, x22), min(y12, y22)
    if xi1 >= xi2 or yi1 >= yi2:
        return 0.0
    intersection = (xi2 - xi1) * (yi2 - yi1)
    union = (x12 - x11) * (y12 - y11) + (x22 - x21) * (y22 - y21) - intersection
    return intersection / union if union > 0 else 0.0

def load_preds(model_name):
    preds = {}
    path = eval_dir / model_name / "predictions.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            pred_box = None
            if data.get("prediction") and data["prediction"].get("parse_ok"):
                pixel_boxes = data["prediction"].get("pixel", [])
                if pixel_boxes:
                    pred_box = pixel_boxes[0]
            preds[data["id"]] = pred_box
    return preds

all_preds = {m: load_preds(m) for m in models}

with open(hp_path, "r", encoding="utf-8") as f:
    hard_pairs = [json.loads(line) for line in f]

results = []
for idx, pair in enumerate(hard_pairs):
    for q_name, q_data, q_id in [("A", pair["sample_a"], pair["sample_a"]["sample_id"]), 
                                 ("B", pair["sample_b"], pair["sample_b"]["sample_id"])]:
        if not all(q_id in all_preds[m] for m in models):
            continue
        ious = {m: compute_iou(all_preds[m][q_id], q_data["bbox_xyxy"]) for m in models}
        if ious["random_ft"] > 0.7 and ious["hardpair_ft"] > 0.7:
            if ious["base"] < 0.3 and ious["grounding_dino"] < 0.3 and ious["qwen2_5_vl_3b"] < 0.3:
                results.append({
                    "idx": idx,
                    "query_name": q_name,
                    "category": pair["category_name"],
                    "expression": q_data["expression_vi"],
                    "image": q_data["file_name"],
                    "ious": ious
                })

print(f"Total matching cases: {len(results)}")
for r in sorted(results, key=lambda x: min(x["ious"]["random_ft"], x["ious"]["hardpair_ft"]) - max(x["ious"]["base"], x["ious"]["grounding_dino"], x["ious"]["qwen2_5_vl_3b"]), reverse=True):
    print(f"Index: {r['idx']} (Query {r['query_name']}) | Category: {r['category'].upper()} | Image: {r['image']}")
    print(f"  Query: '{r['expression']}'")
    print(f"  IoU values -> Base: {r['ious']['base']:.3f} | GDino: {r['ious']['grounding_dino']:.3f} | Qwen: {r['ious']['qwen2_5_vl_3b']:.3f} | Rand_FT: {r['ious']['random_ft']:.3f} | Hard_FT: {r['ious']['hardpair_ft']:.3f}")
    print("-" * 80)
