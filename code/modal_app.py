import modal

# 1. Define Modal App and container requirements
app = modal.App("viground-contrast-demo")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate",
        "datasets",
        "decord==0.6.0",
        "huggingface_hub",
        "lmdb==1.7.5",
        "matplotlib",
        "numpy",
        "opencv-python-headless==4.11.0.86",
        "pandas",
        "peft",
        "Pillow==11.1.0",
        "PyYAML",
        "qwen-vl-utils==0.0.8",
        "safetensors",
        "scipy",
        "torch",
        "torchvision",
        "tqdm",
        "transformers==4.57.1",
        "gradio",
    )
)

volume = modal.Volume.from_name("hf-cache-volume", create_if_missing=True)

# 2. Main ASGI app function
@app.function(
    image=image,
    gpu="A100",
    volumes={"/cache": volume},
    env={"HF_HOME": "/cache"},
    timeout=3600,
    min_containers=1,
    max_containers=1,
)
@modal.asgi_app()
def ui():
    # Heavy ML packages are imported inside the function to prevent local ModuleNotFoundError
    import io
    import re
    import os
    import zipfile
    import json
    import torch
    import gradio as gr
    import pandas as pd
    from pathlib import Path
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from fastapi import FastAPI
    from huggingface_hub import snapshot_download, hf_hub_download
    from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoModelForZeroShotObjectDetection, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    # Config & constants matching Final_CV.ipynb
    BASE_REPO = "nvidia/LocateAnything-3B"
    BASE_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"

    MODEL_REPOS = {
        "base": None,
        "random_ft": "thanhhoangnvbg/viground-random-ft-locany",
        "hardpair_ft": "thanhhoangnvbg/viground-hardpair-ft-locany",
    }

    PROMPT_TEMPLATE = "Locate a single instance that matches the following description: {}."
    MAX_NEW_TOKENS = 128
    GENERATION_MODE = "hybrid"
    DO_SAMPLE = False

    GROUNDING_DINO_REPO = "IDEA-Research/grounding-dino-base"
    QWEN_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
    QWEN_MIN_PIXELS = 256 * 28 * 28
    QWEN_MAX_PIXELS = 2048 * 28 * 28

    # Helper functions for loading model
    def patch_lora_targets(local_dir):
        local_dir = Path(local_dir)
        pattern = re.compile(
            r"target_modules=\[\s*['\"]self_attn\.q_proj['\"],\s*['\"]self_attn\.k_proj['\"],\s*['\"]self_attn\.v_proj['\"],\s*['\"]self_attn\.o_proj['\"],\s*['\"]mlp\.gate_proj['\"],\s*['\"]mlp\.down_proj['\"],\s*['\"]mlp\.up_proj['\"]\s*\]",
            flags=re.S,
        )
        replacement = "target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']"
        changed = 0
        for p in local_dir.glob("**/modeling_locateanything.py"):
            text = p.read_text(errors="ignore")
            new_text, n = pattern.subn(replacement, text, count=1)
            if n:
                p.write_text(new_text)
                changed += n
        print("patched lora target files:", changed)

    def remap_lora_checkpoint_keys_for_loaded_model(state, model):
        model_keys = set(model.state_dict().keys())
        remapped = {}
        lora_total = 0
        before = 0
        after = 0
        for key, value in state.items():
            new_key = key
            if "lora_" in key:
                lora_total += 1
                if key in model_keys:
                    before += 1
                else:
                    candidate = key.replace(
                        "language_model.base_model.model.",
                        "language_model.base_model.model.base_model.model.",
                        1,
                    )
                    if candidate in model_keys:
                        new_key = candidate
            if "lora_" in new_key and new_key in model_keys:
                after += 1
            remapped[new_key] = value
        print(f"LoRA key matches after remap: {after}/{lora_total}")
        return remapped

    def load_checkpoint_state(checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        state = {}
        shards = sorted(checkpoint_dir.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No *.safetensors found in {checkpoint_dir}")
        from safetensors.torch import load_file
        for shard in shards:
            state.update(load_file(str(shard), device="cpu"))
        return state

    # Helper functions for inference
    def build_prompt(query):
        return PROMPT_TEMPLATE.format(query)

    def extract_answer(tokenizer, response):
        candidate = response[0] if isinstance(response, tuple) else response
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, (list, tuple)):
            if not candidate:
                return ""
            if isinstance(candidate[0], str):
                return candidate[0]
            candidate = candidate[0]
        if torch.is_tensor(candidate):
            if candidate.ndim == 1:
                candidate = candidate.unsqueeze(0)
            return tokenizer.batch_decode(candidate, skip_special_tokens=False)[0]
        return str(candidate)

    def predict_one(tokenizer, processor, model, image, query):
        image = image.convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_prompt(query)},
            ],
        }]
        text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = processor.process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt")
        inputs = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inputs.items()}
        response = model.generate(
            pixel_values=inputs["pixel_values"].to(torch.float16),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            generation_mode=GENERATION_MODE,
            do_sample=DO_SAMPLE,
            verbose=False,
        )
        return extract_answer(tokenizer, response)

    def parse_box(answer, image_width, image_height):
        nums = re.findall(r"<box>\s*<([+-]?\d+(?:\.\d+)?)><([+-]?\d+(?:\.\d+)?)><([+-]?\d+(?:\.\d+)?)><([+-]?\d+(?:\.\d+)?)>\s*</box>", answer)
        if not nums:
            nums = re.findall(r"<box>\s*\(?\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)?\s*</box>", answer)
        if not nums:
            return None
        x1, y1, x2, y2 = map(float, nums[0])
        x1 = int(round(x1 / 1000 * image_width))
        y1 = int(round(y1 / 1000 * image_height))
        x2 = int(round(x2 / 1000 * image_width))
        y2 = int(round(y2 / 1000 * image_height))
        return [x1, y1, x2, y2]

    BOX_PATTERNS = (
        re.compile(r'"(?:bbox|bbox_2d|box|bounding_box)"\s*:\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]', re.I),
        re.compile(r'\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]'),
    )

    def normalized_from_pixel(box, width, height):
        return [
            int(round(max(0, min(box[0] / width * 1000, 1000)))),
            int(round(max(0, min(box[1] / height * 1000, 1000)))),
            int(round(max(0, min(box[2] / width * 1000, 1000)))),
            int(round(max(0, min(box[3] / height * 1000, 1000)))),
        ]

    def parse_qwen_box(answer, image_width, image_height):
        box = parse_box(answer, image_width, image_height)
        if box:
            return box
        values = None
        for pattern in BOX_PATTERNS:
            match = pattern.search(answer or "")
            if match:
                values = [float(match.group(i)) for i in range(1, 5)]
                break
        if values is None:
            return None
        x1, y1, x2, y2 = values
        if max(values) <= 1.0:
            values = [v * 1000 for v in values]
            return [
                int(round(values[0] / 1000 * image_width)),
                int(round(values[1] / 1000 * image_height)),
                int(round(values[2] / 1000 * image_width)),
                int(round(values[3] / 1000 * image_height)),
            ]
        if x2 <= image_width and y2 <= image_height:
            return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
        if all(0 <= v <= 1000 for v in values):
            return [
                int(round(x1 / 1000 * image_width)),
                int(round(y1 / 1000 * image_height)),
                int(round(x2 / 1000 * image_width)),
                int(round(y2 / 1000 * image_height)),
            ]
        return None

    # Shared mutable state inside closure
    models_dict = {}

    def load_all_models_concurrently():
        if models_dict:
            return
        
        print("=" * 80)
        print("INITIALIZING ALL 5 MODELS CONCURRENTLY IN VRAM (A100)...")
        print("=" * 80)
        
        base_dir = snapshot_download(
            BASE_REPO,
            revision=BASE_REVISION,
            local_dir="/cache/models/base_locateanything_3b",
        )
        patch_lora_targets(base_dir)
        
        # 1. Base LocateAnything
        print("[1/5] Loading LocateAnything Base...")
        tokenizer_base = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
        processor_base = AutoProcessor.from_pretrained(base_dir, trust_remote_code=True)
        model_base = AutoModel.from_pretrained(
            base_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        model_base.eval()
        models_dict["base"] = (tokenizer_base, processor_base, model_base)
        
        # 2. Random-FT
        print("[2/5] Loading LocateAnything Random-FT...")
        tokenizer_rand = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
        processor_rand = AutoProcessor.from_pretrained(base_dir, trust_remote_code=True)
        model_rand = AutoModel.from_pretrained(
            base_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        model_rand.wrap_llm_lora(r=8, lora_alpha=16, lora_dropout=0.05)
        ft_dir_rand = snapshot_download(MODEL_REPOS["random_ft"], local_dir="/cache/models/random_ft")
        state_rand = load_checkpoint_state(ft_dir_rand)
        state_rand = remap_lora_checkpoint_keys_for_loaded_model(state_rand, model_rand)
        model_rand.load_state_dict(state_rand, strict=False)
        model_rand.eval()
        models_dict["random_ft"] = (tokenizer_rand, processor_rand, model_rand)
        
        # 3. HardPair-FT
        print("[3/5] Loading LocateAnything HardPair-FT...")
        tokenizer_hard = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
        processor_hard = AutoProcessor.from_pretrained(base_dir, trust_remote_code=True)
        model_hard = AutoModel.from_pretrained(
            base_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        model_hard.wrap_llm_lora(r=8, lora_alpha=16, lora_dropout=0.05)
        ft_dir_hard = snapshot_download(MODEL_REPOS["hardpair_ft"], local_dir="/cache/models/hardpair_ft")
        state_hard = load_checkpoint_state(ft_dir_hard)
        state_hard = remap_lora_checkpoint_keys_for_loaded_model(state_hard, model_hard)
        model_hard.load_state_dict(state_hard, strict=False)
        model_hard.eval()
        models_dict["hardpair_ft"] = (tokenizer_hard, processor_hard, model_hard)
        
        # 4. Grounding DINO
        print("[4/5] Loading Grounding DINO...")
        gd_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_REPO)
        gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_REPO).to("cuda").eval()
        models_dict["grounding_dino"] = (gd_processor, gd_model)
        
        # 5. Qwen2.5-VL-3B
        print("[5/5] Loading Qwen2.5-VL-3B...")
        qwen_processor = AutoProcessor.from_pretrained(
            QWEN_REPO,
            min_pixels=QWEN_MIN_PIXELS,
            max_pixels=QWEN_MAX_PIXELS,
        )
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_REPO,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()
        models_dict["qwen2_5_vl_3b"] = (qwen_processor, qwen_model)
        print("All models loaded successfully!")

    # 1. Ensure models are loaded
    load_all_models_concurrently()
    
    # 2. Download benchmark files to Volume if not present
    hp_path = Path("/cache/viground/data/benchmark/hard_pairs.jsonl")
    zip_path = Path("/cache/viground/data/images.zip")
    
    os.makedirs("/cache/viground/data/benchmark", exist_ok=True)
    os.makedirs("/cache/viground/data", exist_ok=True)
    
    if not hp_path.exists():
        print("Downloading hard_pairs.jsonl to volume...")
        hf_hub_download(
            repo_id="thanhhoangnvbg/viground-contrast-data",
            repo_type="dataset",
            filename="data/benchmark/hard_pairs.jsonl",
            local_dir="/cache/viground",
        )
    if not zip_path.exists():
        print("Downloading images.zip to volume...")
        hf_hub_download(
            repo_id="thanhhoangnvbg/viground-contrast-data",
            repo_type="dataset",
            filename="data/images.zip",
            local_dir="/cache/viground",
        )
        
    # 3. Read hard pairs
    with open(hp_path, "r", encoding="utf-8") as f:
        hard_pairs = [json.loads(line) for line in f]
        
    preset_choices = [
        f"[{i}] {pair['category_name'].upper()}: {pair['sample_a']['expression_vi']} vs {pair['sample_b']['expression_vi']}" 
        for i, pair in enumerate(hard_pairs)
    ]
    
    def run_gradio_inference(input_mode, preset_str, query_selection, custom_query, custom_img):
        # 1. Resolve image
        if input_mode == "Preset Hard Pair":
            idx = int(preset_str.split("]")[0].replace("[", "").strip())
            pair = hard_pairs[idx]
            sample_a = pair["sample_a"]
            sample_b = pair["sample_b"]
            image_name = sample_a["file_name"]
            
            with zipfile.ZipFile(zip_path, "r") as archive:
                matching = [name for name in archive.namelist() if name.endswith(image_name)]
                img_data = archive.read(matching[0])
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
                temp_path = f"/tmp/{image_name}"
                image.save(temp_path)
                
            query = sample_a["expression_vi"] if query_selection == "Query A" else sample_b["expression_vi"]
        else:
            if custom_img is None:
                return None, pd.DataFrame([{"Error": "Please upload an image."}])
            image = custom_img.convert("RGB")
            temp_path = "/tmp/custom_uploaded.jpg"
            image.save(temp_path)
            query = custom_query
            
        if not query:
            return None, pd.DataFrame([{"Error": "Please specify a query."}])
            
        # 2. Prepare resized image for LocateAnything
        image_infer = image.copy()
        max_side = 640
        w, h = image_infer.size
        if max(w, h) > max_side:
            if w > h:
                new_w = max_side
                new_h = int(round(h * (max_side / w)))
            else:
                new_h = max_side
                new_w = int(round(w * (max_side / h)))
            image_infer = image_infer.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        # 3. Model predictions
        results = {}
        
        # LocateAnything base, random_ft, hardpair_ft
        for model_key in ["base", "random_ft", "hardpair_ft"]:
            tokenizer, processor, model = models_dict[model_key]
            answer = predict_one(tokenizer, processor, model, image_infer, query)
            box = parse_box(answer, image.width, image.height)
            results[model_key] = {"answer": answer, "box": box}
            
        # Grounding DINO
        gd_processor, gd_model = models_dict["grounding_dino"]
        text = query.strip().lower()
        if not text.endswith("."):
            text += "."
        inputs = gd_processor(images=image, text=text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = gd_model(**inputs)
        res = gd_processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=0.2,
            text_threshold=0.15,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = res.get("boxes", [])
        scores = res.get("scores", [])
        if len(boxes) > 0:
            best_idx = int(torch.argmax(scores).item()) if torch.is_tensor(scores) else 0
            box = [int(round(v)) for v in boxes[best_idx].tolist()]
            norm = normalized_from_pixel(box, image.width, image.height)
            raw = f"<box><{norm[0]}><{norm[1]}><{norm[2]}><{norm[3]}></box>"
        else:
            raw, box = "", None
        results["grounding_dino"] = {"answer": raw, "box": box}
        
        # Qwen2.5-VL-3B
        qwen_processor, qwen_model = models_dict["qwen2_5_vl_3b"]
        prompt = (
            "Locate exactly one object matching this referring expression:\n"
            f"{query}\n\n"
            "Return only one bounding box and no other text. "
            "Prefer this exact format: <box><x1><y1><x2><y2></box>. "
            "If you cannot use that format, return valid JSON: {\"bbox_2d\":[x1,y1,x2,y2]}. "
            "Coordinates may be image pixels or normalized 0-1000."
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": Path(temp_path).resolve().as_uri()},
                {"type": "text", "text": prompt},
            ],
        }]
        text_qwen = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs_qwen = qwen_processor(
            text=[text_qwen],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            generated_ids = qwen_model.generate(**inputs_qwen, max_new_tokens=64, do_sample=False)
        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_qwen.input_ids, generated_ids)]
        answer = qwen_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        box = parse_qwen_box(answer, image.width, image.height)
        results["qwen2_5_vl_3b"] = {"answer": answer, "box": box}
        
        # 4. Plot results
        colors = {
            "base": "red",
            "random_ft": "blue",
            "hardpair_ft": "lime",
            "grounding_dino": "orange",
            "qwen2_5_vl_3b": "magenta",
        }
        
        img = image.convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        for model_key, result in results.items():
            box = result["box"]
            if box:
                draw.rectangle(box, outline=colors.get(model_key, "white"), width=5)
                
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Query: {query}", fontsize=12, pad=10)
        
        legend_patches = []
        for model_key, color in colors.items():
            if model_key in results and results[model_key]["box"] is not None:
                legend_patches.append(mpatches.Patch(color=color, label=model_key))
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        output_pil_img = Image.open(buf)
        plt.close(fig)
        
        # 5. Output DataFrame
        rows = []
        for k, v in results.items():
            rows.append({
                "Model": k,
                "Raw Output": v["answer"],
                "Parsed Box (xyxy)": str(v["box"])
            })
        df = pd.DataFrame(rows)
        return output_pil_img, df

    # Build Gradio UI
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray")) as demo:
        gr.Markdown("# 🚀 ViGround-Contrast Live Demo")
        gr.Markdown("Compare Referring Expression Grounding models live on an A100 GPU (Modal.com serverless backend).")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_mode = gr.Dropdown(
                    choices=["Preset Hard Pair", "Custom Upload"],
                    value="Preset Hard Pair",
                    label="Input Mode"
                )
                preset_selector = gr.Dropdown(
                    choices=preset_choices,
                    value=preset_choices[213],
                    label="Select Preset Hard Pair"
                )
                
                # Active queries display
                preset_info = gr.Markdown(
                    "**[Query A]:** 'Người đàn ông bên trái trong bộ đồ đen'  \n**[Query B]:** 'người đàn ông trước bộ đồ xám trái'"
                )
                
                query_selection = gr.Radio(
                    choices=["Query A", "Query B"],
                    value="Query B",
                    label="Select Target Query"
                )
                
                custom_query_input = gr.Textbox(
                    value="",
                    label="Custom Query",
                    placeholder="Enter referring expression (e.g. người mặc áo đen...)",
                    visible=False
                )
                custom_image_input = gr.Image(
                    type="pil",
                    label="Upload Custom Image",
                    visible=False
                )
                run_btn = gr.Button("Run Inference", variant="primary")
                
            with gr.Column(scale=2):
                output_img = gr.Image(label="Overlay Visualization")
                output_table = gr.Dataframe(label="Models Output Metadata")
                
        # Interactive UI controls
        def handle_mode_change(mode):
            if mode == "Preset Hard Pair":
                return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
                
        input_mode.change(
            handle_mode_change,
            inputs=[input_mode],
            outputs=[preset_selector, query_selection, custom_query_input, custom_image_input]
        )
        
        def update_preset_info(preset_str):
            idx = int(preset_str.split("]")[0].replace("[", "").strip())
            pair = hard_pairs[idx]
            qa = pair["sample_a"]["expression_vi"]
            qb = pair["sample_b"]["expression_vi"]
            return f"**[Query A]:** '{qa}'  \n**[Query B]:** '{qb}'"
            
        preset_selector.change(
            update_preset_info,
            inputs=[preset_selector],
            outputs=[preset_info]
        )
        
        run_btn.click(
            run_gradio_inference,
            inputs=[input_mode, preset_selector, query_selection, custom_query_input, custom_image_input],
            outputs=[output_img, output_table]
        )
        
    web_app = FastAPI()
    return gr.mount_gradio_app(web_app, demo, path="/")
