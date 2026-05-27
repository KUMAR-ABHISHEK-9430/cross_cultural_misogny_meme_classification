"""
generate_labels_hf.py
---------------------
Ensemble label generation for English-only meme datasets using Hugging Face VLMs.

Three Open-Weight VLMs vote independently on each (image, culture) pair.
Final label = majority vote (2/3 or 3/3 agree).

VLMs used (loaded in 4-bit to save VRAM):
  1. LLaVA 1.5 7B       (llava-hf/llava-1.5-7b-hf)
  2. Phi-3 Vision       (microsoft/Phi-3-vision-128k-instruct)
  3. Qwen-VL Chat       (Qwen/Qwen-VL-Chat)

Usage:
    pip install transformers torch accelerate bitsandbytes pillow pandas
    
    python generate_labels_hf.py \
        --input  data/train.csv \
        --images data/images/ \
        --output data/train_labelled.csv
"""

import argparse
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CULTURES = ["indian", "irish", "chinese"]

# ── Culture-specific prompts ───────────────────────────────────────────────────

CULTURE_PROMPTS = {
    "indian": (
        "You are an expert content moderator with deep knowledge of Indian culture, "
        "Bollywood, regional traditions, and contemporary Indian social norms. "
        "Examine this meme carefully. "
        "The text on the meme is: \"{transcript}\". "
        "From the perspective of Indian cultural values, "
        "does this meme demean, stereotype, or portray women negatively? "
        "Reply with EXACTLY one word: misogyny OR not-misogyny. Do not explain."
    ),
    "irish": (
        "You are an expert content moderator with deep knowledge of Irish and Western European culture, "
        "humour styles, and contemporary social norms. "
        "Examine this meme carefully. "
        "The text on the meme is: \"{transcript}\". "
        "From the perspective of Irish/Western cultural values, "
        "does this meme demean, stereotype, or portray women negatively? "
        "Reply with EXACTLY one word: misogyny OR not-misogyny. Do not explain."
    ),
    "chinese": (
        "You are an expert content moderator with deep knowledge of Chinese culture, "
        "Confucian social values, internet meme culture, and contemporary Chinese norms. "
        "Examine this meme carefully. "
        "The text on the meme is: \"{transcript}\". "
        "From the perspective of Chinese cultural values, "
        "does this meme demean, stereotype, or portray women negatively? "
        "Reply with EXACTLY one word: misogyny OR not-misogyny. Do not explain."
    ),
}

# ── Per-model callers ──────────────────────────────────────────────────────────

def _parse_response(raw: str, model_name: str, culture: str) -> str:
    """Normalise any model response to 'misogyny' or 'not-misogyny'."""
    text = raw.strip().lower()
    if "not-misogyny" in text or "not misogyny" in text or "not" in text:
        return "not-misogyny"
    if "misogyny" in text or "misogynous" in text or "misogynistic" in text:
        return "misogyny"
    
    logger.warning(
        f"[{model_name}] Unexpected response for culture={culture}: {raw!r} "
        f"— treating as not-misogyny"
    )
    return "not-misogyny"


def call_llava(model_dict: dict, image: Image.Image, transcript: str, culture: str) -> str:
    """Call LLaVA 1.5"""
    model = model_dict["model"]
    processor = model_dict["processor"]
    
    prompt_text = CULTURE_PROMPTS[culture].format(transcript=transcript)
    # LLaVA specific prompt format
    prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
    
    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10)
    
    # Decode only the newly generated tokens
    generated_text = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _parse_response(generated_text, "llava", culture)


def call_phi3v(model_dict: dict, image: Image.Image, transcript: str, culture: str) -> str:
    """Call Phi-3-Vision"""
    model = model_dict["model"]
    processor = model_dict["processor"]
    
    prompt_text = CULTURE_PROMPTS[culture].format(transcript=transcript)
    messages = [{"role": "user", "content": f"<|image_1|>\n{prompt_text}"}]
    
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10, temperature=0.0)
        
    generated_text = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _parse_response(generated_text, "phi3v", culture)


def call_qwen(model_dict: dict, image: Image.Image, transcript: str, culture: str) -> str:
    """Call Qwen-VL-Chat"""
    model = model_dict["model"]
    processor = model_dict["processor"]
    
    prompt_text = CULTURE_PROMPTS[culture].format(transcript=transcript)
    
    # Qwen-VL has its own unique chat template handling
    query = processor.from_list_format([
        {'image': image},
        {'text': prompt_text},
    ])
    
    inputs = processor(query, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10)
        
    generated_text = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _parse_response(generated_text, "qwenvl", culture)


# ── Ensemble voting ────────────────────────────────────────────────────────────

def majority_vote(votes: List[str]) -> Tuple[str, int]:
    counts = Counter(votes)
    winner, count = counts.most_common(1)[0]
    return winner, count


# ── Main ───────────────────────────────────────────────────────────────────────

def find_image(image_id: str, img_dir: Path) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = img_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def load_models(models_to_load: List[str]):
    """Load requested HF models into memory using 4-bit quantization."""
    clients = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 4-bit quantization config to fit 3 models on one GPU
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    if "llava" in models_to_load:
        logger.info("Loading LLaVA 1.5...")
        model_id = "llava-hf/llava-1.5-7b-hf"
        clients["llava"] = {
            "processor": AutoProcessor.from_pretrained(model_id),
            "model": AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, device_map="auto")
        }

    if "phi3v" in models_to_load:
        logger.info("Loading Phi-3-Vision...")
        model_id = "microsoft/Phi-3-vision-128k-instruct"
        clients["phi3v"] = {
            "processor": AutoProcessor.from_pretrained(model_id, trust_remote_code=True),
            "model": AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, quantization_config=quant_config, device_map="auto")
        }

    if "qwenvl" in models_to_load:
        logger.info("Loading Qwen-VL-Chat...")
        model_id = "Qwen/Qwen-VL-Chat"
        clients["qwenvl"] = {
            "processor": AutoProcessor.from_pretrained(model_id, trust_remote_code=True),
            "model": AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, quantization_config=quant_config, device_map="auto")
        }

    if len(clients) < 2:
        raise RuntimeError("Need at least 2 models for ensemble voting.")

    return clients


def label_one_meme(clients: dict, image: Image.Image, transcript: str, culture: str) -> Dict:
    per_model = {}
    callers = {
        "llava": call_llava,
        "phi3v": call_phi3v,
        "qwenvl": call_qwen,
    }

    for model_name, client in clients.items():
        try:
            label = callers[model_name](client, image, transcript, culture)
            per_model[model_name] = label
        except Exception as e:
            logger.warning(f"[{model_name}] failed for culture={culture}: {e}")

    if not per_model:
        return {"votes": [], "final_label": "not-misogyny", "agreement": 0, "per_model": {}}

    votes = list(per_model.values())
    final_label, count = majority_vote(votes)

    return {
        "votes": votes,
        "final_label": final_label,
        "agreement": count,
        "per_model": per_model,
    }


def main():
    parser = argparse.ArgumentParser(description="Ensemble VLM label generation using Hugging Face.")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--images", required=True, help="Directory containing meme images")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--models", nargs="+", default=["llava", "phi3v", "qwenvl"], choices=["llava", "phi3v", "qwenvl"])
    parser.add_argument("--limit", type=int, default=None, help="Process only N rows")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-votes", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)

    id_col = next((c for c in df.columns if "image_id" in c.lower()), df.columns[0])
    tx_col = next((c for c in df.columns if "transcript" in c.lower()), None)

    # ── Load Hugging Face Models ───────────────────────────────────────────────
    clients = load_models(args.models)
    logger.info(f"Ensemble loaded: {list(clients.keys())}")

    img_dir = Path(args.images)
    results = []

    for row_idx, row in df.iterrows():
        image_id = str(row[id_col])
        transcript = str(row[tx_col]) if tx_col and pd.notna(row.get(tx_col)) else ""

        img_path = find_image(image_id, img_dir)
        if img_path is None:
            continue

        try:
            # Resize image to prevent massive RAM usage during tensor creation
            image = Image.open(img_path).convert("RGB")
            image.thumbnail((512, 512)) 
        except Exception as e:
            logger.error(f"Failed to open {image_id}: {e}")
            continue

        row_result = {id_col: image_id, tx_col or "transcriptions": transcript}
        culture_summaries = {}

        for culture in CULTURES:
            result = label_one_meme(clients, image, transcript, culture)
            row_result[f"{culture}_label"] = result["final_label"]

            if args.save_votes:
                for model_name, vote in result["per_model"].items():
                    row_result[f"{culture}_{model_name}_vote"] = vote
                row_result[f"{culture}_agreement"] = result["agreement"]

            culture_summaries[culture] = f"{result['final_label']} ({result['agreement']}/{len(result['votes'])})"

        results.append(row_result)

        if args.verbose or row_idx % 10 == 0:
            summary = " | ".join(f"{c}: {v}" for c, v in culture_summaries.items())
            logger.info(f"[{row_idx+1}/{len(df)}] {image_id} — {summary}")

        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(str(args.output).replace(".csv", ".partial.csv"), index=False)

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    logger.info("Done.")

if __name__ == "__main__":
    main()