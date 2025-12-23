# generate_cot_commonsenseqa.py
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from datasets import load_dataset
from tqdm import tqdm
import re

# === 配置 ===
MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3-8B"
DATA_PATH = "/root/autodl-tmp/ProtoCoT/database/commonsenseQA/train-00000-of-00001.parquet"
OUTPUT_JSON = "/root/autodl-tmp/ProtoCoT/qwentry/commonsenseqa/commonsenseqa_500_cot_qwen3.json"
NUM_SAMPLES = 500
NUM_PATHS = 10
MAX_NEW_TOKENS = 512
SEED = 42

set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# === 加载模型 ===
print("Loading Qwen3-8B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    trust_remote_code=True,
    device_map="auto"
)
model.eval()

def build_prompt(question, choices):
    choice_str = "\n".join([f"{label}. {text}" for label, text in choices.items()])
    return (
        f"Question: {question}\n"
        f"Choices:\n{choice_str}\n\n"
        "Let's solve this step by step. At the end, state your final answer as 'Final Answer: X' where X is one of A, B, C, D, or E."
    )

def extract_answer(text):
    match = re.search(r'Final Answer:\s*([A-E])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines and len(lines[-1]) == 1 and lines[-1].upper() in "ABCDE":
        return lines[-1].upper()
    return None

# === 加载并处理数据 ===
print("Loading CommonsenseQA dataset...")
dataset = load_dataset('parquet', data_files=DATA_PATH, split='train')
subset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))

results = []

for idx, item in enumerate(tqdm(subset, desc="Generating CoT")):
    # ✅ 正确解析嵌套 choices 结构
    question = item['question']
    labels = item['choices']['label']
    texts = item['choices']['text']
    choices_dict = dict(zip(labels, texts))
    ground_truth = item['answerKey']

    prompt = build_prompt(question, choices_dict)
    paths = []

    for i in range(NUM_PATHS):
        torch.manual_seed(SEED + idx * NUM_PATHS + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + idx * NUM_PATHS + i)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        cot_text = full_text[len(prompt_text):].strip()

        pred_answer = extract_answer(cot_text)
        is_correct = (pred_answer == ground_truth) if pred_answer else False

        paths.append({
            "text": cot_text,
            "pred_answer": pred_answer,
            "is_correct": is_correct
        })

    results.append({
        "qid": item['id'],
        "question": question,
        "choices": choices_dict,
        "ground_truth": ground_truth,
        "paths": paths
    })

# === 保存结果 ===
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Done! Saved {len(results)} samples to {OUTPUT_JSON}")