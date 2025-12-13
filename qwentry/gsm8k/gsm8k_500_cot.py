# generate_gsm8k_cot_qwen3.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import os

# === 配置 ===
MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3-8B"
DATA_PATH = "/root/autodl-tmp/ProtoCoT/database/gsm8k/train-00000-of-00001.parquet"
OUTPUT_PATH = "/root/autodl-tmp/ProtoCoT/qwentry/gsm8k/gsm8k_500_cot_qwen3.json"
NUM_SAMPLES = 500
NUM_PATHS = 10
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7  # for diversity

# === 加载模型 ===
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,  # or torch.float16 if bfloat16 not supported
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# === Prompt Template ===
COT_PROMPT_TEMPLATE = """Please solve the following math word problem strictly by following these steps:
1. Analyze the given conditions and the question to be answered.
2. List the formulas or reasoning steps needed.
3. Perform step-by-step calculations, showing each intermediate result.
4. Final Answer: Only output the numerical answer (no units, no extra text).

Problem: {question}

Solution:
"""

def extract_answer(text):
    """从 CoT 末尾提取最终数字答案"""
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("Final Answer:") or "Final Answer" in line:
            # Extract number after colon or from end
            parts = line.split(":")
            if len(parts) > 1:
                candidate = parts[-1].strip()
            else:
                candidate = line
            # Keep only digits, dot, negative sign
            num_str = ""
            for c in candidate:
                if c.isdigit() or c in ".-":
                    num_str += c
                elif num_str:
                    break
            if num_str.replace(".", "").replace("-", "").isdigit():
                try:
                    return float(num_str) if "." in num_str else int(num_str)
                except:
                    pass
        # Fallback: last number in last few lines
    for line in reversed(lines[-3:]):
        words = line.split()
        for word in reversed(words):
            clean = word.strip(".,;:!?()")
            if clean.replace(".", "").replace("-", "").isdigit():
                try:
                    return float(clean) if "." in clean else int(clean)
                except:
                    continue
    return None

# === 加载数据 ===
print("Loading GSM8K data...")
df = pd.read_parquet(DATA_PATH)
samples = df.head(NUM_SAMPLES).to_dict(orient="records")

results = []

for i, sample in enumerate(tqdm(samples, desc="Generating CoT paths")):
    question = sample["question"]
    ground_truth = sample["answer"]  # string like "#### 42"

    # Parse ground truth
    gt_num = None
    if "####" in ground_truth:
        try:
            gt_num = float(ground_truth.split("####")[-1].strip())
        except:
            pass

    paths = []
    prompt = COT_PROMPT_TEMPLATE.format(question=question)

    for _ in range(NUM_PATHS):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from output
        if full_output.startswith(prompt):
            cot_text = full_output[len(prompt):].strip()
        else:
            cot_text = full_output.strip()

        pred_answer = extract_answer(cot_text)
        is_correct = False
        if pred_answer is not None and gt_num is not None:
            # Handle float/int comparison
            try:
                is_correct = abs(float(pred_answer) - float(gt_num)) < 1e-5
            except:
                is_correct = str(pred_answer).strip() == str(gt_num).strip()

        paths.append({
            "text": cot_text,
            "pred_answer": pred_answer,
            "is_correct": int(is_correct)
        })

    results.append({
        "qid": f"gsm8k_{i}",
        "question": question,
        "ground_truth": gt_num,
        "paths": paths
    })

# === 保存结果 ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(results)} samples with {NUM_PATHS} paths each to {OUTPUT_PATH}")