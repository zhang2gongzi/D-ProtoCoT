import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3-8B"
DATASET_PATH = "/root/autodl-tmp/ProtoCoT/database/gsm8k/train-00000-of-00001.parquet"
MAX_SAMPLES = 500
MAX_NEW_TOKENS = 256

# -----------------------------
# 1. Load model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side="left"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# -----------------------------
# 2. Fixed contrastive demos (paper-style)
# -----------------------------
CONTRASTIVE_DEMOS = """
Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?

Explanation:
He writes each friend 3 × 2 = 6 pages per week.
Since he writes to 2 friends, he writes 6 × 2 = 12 pages per week.
There are 52 weeks in a year, so he writes 12 × 52 = 624 pages in total.

Wrong Explanation:
He writes 12 × 52 = 624 pages per week.
So he writes 3 × 2 = 6 pages every week.
That means he writes 6 × 2 = 12 pages a year.

Answer: 624
"""

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves math word problems. "
    "You will be shown examples with both correct and incorrect reasoning. "
    "Learn from the contrast between them. "
    "For the final question, provide a step-by-step explanation and the final answer."
)

# -----------------------------
# 3. Helper functions (MODIFIED AS PER YOUR INSTRUCTIONS)
# -----------------------------

def extract_gold_answer(answer_text):
    match = re.search(r"####\s*(-?\d+)", answer_text)
    return match.group(1) if match else None

def build_prompt(question):
    return f"""{CONTRASTIVE_DEMOS}

Question: {question}

Explanation:"""

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
    # Decode and remove prompt
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = text[len(prompt):]
    # Prevent next-question leakage
    response = response.split("\nQuestion:")[0]
    return response.strip()

def extract_pred_answer(text):
    match = re.search(r"Answer:\s*\$?\s*(-?\d+)", text)
    if match:
        return match.group(1)
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else None

# -----------------------------
# 4. Evaluation loop (MODIFIED GOLD/PRED EXTRACTION)
# -----------------------------
df = pd.read_parquet(DATASET_PATH)
df = df.head(MAX_SAMPLES)

correct = 0
total = 0

results = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    question = row["question"]
    gold = extract_gold_answer(row["answer"])  # ✅ FIXED

    if gold is None:
        continue  # skip malformed

    prompt = build_prompt(question)
    response = generate_answer(prompt)
    pred = extract_pred_answer(response)  # ✅ FIXED

    is_correct = (pred == gold)
    correct += int(is_correct)
    total += 1

    results.append({
        "question": question,
        "gold": gold,
        "prediction": pred,
        "correct": is_correct,
        "output": response
    })

accuracy = correct / total * 100
print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")

with open("/root/autodl-tmp/ProtoCoT/baseline/ccot/qwen/contrastive_cot_results/gsm8k_contrastive_cot_qwentryagain.json", "w") as f:
    json.dump(results, f, indent=2)