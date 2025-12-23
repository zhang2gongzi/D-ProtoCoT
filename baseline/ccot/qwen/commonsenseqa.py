import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3-8B"
DATASET_PATH = "/root/autodl-tmp/ProtoCoT/database/commonsenseQA/train-00000-of-00001.parquet"
MAX_SAMPLES = 500
MAX_NEW_TOKENS = 64

# -----------------------------
# Load model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side="left"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    dtype=torch.float16  # fixed deprecation
)
model.eval()

# -----------------------------
# Prompt demos
# -----------------------------
CONTRASTIVE_DEMOS = """
Question: What do people use to absorb extra ink from a fountain pen?
Choices:
(A) shirt pocket
(B) calligrapher's hand
(C) inkwell
(D) desk drawer
(E) blotter

Explanation:
A blotter is specifically designed to absorb excess ink.
Other options like shirt pocket or desk drawer are not used for this purpose.

Wrong Explanation:
People usually use their hand to wipe ink, so the answer is (B).
Or they keep extra ink in the inkwell, so (C) is correct.

Answer: E
"""

def format_choices(choices):
    # Exact structure confirmed by user debug print:
    # choices = {'label': ['A','B',...], 'text': ['...', ...]}
    labels = choices['label']
    texts = choices['text']
    return "\n".join(f"({label}) {text}" for label, text in zip(labels, texts))

def build_prompt(question, choices_text):
    return f"""{CONTRASTIVE_DEMOS}

Question: {question}
Choices:
{choices_text}

Explanation:"""

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text[len(prompt):]
    # Prevent next-question leakage
    response = response.split("\nQuestion:")[0]
    return response.strip()

def extract_pred_answer(text):
    # Match "Answer: A" or "Answer: (A)" etc.
    match = re.search(r"Answer:\s*\(?([A-E])\)?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: first standalone A-E
    match = re.search(r"\b([A-E])\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None

# -----------------------------
# Main evaluation
# -----------------------------
df = pd.read_parquet(DATASET_PATH)
df = df.head(MAX_SAMPLES)

correct = 0
total = 0
results = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    question = row["question"]
    gold = row["answerKey"].strip().upper()  # e.g., "A"

    # Directly use the known dict structure
    try:
        choices_text = format_choices(row["choices"])
    except Exception as e:
        print(f"Error formatting choices: {e}")
        continue

    prompt = build_prompt(question, choices_text)
    response = generate_answer(prompt)
    pred = extract_pred_answer(response)

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

accuracy = correct / total * 100 if total > 0 else 0
print(f"\nCommonsenseQA Accuracy: {accuracy:.2f}% ({correct}/{total})")

with open("commonsenseqa_ccot_qwen.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)