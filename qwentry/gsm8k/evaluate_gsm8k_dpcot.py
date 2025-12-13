# evaluate_robust.py
import json
import torch
from transformers import AutoTokenizer, AutoModel
from collections import Counter

JSON_PATH = "/root/autodl-tmp/ProtoCoT/qwentry/gsm8k/gsm8k_500_cot_qwen3.json"
MODEL_NAME = "/root/autodl-tmp/models/bert-base-uncased"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading BERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

def encode_texts(texts):
    if not texts:
        return None
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert(**inputs)
        last_hidden = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
        sum_emb = torch.sum(last_hidden * mask, dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        return sum_emb / sum_mask

def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None

# Load data
with open(JSON_PATH, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples.")

correct_sc = correct_dp = oracle_correct = 0
total = len(data)

for idx, item in enumerate(data):
    gt = safe_float(item.get("ground_truth"))
    if gt is None:
        print(f"⚠️ Sample {idx}: invalid ground_truth, skipping.")
        total -= 1
        continue

    paths = item.get("paths", [])
    valid_preds = []
    cot_texts = []
    is_correct_flags = []

    for p in paths:
        ans = safe_float(p.get("pred_answer"))
        text = p.get("text", "")
        if ans is not None and isinstance(text, str) and text.strip():
            valid_preds.append(ans)
            cot_texts.append(text)
            is_correct_flags.append(bool(p.get("is_correct", False)))

    if not valid_preds:
        print(f"⚠️ Sample {idx}: no valid predictions, skipping.")
        total -= 1
        continue

    # Oracle
    if any(is_correct_flags):
        oracle_correct += 1

    # Self-Consistency: vote on rounded string to avoid float key issues
    str_preds = [str(round(a, 5)) for a in valid_preds]  # normalize
    vote = Counter(str_preds).most_common(1)[0][0]
    sc_pred = float(vote)
    if abs(sc_pred - gt) < 1e-3:
        correct_sc += 1

    # D-ProtoCoT (current version: mean of all paths)
    try:
        emb = encode_texts(cot_texts)
        proto = emb.mean(dim=0, keepdim=True)
        sims = torch.nn.functional.cosine_similarity(emb, proto)
        best = valid_preds[sims.argmax().item()]
        if abs(best - gt) < 1e-3:
            correct_dp += 1
    except Exception as e:
        # fallback to first
        if abs(valid_preds[0] - gt) < 1e-3:
            correct_dp += 1

# Results
print("\n" + "="*50)
print(f"Evaluation on {total} samples:")
print(f"Self-Consistency:     {correct_sc / total * 100:.2f}%")
print(f"D-ProtoCoT:           {correct_dp / total * 100:.2f}%")
print(f"Oracle (≥1 correct):   {oracle_correct / total * 100:.2f}%")
print("="*50)