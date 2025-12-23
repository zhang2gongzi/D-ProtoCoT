# evaluate_robust.py (for CommonsenseQA)
import json
import torch
from transformers import AutoTokenizer, AutoModel
from collections import Counter
import os

# === 配置 ===
JSON_PATH = "/root/autodl-tmp/ProtoCoT/qwentry/commonsenseqa/commonsenseqa_500_cot_qwen3.json"
MODEL_NAME = "/root/autodl-tmp/models/bert-base-uncased"  # 或改用 sentence-transformers/all-MiniLM-L6-v2 更好

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

def safe_choice(x):
    """确保 x 是 A-E 中的一个大写字母"""
    if x is None:
        return None
    s = str(x).strip().upper()
    if len(s) == 1 and s in "ABCDE":
        return s
    return None

# === 加载数据 ===
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples.")

correct_sc = correct_dp = oracle_correct = 0
total = len(data)

for idx, item in enumerate(data):
    gt = safe_choice(item.get("ground_truth"))
    if gt is None:
        print(f"⚠️ Sample {idx}: invalid ground_truth '{item.get('ground_truth')}', skipping.")
        total -= 1
        continue

    paths = item.get("paths", [])
    valid_preds = []
    cot_texts = []
    is_correct_flags = []

    for p in paths:
        ans = safe_choice(p.get("pred_answer"))
        text = p.get("text", "")
        if ans is not None and isinstance(text, str) and text.strip():
            valid_preds.append(ans)
            cot_texts.append(text)
            is_correct_flags.append(bool(p.get("is_correct", False)))

    if not valid_preds:
        print(f"⚠️ Sample {idx}: no valid predictions, skipping.")
        total -= 1
        continue

    # Oracle: 只要有一条路径正确
    if any(is_correct_flags):
        oracle_correct += 1

    # Self-Consistency: 多数投票（直接对字母投票）
    vote_result = Counter(valid_preds).most_common(1)[0][0]
    if vote_result == gt:
        correct_sc += 1

    # D-ProtoCoT: 基于 CoT 文本语义相似度选最佳路径
    try:
        emb = encode_texts(cot_texts)
        if emb is None:
            raise ValueError("Empty embedding")
        proto = emb.mean(dim=0, keepdim=True)  # prototype = mean of all path embeddings
        sims = torch.nn.functional.cosine_similarity(emb, proto)  # [num_paths]
        best_idx = sims.argmax().item()
        dp_pred = valid_preds[best_idx]
        if dp_pred == gt:
            correct_dp += 1
    except Exception as e:
        # fallback: use first valid prediction
        if valid_preds[0] == gt:
            correct_dp += 1

# === 输出结果 ===
print("\n" + "="*50)
print(f"Evaluation on {total} valid samples:")
print(f"Self-Consistency:     {correct_sc / total * 100:.2f}%")
print(f"D-ProtoCoT:           {correct_dp / total * 100:.2f}%")
print(f"Oracle (≥1 correct):   {oracle_correct / total * 100:.2f}%")
print("="*50)

# === 保存日志 ===
log_dir = os.path.dirname(JSON_PATH)
log_path = os.path.join(log_dir, "robust_evaluation.txt")
with open(log_path, "w") as f:
    f.write(f"Robust Evaluation Results\n")
    f.write(f"JSON: {JSON_PATH}\n")
    f.write(f"Total valid samples: {total}\n")
    f.write(f"Self-Consistency: {correct_sc / total * 100:.2f}%\n")
    f.write(f"D-ProtoCoT: {correct_dp / total * 100:.2f}%\n")
    f.write(f"Oracle: {oracle_correct / total * 100:.2f}%\n")

print(f"\n✅ Log saved to: {log_path}")