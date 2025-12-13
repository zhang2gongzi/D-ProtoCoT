# evaluate_dpcot_inference_only.py (NO sklearn)
import json
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# === 1. 加载模型 ===
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(device).eval()

# === 2. Cosine Similarity (PyTorch) ===
def cosine_similarity_torch(x, y):
    """
    x: [N, D]
    y: [M, D]  (here M=1 for prototype)
    returns: [N, M]
    """
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=1)
    return torch.mm(x_norm, y_norm.transpose(0, 1))  # [N, M]

# === 3. 加载数据 ===
with open("/root/autodl-tmp/ProtoCoT/qwentry/strategyqa/strategyqa_500_cot_template.json", "r") as f:
    data = json.load(f)

def encode_text(text_list):
    inputs = tokenizer(
        text_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = bert(**inputs)
        embeddings = outputs.last_hidden_state  # [B, L, D]
        attention_mask = inputs['attention_mask']
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        return mean_pooled  # [B, D]

# === 4. 评估 ===
correct_dpcot = 0
correct_majority = 0
total = len(data)

for idx, item in enumerate(data):
    question = item["question"]
    desc = item["description"]
    ground_truth = item["ground_truth"]
    paths = item["paths"]

    # 构造输入文本
    texts = []
    pred_answers = []
    for p in paths:
        text = p["text"]
        # 清理重复的“最终答案”部分（保留主推理）
        if "最终答案" in text:
            text = text.split("最终答案")[0].strip()
        full_text = f"{question} {desc} {text}"
        texts.append(full_text)
        pred_answers.append(p["pred_answer"])

    try:
        embeddings = encode_text(texts)  # [N, 768], on GPU
    except Exception as e:
        print(f"Skipping qid={item.get('qid', idx)} due to encoding error: {e}")
        continue

    # === D-ProtoCoT: 动态原型 = 所有路径均值 ===
    prototype = torch.mean(embeddings, dim=0, keepdim=True)  # [1, 768]
    sims = cosine_similarity_torch(embeddings, prototype)      # [N, 1]
    best_idx = torch.argmax(sims).item()
    dpcot_pred = pred_answers[best_idx]

    if dpcot_pred == ground_truth:
        correct_dpcot += 1

    # === Self-Consistency: Majority Vote ===
    from collections import Counter
    vote = Counter(pred_answers).most_common(1)[0][0]
    if vote == ground_truth:
        correct_majority += 1

# === 5. 输出结果 ===
print(f"\n✅ Evaluation on {total} StrategyQA samples:")
print(f"Self-Consistency Acc:     {correct_majority / total * 100:.2f}%")
print(f"D-ProtoCoT (Inference) Acc: {correct_dpcot / total * 100:.2f}%")

# Oracle
oracle_correct = sum(any(p["is_correct"] for p in item["paths"]) for item in data)
print(f"Oracle Acc (≥1 correct path): {oracle_correct / total * 100:.2f}%")