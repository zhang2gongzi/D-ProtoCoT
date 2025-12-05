# evaluate_dprotocot_v2.py
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import re
import os

# --- MLP 定义（必须和训练时一致）---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

def clean_answer(ans):
    if ans is None:
        return ""
    # 只保留数字（GSM8K 答案都是整数）
    return re.sub(r"\D", "", str(ans))

def is_correct(pred, gt):
    return clean_answer(pred) == clean_answer(gt)

def main():
    # Paths
    cot_file = "/root/autodl-tmp/ProtoCoT/preparatory-work/gsm8k_test_500_cot_paths.json"
    mlp_path = "/root/autodl-tmp/mlp_projection_head.pth"
    prototype_path = "/root/autodl-tmp/global_prototype.pt"
    bert_path = "/root/autodl-tmp/models/bert-base-uncased"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load BERT
    print("Loading BERT...")
    bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert_model = BertModel.from_pretrained(bert_path).eval().to(device)

    # Load MLP
    print("Loading MLP...")
    mlp = SimpleMLP().to(device)
    mlp.load_state_dict(torch.load(mlp_path, map_location=device))
    mlp.eval()

    # Load global prototype
    print("Loading global prototype...")
    global_prototype = torch.load(prototype_path, map_location=device)  # (256,)
    global_prototype = global_prototype.unsqueeze(0)  # (1, 256)

    # Load test data
    with open(cot_file, "r") as f:
        data = json.load(f)

    correct = 0
    total = 0

    for item in tqdm(data, desc="Evaluating D-ProtoCoT"):
        question = item["question"]
        gt = item["ground_truth"]
        paths = item["paths"]

        if not paths:
            continue

        reps = []
        pred_answers = []

        for path in paths:
            reasoning = path.get("reasoning", "").strip()
            pred_ans = path.get("predicted_answer", "")
            if not reasoning:
                continue

            pred_answers.append(pred_ans)

            # 构造输入文本（与训练时一致）
            text = f"Q: {question} A: {reasoning}"
            inputs = bert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = bert_model(**inputs)
                # 使用 [CLS] token（与训练时一致！）
                cls_rep = outputs.last_hidden_state[:, 0, :]  # (1, 768)
                proj_rep = mlp(cls_rep)  # (1, 256)
            reps.append(proj_rep)

        if not reps or not pred_answers:
            continue

        reps = torch.cat(reps, dim=0)  # (K, 256)

        # ✅ 关键：与全局 prototype 计算相似度
        sims = F.cosine_similarity(reps, global_prototype, dim=1)  # (K,)
        best_idx = torch.argmax(sims).item()

        selected_ans = pred_answers[best_idx]
        if is_correct(selected_ans, gt):
            correct += 1
        total += 1

    if total == 0:
        print("⚠️ No valid samples!")
        return

    accuracy = correct / total * 100
    print(f"\n✅ D-ProtoCoT Accuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()