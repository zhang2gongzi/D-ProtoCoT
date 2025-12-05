# evaluate_dprotocot.py

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import re

# 替换原来的 SimpleMLP 类
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=256):
        super().__init__()
        self.layers = nn.Sequential(  # ⭐ 必须叫 "layers"，和训练时一致！
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.layers(x)
        
def clean_answer(ans):
    if ans is None:
        return ""
    return re.sub(r"\D", "", str(ans))

def is_correct(pred, gt):
    return clean_answer(pred) == clean_answer(gt)

def main():
    # Paths
    cot_file = "/root/autodl-tmp/ProtoCoT/preparatory-work/gsm8k_test_500_cot_paths.json"
    mlp_path = "/root/autodl-tmp/ProtoCoT/preparatory-work/mlp_projection_head_1205.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load BERT
    print("Loading BERT...")
    bert_tokenizer = BertTokenizer.from_pretrained("/root/autodl-tmp/models/bert-base-uncased")
    bert_model = BertModel.from_pretrained("/root/autodl-tmp/models/bert-base-uncased").eval().to(device)

    # Load MLP
    print("Loading MLP...")
    mlp = SimpleMLP().to(device)
    mlp.load_state_dict(torch.load(mlp_path, map_location=device))
    mlp.eval()

    # Load test CoT
    with open(cot_file, "r") as f:
        data = json.load(f)

    correct = 0
    total = 0

    for item in tqdm(data, desc="Evaluating D-ProtoCoT"):
        question = item["question"]
        gt = item["ground_truth"]
        paths = item["paths"]

        if not paths or all(p["reasoning"] == "" for p in paths):
            continue

        # Encode all paths
        reps = []
        pred_answers = []
        for path in paths:
            reasoning = path["reasoning"]
            pred_ans = path["predicted_answer"]
            pred_answers.append(pred_ans)

            # BERT encode
            text = f"{question} [SEP] {reasoning}"
            inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = bert_model(**inputs)
                seq_rep = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # (768,)
                proj_rep = mlp(seq_rep.unsqueeze(0)).squeeze(0)  # (256,)
            reps.append(proj_rep)

        reps = torch.stack(reps)  # (K, 256)

        # Construct prototype: mean of all path representations
        prototype = reps.mean(dim=0)  # (256,)

        # Compute cosine similarity
        sims = F.cosine_similarity(reps, prototype.unsqueeze(0), dim=1)  # (K,)
        best_idx = torch.argmax(sims).item()

        selected_ans = pred_answers[best_idx]
        if is_correct(selected_ans, gt):
            correct += 1
        total += 1

    accuracy = correct / total * 100
    print(f"\n✅ D-ProtoCoT Accuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()