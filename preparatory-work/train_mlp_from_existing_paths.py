# train_mlp_from_existing_paths.py (FIXED)
import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

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

def main():
    # === 训练阶段 ===
    with open("/root/autodl-tmp/ProtoCoT/preparatory-work/gsm8k_train_500_with_labels.json", "r") as f:
        data = json.load(f)
    
    tokenizer = BertTokenizer.from_pretrained("/root/autodl-tmp/models/bert-base-uncased")
    bert = BertModel.from_pretrained("/root/autodl-tmp/models/bert-base-uncased").to(device).eval()
    mlp = SimpleMLP().to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=1e-4)
    criterion = nn.CosineEmbeddingLoss(margin=0.3)

    print("🚀 Training MLP using existing CoT paths...")
    mlp.train()
    for epoch in range(3):
        total_loss = 0
        num_batches = 0
        for item in tqdm(data, desc=f"Epoch {epoch+1}"):
            question = item["question"]
            ground_truth = str(item["ground_truth"]).strip()
            paths = item["paths"]

            valid_paths = []
            labels = []
            for p in paths:
                ans = str(p.get("predicted_answer", "")).strip()
                reasoning = p.get("reasoning", "").strip()
                if not reasoning or not ans:
                    continue
                text = f"Q: {question} A: {reasoning}"
                valid_paths.append(text)
                labels.append(1 if ans == ground_truth else -1)

            if len(valid_paths) < 2:
                continue

            inputs = tokenizer(
                valid_paths,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = bert(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token

            proj = mlp(embeddings)

            loss = 0
            count = 0
            n = len(labels)
            for i in range(n):
                for j in range(n):
                    if i >= j:
                        continue
                    y = 1 if labels[i] == labels[j] else -1
                    if labels[i] == 1 or labels[j] == 1:
                        loss += criterion(
                            proj[i].unsqueeze(0),
                            proj[j].unsqueeze(0),
                            torch.tensor([y], device=device, dtype=torch.float)
                        )
                        count += 1
            if count > 0:
                loss = loss / count
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"✅ Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")

    # Save MLP
    torch.save(mlp.state_dict(), "mlp_projection_head.pth")
    print("🎉 MLP saved to mlp_projection_head.pth")

    # === 计算全局 prototype（必须在 main 内部！）===
    print("Computing global prototype from correct paths...")
    all_correct_reps = []
    mlp.eval()
    with torch.no_grad():
        for item in tqdm(data, desc="Building prototype"):
            question = item["question"]
            ground_truth = str(item["ground_truth"]).strip()
            for p in item["paths"]:
                ans = str(p.get("predicted_answer", "")).strip()
                reasoning = p.get("reasoning", "").strip()
                if ans == ground_truth and reasoning:
                    text = f"Q: {question} A: {reasoning}"
                    inputs = tokenizer(
                        text, return_tensors="pt", max_length=512, truncation=True, padding=True
                    ).to(device)
                    bert_emb = bert(**inputs).last_hidden_state[:, 0, :]  # [CLS]
                    proj = mlp(bert_emb)  # (1, 256)
                    all_correct_reps.append(proj)

    if all_correct_reps:
        all_correct_reps = torch.cat(all_correct_reps, dim=0)  # (N, 256)
        global_prototype = all_correct_reps.mean(dim=0).cpu()  # (256,)
        torch.save(global_prototype, "global_prototype.pt")
        print(f"✅ Global prototype saved. Shape: {global_prototype.shape}")
        print(f"✅ Total correct paths used: {len(all_correct_reps)}")
    else:
        raise ValueError("No correct paths found to build prototype!")

if __name__ == "__main__":
    main()