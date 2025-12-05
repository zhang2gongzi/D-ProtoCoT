# train_mlp_contrastive.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=256):
        super(SimpleMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

def compute_prototype(positive_reps):
    # positive_reps: list of tensors [N, D]
    stacked = torch.stack(positive_reps)  # (N, D)
    return torch.mean(stacked, dim=0)     # (D,)

def info_nce_loss(z_i, z_j, temperature=0.07):
    """
    简化版：z_i 是当前样本，z_j 是 prototype（视为正样本）
    负样本：同 batch 中其他路径的表示（这里我们用所有路径作为负样本池）
    但为简化，我们采用：对每个样本，计算与 prototype 的相似度，并用 InfoNCE 风格损失
    """
    # 实际上，在 D-ProtoCoT 中，loss 是：
    # L = -log exp(sim(z_i, p)/τ) / Σ_j exp(sim(z_j, p)/τ)
    # 即：以 prototype 为锚点，最大化正样本（所有路径）与 p 的对齐
    # 但我们这里简化：只让 MLP 学会将表示映射到能与 prototype 对齐的空间
    pass  # 我们换一种更直接的方式

def main():
    input_file = "/root/autodl-tmp/ProtoCoT/preparatory-work/gsm8k_train_500_encoded.json"
    output_model_path = "/root/autodl-tmp/ProtoCoT/preparatory-work/mlp_projection_head.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(input_file, "r") as f:
        data = json.load(f)

    # 加载所有路径表示
    all_reps = []
    all_labels = []
    for item in data:
        for path in item["paths"]:
            rep = torch.tensor(path["encoded_seq"], dtype=torch.float32).to(device)
            label = path["is_correct"]
            all_reps.append(rep)
            all_labels.append(label)
    
    print(f"Total paths: {len(all_reps)}")

    mlp = SimpleMLP().to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-4)
    num_epochs = 20

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for item in data:
            labels = [path["is_correct"] for path in item["paths"]]
            reps = [torch.tensor(path["encoded_seq"], dtype=torch.float32).to(device) for path in item["paths"]]

            if not any(labels):
                continue  # 跳过无正样本的问题

            # 获取正样本表示（用于构建 prototype）
            positive_reps = [rep for rep, lab in zip(reps, labels) if lab]
            prototype = compute_prototype(positive_reps)  # (D,)
            prototype_proj = mlp(prototype.unsqueeze(0)).squeeze(0)  # (D_out,)

            # 所有路径经过 MLP
            proj_reps = [mlp(rep.unsqueeze(0)).squeeze(0) for rep in reps]  # list of (D_out,)

            # 计算每个路径与 prototype 的相似度（cosine）
            sims = []
            for z in proj_reps:
                sim = F.cosine_similarity(z.unsqueeze(0), prototype_proj.unsqueeze(0))
                sims.append(sim)

            sims = torch.stack(sims)  # (K,)

            # 构造 InfoNCE 风格损失：希望正样本相似度高，但这里我们不知道哪条是正（训练时知道！）
            # 实际上，我们可以直接最大化所有路径与 prototype 的平均相似度（因为 prototype 来自正样本）
            # 或者：只优化正样本的相似度（更合理）
            positive_mask = torch.tensor(labels, dtype=torch.float32).to(device)  # (K,)
            if positive_mask.sum() == 0:
                continue

            # 只计算正样本的 loss：希望它们与 prototype 相似度接近 1
            positive_sims = sims[positive_mask.bool()]
            loss = (1 - positive_sims).mean()  # 最大化 cosine similarity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    # 保存模型
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(mlp.state_dict(), output_model_path)
    print(f"✅ MLP trained and saved to {output_model_path}")

if __name__ == "__main__":
    main()