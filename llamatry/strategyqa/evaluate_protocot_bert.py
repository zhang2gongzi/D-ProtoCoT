# evaluate_protocot_bert.py
import json
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import os

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # [batch, seq, hidden]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_texts(texts, tokenizer, model, device):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**encoded)
    
    embeddings = mean_pooling(outputs, encoded['attention_mask'])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to CoT JSON file")
    parser.add_argument("--bert_model", type=str, default="/root/autodl-tmp/models/bert-base-uncased", 
                        help="Path to local BERT model")
    args = parser.parse_args()

    # Load data
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load BERT
    print(f"Loading BERT model from: {args.bert_model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModel.from_pretrained(args.bert_model).to(device)
    model.eval()

    # Standard answer templates
    POS_TEMPLATE = "The answer is yes."
    NEG_TEMPLATE = "The answer is no."

    total = len(data)
    correct = 0

    for idx, item in enumerate(data):
        question = item["question"]
        ground_truth = item["ground_truth"]  # bool
        paths = item["paths"]

        cot_texts = [p["text"] for p in paths if p["text"].strip()]
        if not cot_texts:
            continue  # skip if no text

        # Encode all COT texts + two templates
        all_texts = cot_texts + [POS_TEMPLATE, NEG_TEMPLATE]
        embeddings = encode_texts(all_texts, tokenizer, model, device)

        cot_embs = embeddings[:-2]           # (N, 768)
        pos_emb = embeddings[-2].reshape(1, -1)  # (1, 768)
        neg_emb = embeddings[-1].reshape(1, -1)  # (1, 768)

        # Compute similarities
        sim_pos = cosine_similarity(cot_embs, pos_emb).flatten()  # (N,)
        sim_neg = cosine_similarity(cot_embs, neg_emb).flatten()  # (N,)

        # Predict: True if sim_pos > sim_neg
        preds = sim_pos > sim_neg  # bool array

        # Check if any prediction matches ground truth
        if ground_truth:
            is_correct = any(preds)
        else:
            is_correct = any(~preds)  # i.e., at least one predicts False

        if is_correct:
            correct += 1

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{total} samples...")

    accuracy = correct / total
    print("\n" + "="*70)
    print("📊 ProtoCoT Evaluation using BERT Semantic Similarity")
    print(f"Input file: {args.input}")
    print(f"BERT model: {args.bert_model}")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print("-" * 70)
    print(f"✅ ProtoCoT Accuracy (best-of-{len(paths)}): {accuracy:.4f} ({correct}/{total})")
    print("="*70)

if __name__ == "__main__":
    main()