# evaluate_protocot.py (with local BERT, final version)
import json
import argparse
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

def mean_pooling(model_output, attention_mask):
    """Apply mean pooling to get sentence embeddings."""
    token_embeddings = model_output[0]  # [batch, seq_len, hidden_size]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_bert_embeddings(texts, tokenizer, model, device):
    """Encode a list of texts into normalized BERT embeddings."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)

    embeddings = mean_pooling(outputs, encoded['attention_mask'])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # L2 normalize
    return embeddings.cpu().numpy()

def select_representative_path_bert(cot_texts, pred_answers, tokenizer, model, device):
    """Select the answer from the CoT path closest to the prototype (mean embedding)."""
    if len(cot_texts) == 0:
        return None

    embeddings = get_bert_embeddings(cot_texts, tokenizer, model, device)
    # Fix: use keepdims=True so prototype has shape (1, D)
    prototype = np.mean(embeddings, axis=0, keepdims=True)
    similarities = cosine_similarity(prototype, embeddings)[0]  # shape: (N,)
    best_idx = int(np.argmax(similarities))
    return pred_answers[best_idx]

def evaluate_protocot_with_bert(json_path, bert_model_path="/root/autodl-tmp/models/bert-base-uncased"):
    print(f"Loading local BERT model from: {bert_model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model from local path
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    model = BertModel.from_pretrained(bert_model_path).to(device)
    model.eval()

    # Load data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_samples = len(data)
    correct_count = 0

    for i, item in enumerate(data):
        ground_truth = item["ground_truth"]
        paths = item["paths"]

        cot_texts = []
        pred_answers = []

        for p in paths:
            text = p.get("text", "").strip()
            pred = p.get("pred_answer")
            if text and pred is not None:
                cot_texts.append(text)
                pred_answers.append(pred)

        # Skip if no valid paths
        if not cot_texts:
            continue

        # Safety check (should always hold)
        assert len(cot_texts) == len(pred_answers), f"Mismatch at sample {i}"

        selected_pred = select_representative_path_bert(cot_texts, pred_answers, tokenizer, model, device)
        if selected_pred == ground_truth:
            correct_count += 1

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{total_samples} samples...")

    protocot_acc = correct_count / total_samples
    print("\n" + "=" * 60)
    print(f"📊 ProtoCoT Evaluation (using local BERT for similarity)")
    print(f"BERT model path: {bert_model_path}")
    print(f"Total samples: {total_samples}")
    print(f"Correct predictions: {correct_count}")
    print("-" * 60)
    print(f"✅ ProtoCoT Accuracy: {protocot_acc:.4f} ({correct_count}/{total_samples})")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ProtoCoT using local BERT embeddings.")
    parser.add_argument("--input", type=str, required=True, help="Path to the generated JSON file")
    args = parser.parse_args()

    evaluate_protocot_with_bert(args.input)