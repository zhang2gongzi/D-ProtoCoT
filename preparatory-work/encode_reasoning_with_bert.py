# encode_reasoning_with_bert.py

from transformers import BertTokenizer, BertModel
import torch
import json
import tqdm

def encode_reasoning(bert_model, bert_tokenizer, question, reasoning):
    text = f"{question} [SEP] {reasoning}"
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = bert_model(**inputs, output_hidden_states=True)
        token_embeddings = outputs.last_hidden_state[0]  # (L, 768)

    # Sequence-level: mean of all tokens (excluding [CLS]/[SEP] if desired)
    seq_rep = token_embeddings.mean(dim=0)  # (768,)
    
    return {
        "token": token_embeddings,
        "seq": seq_rep
    }

def main():
    bert_tokenizer = BertTokenizer.from_pretrained("/root/autodl-tmp/models/bert-base-uncased")
    bert_model = BertModel.from_pretrained("/root/autodl-tmp/models/bert-base-uncased").eval().to("cuda")

    input_file = "/root/autodl-tmp/ProtoCoT/preparatory-work/gsm8k_train_500_with_labels.json"
    output_file = "/root/autodl-tmp/ProtoCoT/preparatory-work/gsm8k_train_500_encoded.json"

    with open(input_file, "r") as f:
        data = json.load(f)

    encoded_data = []

    for item in tqdm.tqdm(data):
        question = item["question"]
        paths = item["paths"]
        
        encoded_paths = []
        for path in paths:
            reasoning = path["reasoning"]
            encoding = encode_reasoning(bert_model, bert_tokenizer, question, reasoning)
            encoded_paths.append({
                "reasoning": reasoning,
                "encoded_seq": encoding["seq"].cpu().numpy().tolist(),
                "is_correct": path["is_correct"]
            })
        
        encoded_data.append({
            "question": question,
            "ground_truth": item["ground_truth"],
            "paths": encoded_paths
        })

    with open(output_file, "w") as f:
        json.dump(encoded_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Encoded representations saved to {output_file}")

if __name__ == "__main__":
    main()