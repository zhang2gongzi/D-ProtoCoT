# generate_test_cot.py

import os
import json
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def extract_ground_truth(answer_str):
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip()
    match = re.search(r"\\boxed\{(.+?)\}", answer_str)
    if match:
        return match.group(1).strip()
    return ""

def extract_pred_answer(cot_text):
    boxed_match = re.search(r"\\boxed\{([^}]*)\}", cot_text)
    if boxed_match:
        ans = boxed_match.group(1).strip()
        ans = re.sub(r"[^\d]", "", ans)
        return ans if ans else None

    lines = cot_text.strip().split("\n")
    for line in reversed(lines):
        if any(kw in line.lower() for kw in ["answer", "therefore", "thus", "so "]):
            numbers = re.findall(r"\d+", line)
            if numbers:
                return numbers[-1]

    all_numbers = re.findall(r"\d+", cot_text)
    return all_numbers[-1] if all_numbers else None

def main():
    model_path = "/root/autodl-tmp/models/Qwen/Qwen3-8B"
    test_data_path = "/root/autodl-tmp/ProtoCoT/database/gsm8k/test-00000-of-00001.parquet"
    output_dir = "/root/autodl-tmp/ProtoCoT/preparatory-work"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "gsm8k_test_500_cot_paths.json")

    # Load model
    print("Loading Qwen2.5...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True  # 👈 关键：强制从本地加载
    )
    model.eval()

    # Load test data
    df = pd.read_parquet(test_data_path)
    questions = df["question"].tolist()
    answers = df["answer"].tolist()
    
    num_samples = min(500, len(questions))  # 改成 len(questions) 可跑全部 1319
    print(f"Generating CoT for {num_samples} test samples...")

    results = []
    for i in tqdm(range(num_samples)):
        question = questions[i]
        gt_answer = extract_ground_truth(answers[i])

        paths = []
        for _ in range(10):  # K=10
            try:
                messages = [{"role": "user", "content": f"Question: {question}\nLet's think step by step."}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id
                    )

                response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                reasoning = response.strip()
                pred_ans = extract_pred_answer(reasoning)

                paths.append({
                    "reasoning": reasoning,
                    "predicted_answer": pred_ans
                })
            except Exception as e:
                print(f"Error at {i}: {e}")
                paths.append({"reasoning": "", "predicted_answer": None})

        results.append({
            "question": question,
            "ground_truth": gt_answer,
            "paths": paths
        })

        # 实时保存
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved to {output_file}")

if __name__ == "__main__":
    main()