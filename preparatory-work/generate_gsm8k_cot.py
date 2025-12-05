# generate_gsm8k_cot.py

import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import argparse

def extract_ground_truth(answer_str):
    """从 GSM8K 标准 answer 中提取最终答案（#### 后的内容）"""
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip()
    # 备用：尝试 \boxed{}
    match = re.search(r"\\boxed\{(.+?)\}", answer_str)
    if match:
        return match.group(1).strip()
    return ""

def extract_pred_answer(cot_text):
    """从模型生成的 CoT 文本中提取预测答案"""
    # 1. 优先匹配 \boxed{...}
    boxed_match = re.search(r"\\boxed\{([^}]*)\}", cot_text)
    if boxed_match:
        ans = boxed_match.group(1).strip()
        # 清理可能的多余空格或标点
        ans = re.sub(r"[^\d\-+\.]", "", ans)
        if ans:
            return ans

    # 2. 匹配 "The answer is X" 类表述（从后往前找）
    lines = cot_text.strip().split("\n")
    for line in reversed(lines):
        line_lower = line.lower()
        if any(kw in line_lower for kw in ["answer", "therefore", "thus", "so "]):
            # 提取所有数字（支持负数、小数）
            numbers = re.findall(r"-?\d+\.?\d*", line)
            if numbers:
                return numbers[-1]  # 取最后一个数字

    # 3. 全文找最后一个数字
    all_numbers = re.findall(r"-?\d+\.?\d*", cot_text)
    if all_numbers:
        return all_numbers[-1]

    return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output_file", type=str, default="/root/autodl-tmp/ProtoCoT/preparatory-work/gsm8k_500_samples_cot_paths.json")
    parser.add_torch_dtype("torch_dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    # === 1. 加载模型和 tokenizer ===
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=False
    )
    model.eval()
    print("✅ Model loaded.")

    # === 2. 加载数据集 ===
    print("Loading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main")
    test_set = dataset["test"]
    num_samples = min(500, len(test_set))
    print(f"Will process first {num_samples} samples.")

    # === 3. 检查是否已有部分结果（支持续跑）===
    existing_data = {}
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            try:
                existing_data = {item["question"]: item for item in json.load(f)}
                print(f"Found existing file with {len(existing_data)} samples. Resuming...")
            except:
                pass

    results = []
    start_idx = 0
    if existing_data:
        # 找到第一个未完成的索引
        for i in range(num_samples):
            q = test_set[i]["question"]
            if q not in existing_data:
                start_idx = i
                break
        # 加入已有的结果
        for i in range(start_idx):
            results.append(existing_data[test_set[i]["question"]])
        print(f"Resuming from index {start_idx}")

    # === 4. 生成推理路径 ===
    for i in tqdm(range(start_idx, num_samples), desc="Generating CoT paths"):
        sample = test_set[i]
        question = sample["question"]
        gt_answer = extract_ground_truth(sample["answer"])

        paths = []
        for path_id in range(10):
            try:
                # 构造 Llama-3 聊天格式
                messages = [
                    {"role": "user", "content": f"Question: {question}\nLet's think step by step."}
                ]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)

                # 生成
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                # 解码生成内容（去掉 prompt）
                response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                reasoning = response.strip()
                pred_ans = extract_pred_answer(reasoning)

                paths.append({
                    "reasoning": reasoning,
                    "predicted_answer": pred_ans
                })

            except Exception as e:
                print(f"\nError at sample {i}, path {path_id}: {e}")
                paths.append({
                    "reasoning": "",
                    "predicted_answer": None
                })

        results.append({
            "question": question,
            "ground_truth": gt_answer,
            "paths": paths
        })

        # 实时保存（防中断丢失）
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ All done! Saved to {args.output_file}")

if __name__ == "__main__":
    main()