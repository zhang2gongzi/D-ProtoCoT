import torch
import pandas as pd
import json
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ===================== 1. 配置参数 =====================
DATASET_PATH = "/root/autodl-tmp/ProtoCoT/database/commonsenseQA/train-00000-of-00001.parquet"
MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3-8B"
MAX_SAMPLES = 500  # 测评前500条
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# JSON结果保存路径
RESULT_JSON_PATH = "/root/autodl-tmp/ProtoCoT/baseline/standardcot/qwen/commonsenseqa_cot_results_final_1208.json"
# 优化后的CoT提示模板（强化格式约束）
COT_PROMPT_TEMPLATE = """
请严格遵守以下要求回答问题：
1. 先分析问题核心和每个选项的合理性；
2. 推理过程需结合常识，逐一排除错误选项；
3. 最终答案仅输出A/B/C/D/E中的一个字母，无其他内容；
4. 推理过程结束后，必须以「最终答案：X」格式给出答案。

问题：{question}
选项：
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}
E. {choice_e}

推理过程：
"""

# ===================== 2. 加载模型和Tokenizer（适配Qwen原生参数） =====================
def load_model_and_tokenizer():
    """加载Qwen3-8B（使用模型原生生成参数，避免无效警告）"""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    # Qwen模型原生生成参数（仅保留有效参数）
    gen_config = GenerationConfig.from_pretrained(MODEL_PATH)
    # 强制覆盖为确定性推理参数
    gen_config.temprature = 0.0  # Qwen原生参数名：temprature（无e）
    gen_config.do_sample = False  # 关闭采样
    gen_config.max_new_tokens = 512
    gen_config.eos_token_id = tokenizer.eos_token_id
    gen_config.pad_token_id = tokenizer.pad_token_id
    gen_config.bos_token_id = 151643  # Qwen默认bos_token_id
    
    return tokenizer, model, gen_config

# ===================== 3. 加载并预处理数据集 =====================
def load_dataset_commonsenseqa(path, max_samples):
    """加载CommonsenseQA数据集（适配实际格式）"""
    df = pd.read_parquet(path)
    df = df.head(max_samples)
    dataset = []
    
    for idx, row in df.iterrows():
        question = row["question"].strip()
        choices_dict = row["choices"]
        labels = choices_dict["label"].tolist() if isinstance(choices_dict["label"], np.ndarray) else choices_dict["label"]
        texts = choices_dict["text"].tolist() if isinstance(choices_dict["text"], np.ndarray) else choices_dict["text"]
        
        choice_map = {}
        for label, text in zip(labels, texts):
            choice_map[label] = text.strip()
        for letter in ["A", "B", "C", "D", "E"]:
            if letter not in choice_map:
                choice_map[letter] = ""
        
        answer_key = row["answerKey"].strip().upper()
        if question and answer_key in ["A", "B", "C", "D", "E"]:
            dataset.append({
                "question": question,
                "choices": choice_map,
                "answer_key": answer_key,
                "idx": idx,
                "id": row["id"]
            })
    
    print(f"成功加载{len(dataset)}条有效样本（前{max_samples}条）")
    return dataset

# ===================== 4. CoT推理（强化答案提取） =====================
def cot_inference(tokenizer, model, gen_config, question, choices):
    """执行CoT推理（精准提取答案）"""
    prompt = COT_PROMPT_TEMPLATE.format(
        question=question,
        choice_a=choices["A"],
        choice_b=choices["B"],
        choice_c=choices["C"],
        choice_d=choices["D"],
        choice_e=choices["E"]
    )
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(DEVICE)
    
    outputs = model.generate(
        **inputs,
        generation_config=gen_config
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 精准提取「最终答案：X」中的字母
    pred_answer = None
    match = re.search(r"最终答案：\s*([A-E])", response)
    if match:
        pred_answer = match.group(1)
    else:
        # 兜底逻辑
        for char in ["A", "B", "C", "D", "E"]:
            if char in response[-10:]:
                pred_answer = char
                break
    if pred_answer is None:
        pred_answer = "A"
    
    return pred_answer, response

# ===================== 5. 测评主函数 =====================
def evaluate_cot_accuracy():
    """主测评函数"""
    print(f"使用GPU：{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "使用CPU")
    print("加载Qwen3-8B模型...")
    tokenizer, model, gen_config = load_model_and_tokenizer()
    print("加载CommonsenseQA数据集...")
    dataset = load_dataset_commonsenseqa(DATASET_PATH, MAX_SAMPLES)
    
    result_list = []
    correct = 0
    total = len(dataset)
    print(f"开始测评前{total}条有效数据...")
    
    for sample in dataset:
        idx = sample["idx"]
        pred_answer, cot_process = cot_inference(tokenizer, model, gen_config, sample["question"], sample["choices"])
        
        is_correct = 1 if pred_answer == sample["answer_key"] else 0
        correct += is_correct
        
        result_list.append({
            "sample_id": sample["id"],
            "sample_idx": idx,
            "question": sample["question"],
            "choices": sample["choices"],
            "correct_answer": sample["answer_key"],
            "predicted_answer": pred_answer,
            "is_correct": is_correct,
            "cot_reasoning_process": cot_process
        })
        
        if (idx + 1) % 50 == 0:
            current_acc = correct / (idx + 1)
            print(f"进度：{idx+1}/{total}，当前准确率：{current_acc:.4f}")
    
    final_accuracy = correct / total if total > 0 else 0.0
    final_result = {
        "overall_metrics": {
            "total_valid_samples": total,
            "correct_samples": correct,
            "accuracy": final_accuracy
        },
        "sample_details": result_list
    }
    
    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4)
    
    print(f"\n===== 测评完成 ======")
    print(f"有效样本数：{total}")
    print(f"正确样本数：{correct}")
    print(f"最终准确率：{final_accuracy:.4f}")
    print(f"结果已保存至：{RESULT_JSON_PATH}")
    
    return final_accuracy

# ===================== 执行 =====================
if __name__ == "__main__":
    evaluate_cot_accuracy()