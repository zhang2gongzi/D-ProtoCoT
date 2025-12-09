import torch
import pandas as pd
import json
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ===================== 1. 配置参数（适配GSM8K） =====================
DATASET_PATH = "/root/autodl-tmp/ProtoCoT/database/gsm8k/test-00000-of-00001.parquet"
MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3-8B"
MAX_SAMPLES = 500  # 测评前500条
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# JSON结果保存路径
RESULT_JSON_PATH = "/root/autodl-tmp/ProtoCoT/baseline/standardcot/qwen/gsm8k_cot_results_final.json"
# GSM8K专属CoT提示模板（数学推理优化）
COT_PROMPT_TEMPLATE = """
请严格按照以下步骤解答数学应用题：
1. 分析题目中的已知条件和待求问题；
2. 列出解题所需的公式/步骤；
3. 逐步计算，写出每一步的计算过程；
4. 最终答案：仅输出数字（无需单位、文字说明）。

题目：{question}

解题过程：
"""

# ===================== 2. 加载模型和Tokenizer（适配Qwen） =====================
def load_model_and_tokenizer():
    """加载Qwen3-8B（关闭采样，保证数学推理确定性）"""
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
    # 加载模型原生生成配置，覆盖为数学推理参数
    gen_config = GenerationConfig.from_pretrained(MODEL_PATH)
    gen_config.temprature = 0.0  # 0温度保证计算精准
    gen_config.do_sample = False
    gen_config.max_new_tokens = 1024  # 足够容纳数学推导过程
    gen_config.eos_token_id = tokenizer.eos_token_id
    gen_config.pad_token_id = tokenizer.pad_token_id
    gen_config.bos_token_id = 151643
    
    return tokenizer, model, gen_config

# ===================== 3. 加载并预处理GSM8K数据集 =====================
def load_dataset_gsm8k(path, max_samples):
    """加载GSM8K数据集（适配其字段格式）"""
    df = pd.read_parquet(path)
    df = df.head(max_samples)
    dataset = []
    
    print("=== GSM8K数据集格式调试（前2条） ===")
    for idx in range(min(2, len(df))):
        row = df.iloc[idx]
        print(f"样本{idx}字段：{list(df.columns)}")
        print(f"样本{idx} question：{row.get('question', '无')[:200]}")
        print(f"样本{idx} answer：{row.get('answer', '无')[:200]}")
    
    for idx, row in df.iterrows():
        # GSM8K核心字段：question（题目）、answer（带计算过程的答案）
        question = row.get("question", "").strip()
        answer_raw = row.get("answer", "").strip()
        
        # 提取正确答案（从answer_raw中提取最终数字）
        # GSM8K的answer格式："xxx计算过程xxx\n#### 数字"
        correct_answer = None
        if "####" in answer_raw:
            # 提取####后的数字
            num_str = answer_raw.split("####")[-1].strip()
            # 清洗数字（去除逗号、单位等）
            num_str = re.sub(r"[^\d\.]", "", num_str)
            try:
                correct_answer = float(num_str)
                # 整数转int，保持格式统一
                if correct_answer.is_integer():
                    correct_answer = int(correct_answer)
            except:
                correct_answer = None
        
        # 只保留有效样本
        if question and correct_answer is not None:
            dataset.append({
                "question": question,
                "correct_answer": correct_answer,
                "answer_raw": answer_raw,
                "idx": idx,
                "id": row.get("id", idx)  # 兜底ID
            })
    
    print(f"\n成功加载{len(dataset)}条有效GSM8K样本（前{max_samples}条）")
    return dataset

# ===================== 4. GSM8K专属CoT推理函数 =====================
def cot_inference_gsm8k(tokenizer, model, gen_config, question):
    """执行GSM8K的CoT推理，提取预测数字"""
    # 构建数学推理提示
    prompt = COT_PROMPT_TEMPLATE.format(question=question)
    # 编码提示
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(DEVICE)
    # 生成推理过程
    outputs = model.generate(
        **inputs,
        generation_config=gen_config
    )
    # 解码结果
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取预测答案（从推理结果中提取最终数字）
    pred_answer = None
    # 优先匹配「最终答案：数字」格式
    match = re.search(r"最终答案：\s*([\d\.]+)", response)
    if match:
        num_str = match.group(1)
    else:
        # 兜底：提取最后出现的数字
        num_matches = re.findall(r"[\d\.]+", response)
        num_str = num_matches[-1] if num_matches else ""
    
    # 转换为数字（兼容整数/小数）
    try:
        pred_answer = float(num_str)
        if pred_answer.is_integer():
            pred_answer = int(pred_answer)
    except:
        pred_answer = None
    
    return pred_answer, response

# ===================== 5. 测评主函数（GSM8K） =====================
def evaluate_cot_gsm8k():
    """GSM8K测评主函数：计算数字匹配准确率"""
    # 加载模型和数据
    print(f"使用GPU：{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "使用CPU")
    print("加载Qwen3-8B模型...")
    tokenizer, model, gen_config = load_model_and_tokenizer()
    print("加载GSM8K数据集...")
    dataset = load_dataset_gsm8k(DATASET_PATH, MAX_SAMPLES)
    
    # 初始化结果
    result_list = []
    correct = 0
    total = len(dataset)
    print(f"开始测评前{total}条有效GSM8K数据...")
    
    for sample in dataset:
        idx = sample["idx"]
        question = sample["question"]
        correct_ans = sample["correct_answer"]
        
        # 执行CoT推理
        pred_ans, cot_process = cot_inference_gsm8k(tokenizer, model, gen_config, question)
        
        # 判断是否正确（数字完全匹配）
        is_correct = 1 if pred_ans == correct_ans else 0
        if is_correct:
            correct += 1
        
        # 保存单条结果
        result_list.append({
            "sample_id": sample["id"],
            "sample_idx": idx,
            "question": question,
            "correct_answer": correct_ans,
            "predicted_answer": pred_ans,
            "is_correct": is_correct,
            "cot_reasoning_process": cot_process,
            "original_answer": sample["answer_raw"]
        })
        
        # 打印进度
        if (idx + 1) % 50 == 0:
            current_acc = correct / (idx + 1)
            print(f"进度：{idx+1}/{total}，当前准确率：{current_acc:.4f}")
    
    # 计算最终准确率
    final_accuracy = correct / total if total > 0 else 0.0
    # 构造最终结果
    final_result = {
        "overall_metrics": {
            "total_valid_samples": total,
            "correct_samples": correct,
            "accuracy": final_accuracy
        },
        "sample_details": result_list
    }
    
    # 保存JSON结果
    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4)
    
    # 打印最终结果
    print(f"\n===== GSM8K测评完成 ======")
    print(f"有效样本数：{total}")
    print(f"正确样本数：{correct}")
    print(f"最终准确率：{final_accuracy:.4f}")
    print(f"结果已保存至：{RESULT_JSON_PATH}")
    
    return final_accuracy

# ===================== 执行测评 =====================
if __name__ == "__main__":
    evaluate_cot_gsm8k()