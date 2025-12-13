import torch
import pandas as pd
import json
import re
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ===================== 1. 配置参数（SC+CommonsenseQA） =====================
DATASET_PATH = "/root/autodl-tmp/ProtoCoT/database/commonsenseQA/train-00000-of-00001.parquet"
MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3-8B"
MAX_SAMPLES = 500  # 测评前500条
SC_NUM = 5  # SC推理路径条数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# JSON结果保存路径
RESULT_JSON_PATH = "/root/autodl-tmp/ProtoCoT/baseline/selfc/qwen/commonsenseqa_sc_results_final_1210.json"
# CommonsenseQA专属SC提示模板（保留CoT推理引导）
SC_PROMPT_TEMPLATE = """
请按照以下步骤回答常识问题：
1. 分析问题的核心需求和关键信息；
2. 逐一分析每个选项的合理性；
3. 最终给出答案（仅输出选项字母，如A/B/C/D/E）。

问题：{question}
选项：
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}
E. {choice_e}

请输出你的推理过程和最终答案：
"""

# ===================== 2. 加载模型和Tokenizer（适配SC策略） =====================
def load_model_and_tokenizer():
    """加载Qwen3-8B（SC策略需开启采样，保证推理路径多样性）"""
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
    # SC策略生成配置（开启采样，保证路径多样性）
    gen_config = GenerationConfig.from_pretrained(MODEL_PATH)
    gen_config.temprature = 0.7  # 适中温度保证多样性+稳定性
    gen_config.do_sample = True   # 必须开启采样
    gen_config.top_p = 0.95
    gen_config.top_k = 40
    gen_config.max_new_tokens = 512
    gen_config.eos_token_id = tokenizer.eos_token_id
    gen_config.pad_token_id = tokenizer.pad_token_id
    gen_config.bos_token_id = 151643
    gen_config.num_return_sequences = 1  # 单条生成，循环SC_NUM次
    
    return tokenizer, model, gen_config

# ===================== 3. 加载并预处理CommonsenseQA数据集 =====================
def load_dataset_commonsenseqa(path, max_samples):
    """加载CommonsenseQA数据集（匹配实际格式）"""
    df = pd.read_parquet(path)
    df = df.head(max_samples)
    dataset = []
    
    # 打印前2条调试信息
    print("=== CommonsenseQA数据集格式调试（前2条） ===")
    for idx in range(min(2, len(df))):
        row = df.iloc[idx]
        print(f"样本{idx}字段：{list(df.columns)}")
        print(f"样本{idx} question：{row.get('question', '无')[:200]}")
        print(f"样本{idx} choices：{row.get('choices', '无')}")
        print(f"样本{idx} answerKey：{row.get('answerKey', '无')}")
    
    for idx, row in df.iterrows():
        # 提取纯字符串问题
        question = row["question"].strip()
        # 解析choices字典（label/text数组）
        choices_dict = row["choices"]
        labels = choices_dict["label"].tolist() if isinstance(choices_dict["label"], np.ndarray) else choices_dict["label"]
        texts = choices_dict["text"].tolist() if isinstance(choices_dict["text"], np.ndarray) else choices_dict["text"]
        # 构建A-E选项映射
        choice_map = {}
        for label, text in zip(labels, texts):
            choice_map[label] = text.strip()
        # 补全A-E
        for letter in ["A", "B", "C", "D", "E"]:
            if letter not in choice_map:
                choice_map[letter] = ""
        # 正确答案
        answer_key = row["answerKey"].strip().upper()
        
        # 过滤有效样本
        if question and answer_key in ["A", "B", "C", "D", "E"]:
            dataset.append({
                "question": question,
                "choices": choice_map,
                "answer_key": answer_key,
                "idx": idx,
                "id": row["id"]
            })
    
    print(f"\n成功加载{len(dataset)}条有效CommonsenseQA样本（前{max_samples}条）")
    return dataset

# ===================== 4. SC策略核心函数 =====================
def sc_inference(tokenizer, model, gen_config, question, choices, sc_num):
    """执行SC策略：生成sc_num条CoT推理路径，投票确定最终答案"""
    # 存储每条推理路径的答案
    sc_answers = []
    # 存储每条推理路径的完整过程
    sc_processes = []
    
    for i in range(sc_num):
        # 构建SC提示
        prompt = SC_PROMPT_TEMPLATE.format(
            question=question,
            choice_a=choices["A"],
            choice_b=choices["B"],
            choice_c=choices["C"],
            choice_d=choices["D"],
            choice_e=choices["E"]
        )
        # 编码提示
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(DEVICE)
        # 生成单条推理路径
        outputs = model.generate(
            **inputs,
            generation_config=gen_config
        )
        # 解码结果
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取单条路径的答案
        pred_answer = None
        # 优先匹配「最终答案：X」
        match = re.search(r"最终答案：\s*([A-E])", response)
        if match:
            pred_answer = match.group(1)
        else:
            # 兜底：最后10字符找A-E
            for char in ["A", "B", "C", "D", "E"]:
                if char in response[-10:]:
                    pred_answer = char
                    break
        # 兜底为A
        if pred_answer is None:
            pred_answer = "A"
        
        sc_answers.append(pred_answer)
        sc_processes.append({
            "sc_idx": i,
            "pred_answer": pred_answer,
            "reasoning_process": response
        })
    
    # SC投票：取出现次数最多的答案
    vote_counter = Counter(sc_answers)
    final_answer = vote_counter.most_common(1)[0][0]
    
    return final_answer, sc_answers, sc_processes

# ===================== 5. 测评主函数（SC策略） =====================
def evaluate_sc_commonsenseqa():
    """SC策略测评主函数：计算投票后准确率"""
    # 加载模型和数据
    print(f"使用GPU：{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "使用CPU")
    print("加载Qwen3-8B模型（SC策略）...")
    tokenizer, model, gen_config = load_model_and_tokenizer()
    print("加载CommonsenseQA数据集...")
    dataset = load_dataset_commonsenseqa(DATASET_PATH, MAX_SAMPLES)
    
    # 初始化结果
    result_list = []
    correct = 0
    total = len(dataset)
    print(f"开始测评前{total}条有效数据（SC条数：{SC_NUM}）...")
    
    for sample in dataset:
        idx = sample["idx"]
        question = sample["question"]
        choices = sample["choices"]
        answer_key = sample["answer_key"]
        sample_id = sample["id"]
        
        # 执行SC推理
        final_ans, sc_ans_list, sc_process_list = sc_inference(
            tokenizer, model, gen_config,
            question=question,
            choices=choices,
            sc_num=SC_NUM
        )
        
        # 判断是否正确
        is_correct = 1 if final_ans == answer_key else 0
        if is_correct:
            correct += 1
        
        # 保存单条结果（含所有SC路径信息）
        result_list.append({
            "sample_id": sample_id,
            "sample_idx": idx,
            "question": question,
            "choices": choices,
            "correct_answer": answer_key,
            "sc_final_answer": final_ans,
            "sc_answers": sc_ans_list,  # 每条SC路径的答案
            "sc_processes": sc_process_list,  # 每条SC路径的推理过程
            "is_correct": is_correct,
            "vote_distribution": dict(Counter(sc_ans_list))  # 投票分布
        })
        
        # 打印进度
        if (idx + 1) % 50 == 0:
            current_acc = correct / (idx + 1)
            print(f"进度：{idx+1}/{total}，当前准确率：{current_acc:.4f}")
    
    # 计算最终准确率
    final_accuracy = correct / total if total > 0 else 0.0
    # 构造最终结果
    final_result = {
        "sc_config": {
            "sc_num": SC_NUM,
            "temperature": gen_config.temprature,
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k
        },
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
    print(f"\n===== CommonsenseQA SC策略测评完成 ======")
    print(f"SC推理路径条数：{SC_NUM}")
    print(f"有效样本数：{total}")
    print(f"正确样本数：{correct}")
    print(f"最终准确率：{final_accuracy:.4f}")
    print(f"结果已保存至：{RESULT_JSON_PATH}")
    
    return final_accuracy

# ===================== 执行测评 =====================
if __name__ == "__main__":
    evaluate_sc_commonsenseqa()