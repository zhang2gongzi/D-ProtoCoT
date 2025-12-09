import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ===================== 1. 配置参数（适配StrategyQA） =====================
DATASET_PATH = "/root/autodl-tmp/ProtoCoT/database/StrategyQA/strategyqa_train_filtered.json"
MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen3-8B"
MAX_SAMPLES = 500  # 测评前500条
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# JSON结果保存路径
RESULT_JSON_PATH = "/root/autodl-tmp/ProtoCoT/baseline/standardcot/qwen/strategyqa_cot_results_final.json"
# StrategyQA专属CoT提示模板（结合description强化推理）
COT_PROMPT_TEMPLATE = """
请结合以下背景信息，严格按照步骤回答问题：
背景信息：{description}（术语：{term}）
1. 分析问题核心需求，结合背景信息梳理推理逻辑；
2. 逐步推导，说明每一步的推理依据（需关联背景信息）；
3. 最终答案仅输出「是」或「否」（无需其他文字）。

问题：{question}

推理过程：
"""

# ===================== 2. 加载模型和Tokenizer（适配Qwen） =====================
def load_model_and_tokenizer():
    """加载Qwen3-8B（关闭采样，保证推理确定性）"""
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
    # 加载模型原生生成配置
    gen_config = GenerationConfig.from_pretrained(MODEL_PATH)
    gen_config.temprature = 0.0  # 0温度保证推理稳定
    gen_config.do_sample = False
    gen_config.max_new_tokens = 512  # 适配策略推理长度
    gen_config.eos_token_id = tokenizer.eos_token_id
    gen_config.pad_token_id = tokenizer.pad_token_id
    gen_config.bos_token_id = 151643
    
    return tokenizer, model, gen_config

# ===================== 3. 加载并预处理StrategyQA数据集（匹配你的数据格式） =====================
def load_dataset_strategyqa(path, max_samples):
    """加载StrategyQA数据集（适配qid/term/description/question/answer格式）"""
    # 加载JSON文件
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    # 取前max_samples条
    raw_data = raw_data[:max_samples]
    dataset = []
    
    # 打印前2条调试信息（匹配你的数据格式）
    print("=== StrategyQA数据集格式调试（前2条） ===")
    for idx in range(min(2, len(raw_data))):
        sample = raw_data[idx]
        print(f"样本{idx} qid：{sample.get('qid', '无')}")
        print(f"样本{idx} term：{sample.get('term', '无')}")
        print(f"样本{idx} description：{sample.get('description', '无')}")
        print(f"样本{idx} question：{sample.get('question', '无')}")
        print(f"样本{idx} answer：{sample.get('answer', '无')}")
    
    # 预处理：提取所有字段+转换答案为「是/否」
    for idx, sample in enumerate(raw_data):
        # 提取核心字段（匹配你的数据格式）
        qid = sample.get("qid", str(idx))
        term = sample.get("term", "").strip()
        description = sample.get("description", "").strip()
        question = sample.get("question", "").strip()
        raw_answer = sample.get("answer", None)
        
        # 过滤无效样本
        if not question or raw_answer is None:
            print(f"警告：样本{idx}（qid:{qid}）字段缺失，跳过")
            continue
        
        # 转换布尔值答案为「是/否」
        correct_answer = "是" if raw_answer else "否"
        
        dataset.append({
            "qid": qid,
            "term": term,
            "description": description,
            "question": question,
            "correct_answer": correct_answer,
            "raw_answer": raw_answer,
            "idx": idx
        })
    
    print(f"\n成功加载{len(dataset)}条有效StrategyQA样本（前{max_samples}条）")
    return dataset

# ===================== 4. StrategyQA专属CoT推理函数（结合term/description） =====================
def cot_inference_strategyqa(tokenizer, model, gen_config, term, description, question):
    """执行StrategyQA的CoT推理（结合背景信息），提取「是/否」答案"""
    # 构建策略推理提示（融入term和description）
    prompt = COT_PROMPT_TEMPLATE.format(
        term=term,
        description=description,
        question=question
    )
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
    
    # 精准提取「是/否」答案
    pred_answer = None
    # 1. 优先匹配「最终答案：是/否」
    match = re.search(r"最终答案：\s*(是|否)", response)
    if match:
        pred_answer = match.group(1)
    else:
        # 2. 兜底：提取文本中最后出现的「是/否」
        pos_yes = response.rfind("是")
        pos_no = response.rfind("否")
        if pos_yes > pos_no:
            pred_answer = "是"
        elif pos_no > pos_yes:
            pred_answer = "否"
        else:
            # 极端情况兜底为「否」
            pred_answer = "否"
    
    return pred_answer, response

# ===================== 5. 测评主函数（StrategyQA） =====================
def evaluate_cot_strategyqa():
    """StrategyQA测评主函数：计算「是/否」匹配准确率"""
    # 加载模型和数据
    print(f"使用GPU：{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "使用CPU")
    print("加载Qwen3-8B模型...")
    tokenizer, model, gen_config = load_model_and_tokenizer()
    print("加载StrategyQA数据集...")
    dataset = load_dataset_strategyqa(DATASET_PATH, MAX_SAMPLES)
    
    # 初始化结果
    result_list = []
    correct = 0
    total = len(dataset)
    print(f"开始测评前{total}条有效StrategyQA数据...")
    
    for sample in dataset:
        idx = sample["idx"]
        qid = sample["qid"]
        term = sample["term"]
        description = sample["description"]
        question = sample["question"]
        correct_ans = sample["correct_answer"]
        
        # 执行CoT推理（传入term和description）
        pred_ans, cot_process = cot_inference_strategyqa(
            tokenizer, model, gen_config,
            term=term,
            description=description,
            question=question
        )
        
        # 判断是否正确
        is_correct = 1 if pred_ans == correct_ans else 0
        if is_correct:
            correct += 1
        
        # 保存单条结果（保留所有原始字段）
        result_list.append({
            "qid": qid,
            "sample_idx": idx,
            "term": term,
            "description": description,
            "question": question,
            "correct_answer": correct_ans,
            "predicted_answer": pred_ans,
            "is_correct": is_correct,
            "raw_answer": sample["raw_answer"],
            "cot_reasoning_process": cot_process
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
    print(f"\n===== StrategyQA测评完成 ======")
    print(f"有效样本数：{total}")
    print(f"正确样本数：{correct}")
    print(f"最终准确率：{final_accuracy:.4f}")
    print(f"结果已保存至：{RESULT_JSON_PATH}")
    
    return final_accuracy

# ===================== 执行测评 =====================
if __name__ == "__main__":
    evaluate_cot_strategyqa()