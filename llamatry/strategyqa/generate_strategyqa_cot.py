# generate_strategyqa_cot.py
import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 配置 =====
MODEL_PATH = "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct"
DATA_PATH = "/root/autodl-tmp/ProtoCoT/database/StrategyQA/strategyqa_train_filtered.json"
OUTPUT_PATH = "/root/autodl-tmp/ProtoCoT/llamatry/strategyqa/strategyqa_500_cot_llama3.json"

NUM_SAMPLES = 500   # 只处理前500条
NUM_PATHS = 10      # 每个问题生成10条CoT
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.8
TOP_P = 0.9

# ===== 创建输出目录 =====
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ===== 加载模型和 tokenizer =====
print("Loading LLaMA-3.1-8B-Instruct model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"  # 若支持 FlashAttention，可改为 "flash_attention_2"
)
model.eval()

# ===== 加载数据 =====
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

data = data[:NUM_SAMPLES]  # 取前500条

# ===== 辅助函数：从生成文本中提取 yes/no 答案 =====
def extract_answer(text: str):
    text = text.lower().strip()
    words = text.split()
    # 优先看结尾附近是否有明确 yes/no
    for i in range(len(words) - 1, max(-1, len(words) - 10), -1):
        w = words[i].strip(".,!?;:")
        if w == "yes":
            return True
        elif w == "no":
            return False
    # 全文搜索（保守）
    if "yes" in text and "no" not in text:
        return True
    if "no" in text and "yes" not in text:
        return False
    # 默认：无法判断
    return None

# ===== 主循环 =====
results = []

for idx, item in enumerate(data):
    question = item["question"]
    ground_truth = item["answer"]  # 直接是 bool: true / false

    print(f"[{idx+1}/{len(data)}] Q: {question[:70]}...")

    paths = []
    for _ in range(NUM_PATHS):
        # 构造 LLaMA-3 Instruct 格式的 prompt
        messages = [
            {"role": "system", "content": "You are a logical assistant. Answer the question by reasoning step by step."},
            {"role": "user", "content": f"{question}\nLet's think step by step."}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = full_text[len(prompt):].strip()

        pred = extract_answer(generated)
        paths.append({
            "text": generated,
            "pred_answer": pred
        })

    results.append({
        "question": question,
        "ground_truth": ground_truth,
        "paths": paths
    })

# ===== 保存结果 =====
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✅ Generation completed!")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Total samples: {len(results)} × {NUM_PATHS} paths")