# generate_commonsenseqa_llama3.py
import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === 配置 ===
MODEL_PATH = "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct"
DATA_PATH = "/root/autodl-tmp/ProtoCoT/database/commonsenseQA/train-00000-of-00001.parquet"
OUTPUT_JSON = "/root/autodl-tmp/ProtoCoT/llamatry/commonsenseqa/commonsenseqa_500_cot_llama3.json"

NUM_SAMPLES = 500
NUM_PATHS = 10
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# === 加载 tokenizer 和 model ===
print("Loading Llama-3.1-8B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  # Llama 没有 pad_token，需手动设置

# 自动选择 dtype
if torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map="auto",
    # attn_implementation="flash_attention_2"  # 已移除，避免 flash-attn 依赖
)
model.eval()

# === 加载数据集 ===
print("Loading CommonsenseQA train set...")
dataset = load_dataset("parquet", data_files=DATA_PATH)["train"]
print(f"Total samples in dataset: {len(dataset)}")

# 取前 500 条
samples = dataset.select(range(min(NUM_SAMPLES, len(dataset))))

# === 构造 prompt（Llama-3 Instruct 格式）===
def build_prompt(question, choice_dict):
    choice_str = "\n".join([f"{k}. {v}" for k, v in choice_dict.items()])
    user_content = (
        f"Question: {question}\n"
        f"Choices:\n{choice_str}\n\n"
        "Let's think step by step. Provide your reasoning and end with 'Therefore, the answer is X.' where X is one of A, B, C, D, or E."
    )
    messages = [
        {"role": "user", "content": user_content}
    ]
    return messages

# === 提取最终答案 ===
def extract_answer(text):
    text = text.upper()
    # 尝试匹配标准结尾
    for pattern in ["THE ANSWER IS", "ANSWER IS", "FINAL ANSWER IS"]:
        if pattern in text:
            idx = text.rfind(pattern)
            tail = text[idx + len(pattern):].strip()
            if tail and tail[0] in "ABCDE":
                return tail[0]
    # 否则从后往前找第一个 A-E
    for c in reversed(text):
        if c in "ABCDE":
            return c
    return None

# === 主生成循环 ===
results = []

for i, item in enumerate(samples):
    print(f"\nProcessing sample {i+1}/{NUM_SAMPLES}...")

    question = item["question"]  # ← 字符串！
    labels = item["choices"]["label"]   # list of str, e.g. ['A','B',...]
    texts = item["choices"]["text"]     # list of str
    choice_dict = dict(zip(labels, texts))
    ground_truth = item["answerKey"]

    paths = []
    for path_id in range(NUM_PATHS):
        torch.manual_seed(SEED + i * NUM_PATHS + path_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + i * NUM_PATHS + path_id)

        messages = build_prompt(question, choice_dict)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096  # Llama-3 支持 8k，但安全起见用 4k
        ).to(model.device)

        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1]:],
                skip_special_tokens=True
            ).strip()
        except Exception as e:
            print(f"⚠️ Generation error on sample {i}, path {path_id}: {e}")
            generated_text = ""

        pred_answer = extract_answer(generated_text)
        is_correct = (pred_answer == ground_truth) if pred_answer else False

        paths.append({
            "text": generated_text,
            "pred_answer": pred_answer,
            "is_correct": is_correct
        })

    results.append({
        "question": question,
        "choices": choice_dict,
        "ground_truth": ground_truth,
        "paths": paths
    })

# === 保存结果 ===
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Done! Results saved to: {OUTPUT_JSON}")
print(f"Total samples: {len(results)}")
print(f"Paths per sample: {NUM_PATHS}")