# label_gsm8k_paths.py

import json
import re

def clean_answer(ans):
    if ans is None:
        return ""
    ans = str(ans).strip()
    # 只保留数字（GSM8K 答案均为正整数）
    return re.sub(r"\D", "", ans)

def is_correct(pred, gt):
    return clean_answer(pred) == clean_answer(gt)

def main():
    input_file = "/root/autodl-tmp/ProtoCoT/preparatory-work/gsm8k_train_500_cot_paths.json"
    output_file = "/root/autodl-tmp/ProtoCoT/preparatory-work/gsm8k_train_500_with_labels.json"

    with open(input_file, "r") as f:
        data = json.load(f)

    for item in data:
        gt = item["ground_truth"]
        for path in item["paths"]:
            pred = path.get("predicted_answer", "")
            path["is_correct"] = is_correct(pred, gt)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Labels added and saved to {output_file}")

if __name__ == "__main__":
    main()