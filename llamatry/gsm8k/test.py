import json

# 加载你的 JSON 文件
with open("/root/autodl-tmp/ProtoCoT/preparatory-work/gsm8k_test_500_cot_paths.json", "r") as f:
    data = json.load(f)

total = len(data)
correct_oracle = 0
correct_majority = 0

for item in data:
    ground_truth = str(item["ground_truth"]).strip()
    predictions = [str(p["predicted_answer"]).strip() for p in item["paths"]]

    # Oracle: 只要有一个对就算对
    if ground_truth in predictions:
        correct_oracle += 1

    # Majority vote
    from collections import Counter
    vote_count = Counter(predictions)
    majority_pred = vote_count.most_common(1)[0][0]
    if majority_pred == ground_truth:
        correct_majority += 1

print(f"Oracle Accuracy (Upper Bound): {correct_oracle / total * 100:.2f}% ({correct_oracle}/{total})")
print(f"Majority Vote Accuracy: {correct_majority / total * 100:.2f}% ({correct_majority}/{total})")