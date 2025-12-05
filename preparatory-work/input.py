import json
import re

def count_matching_answers(results_path, output_log=True):
    """
    计算JSON结果文件中correct_answer与generated_answer匹配的个数
    适配场景：generated_answer可能包含多余文本、换行、符号等，仅提取核心Yes/No
    """
    # 加载结果文件
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except Exception as e:
        print(f"加载文件失败：{e}")
        return 0, 0, 0.0

    total_samples = len(results)
    matching_count = 0
    mismatched_samples = []  # 存储不匹配的样本（便于后续分析）

    # 遍历每个样本，提取并对比答案
    for idx, sample in enumerate(results):
        # 获取正确答案（标准化为Yes/No）
        correct_ans = sample.get("correct_answer", "").strip()
        if correct_ans not in ["Yes", "No"]:
            print(f"警告：样本{idx}的correct_answer格式异常（值：{correct_ans}），跳过该样本")
            continue

        # 获取生成答案，提取核心Yes/No（忽略多余文本）
        generated_text = sample.get("generated_answer", "").strip()
        # 用正则匹配首个Yes/No（不区分大小写，忽略前后文本）
        match = re.search(r"\b(Yes|No)\b", generated_text, re.IGNORECASE)
        if not match:
            # 若未匹配到，视为不匹配
            mismatched_samples.append({
                "sample_id": idx,
                "correct_answer": correct_ans,
                "generated_answer": generated_text[:100] + "..." if len(generated_text) > 100 else generated_text
            })
            continue
        
        generated_ans = match.group(1).capitalize()  # 标准化为Yes/No

        # 对比答案
        if generated_ans == correct_ans:
            matching_count += 1
        else:
            mismatched_samples.append({
                "sample_id": idx,
                "correct_answer": correct_ans,
                "generated_answer": generated_text[:100] + "..." if len(generated_text) > 100 else generated_text,
                "extracted_generated_answer": generated_ans
            })

    # 计算准确率
    accuracy = (matching_count / total_samples) * 100 if total_samples > 0 else 0.0

    # 输出统计结果
    if output_log:
        print("="*50)
        print(f"结果统计：")
        print(f"总样本数：{total_samples}")
        print(f"答案匹配个数：{matching_count}")
        print(f"答案不匹配个数：{total_samples - matching_count}")
        print(f"准确率：{accuracy:.2f}%")
        print("="*50)

        # 输出前5个不匹配样本（便于快速排查）
        if mismatched_samples:
            print("\n前5个不匹配样本示例：")
            for i, bad_sample in enumerate(mismatched_samples[:5]):
                print(f"\n样本{i+1}（ID：{bad_sample['sample_id']}）：")
                print(f"  正确答案：{bad_sample['correct_answer']}")
                print(f"  生成答案（截取前100字符）：{bad_sample['generated_answer']}")
                if "extracted_generated_answer" in bad_sample:
                    print(f"  提取的核心答案：{bad_sample['extracted_generated_answer']}")

    return total_samples, matching_count, accuracy

# 运行脚本（替换为你的结果文件路径）
if __name__ == "__main__":
    RESULTS_PATH = "/root/autodl-tmp/ProtoCoT/baseline/ccot/llama/contrastive_cot_results/contrastive_cot_strategyqa_results.json"
    count_matching_answers(RESULTS_PATH)