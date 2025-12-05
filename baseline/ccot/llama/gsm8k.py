import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
from datetime import datetime
import re

class GSM8KCoTEvaluator:
    def __init__(self, model_path, device=None):
        """初始化GSM8K的CoT评估器"""
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型和tokenizer
        print(f"正在加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # 创建文本生成pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        print(f"模型加载完成，使用设备: {self.device}")
    
    def generate_cot_response(self, question, max_new_tokens=1024):
        """生成CoT推理响应（加入GSM8K正负样例）"""
        # 4个GSM8K正负样例（2正2反，覆盖加减乘除、应用题场景，强调分步计算）
        gsm8k_examples = """
### Math Reasoning Examples (Learn step-by-step calculation, avoid mistakes)
Example 1 (Correct Reasoning - Addition & Multiplication):
Question: A bakery makes 12 loaves of bread each hour. They work for 8 hours a day, 5 days a week. How many loaves do they make in a week?
Reasoning: Step 1: Calculate loaves made per day. If they make 12 loaves per hour and work 8 hours, that's 12 × 8 = 96 loaves per day. Step 2: Calculate loaves per week. They work 5 days, so 96 × 5 = 480 loaves. Step 3: Verify the calculation—12×8=96 (correct), 96×5=480 (correct).
Final Answer: 480

Example 2 (Correct Reasoning - Subtraction & Division):
Question: A store has 360 apples. They sell 144 apples on Monday and 96 apples on Tuesday. The remaining apples are divided equally into 8 baskets. How many apples are in each basket?
Reasoning: Step 1: Calculate total apples sold. 144 (Monday) + 96 (Tuesday) = 240 apples sold. Step 2: Calculate remaining apples. 360 - 240 = 120 apples left. Step 3: Divide remaining apples into 8 baskets. 120 ÷ 8 = 15 apples per basket. Step 4: Check—144+96=240, 360-240=120, 120÷8=15 (all correct).
Final Answer: 15

Example 3 (Incorrect vs Correct Reasoning - Missing Step):
Question: A train travels 60 miles per hour for 3 hours, then 45 miles per hour for 2 hours. What is the total distance traveled?
Incorrect Reasoning: Add the speeds and multiply by total time: (60+45) × (3+2) = 105 × 5 = 525 miles.
Correct Reasoning: Step 1: Calculate distance for first part. Speed × time = 60 × 3 = 180 miles. Step 2: Calculate distance for second part. 45 × 2 = 90 miles. Step 3: Add both distances for total. 180 + 90 = 270 miles. The mistake in incorrect reasoning is mixing speed and time—total distance requires calculating each segment separately.
Final Answer: 270

Example 4 (Incorrect vs Correct Reasoning - Miscalculation):
Question: Lily has $45. She buys 3 books that cost $8 each and a pen that costs $5. How much money does she have left?
Incorrect Reasoning: Cost of books: 3 × 8 = 24. Total cost: 24 + 5 = 30. Money left: 45 - 30 = 25.
Correct Reasoning: Step 1: Calculate book cost. 3 books × $8 = $24. Step 2: Add pen cost. $24 + $5 = $29 total spent. Step 3: Calculate remaining money. $45 - $29 = $16. The incorrect reasoning had a miscalculation in total cost (24+5=29, not 30) leading to wrong result.
Final Answer: 16
"""
        
        # 构建提示词（融入样例，强调分步计算和格式）
        prompt = f"""{gsm8k_examples}

Now solve the following math problem using the same step-by-step reasoning method. Show each calculation clearly, avoid missing steps or miscalculations, and end with "Final Answer: [number]" (only the number, no extra text).

Question: {question}

Step-by-step reasoning:"""
        
        # 生成响应
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.95,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            response = outputs[0]["generated_text"][len(prompt):].strip()
            # 补充Final Answer格式（若模型遗漏）
            if "Final Answer:" not in response:
                num_match = re.search(r'\d+(?:\.\d+)?', response)
                if num_match:
                    response += f"\nFinal Answer: {num_match.group()}"
            return response
            
        except Exception as e:
            print(f"生成响应时出错: {e}")
            return f"Error: {str(e)}"
    
    def extract_numeric_answer(self, response):
        """从响应中提取数字答案（优先Final Answer）"""
        # 优先提取Final Answer后的数字
        final_answer_pattern = r'Final Answer:\s*(\d+(?:\.\d+)?)'
        match = re.search(final_answer_pattern, response)
        if match:
            return match.group(1)
        
        # 备选：提取所有数字，返回最后一个
        numbers = re.findall(r'\d+(?:\.\d+)?', response)
        if numbers:
            return numbers[-1]
        
        return None
    
    def normalize_answer(self, answer):
        """标准化答案（处理分数、小数等）"""
        if answer is None:
            return None
        
        try:
            # 转换为浮点数再转回整数（如果是整数）
            num = float(answer)
            if num.is_integer():
                return str(int(num))
            return str(num)
        except:
            return answer
    
    def evaluate_dataset(self, dataset_path, max_samples=500, save_results=True):
        """评估GSM8K数据集"""
        # 加载Parquet数据集
        print(f"正在加载数据集: {dataset_path}")
        df = pd.read_parquet(dataset_path)
        
        # 限制样本数量
        if max_samples and len(df) > max_samples:
            df = df.head(max_samples)
        
        results = {
            "total": len(df),
            "correct": 0,
            "incorrect": 0,
            "unknown": 0,
            "accuracy": 0.0,
            "details": []
        }
        
        print(f"开始评估，共{len(df)}条样本...")
        print("提示词已加入4个GSM8K正负样例，强化分步计算和错误规避")
        
        # 逐样本评估
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="评估进度"):
            question = row.get("question", "")
            answer = row.get("answer", "")
            
            # 提取真实答案中的数字
            true_answer = self.extract_numeric_answer(answer)
            
            # 生成CoT响应
            response = self.generate_cot_response(question)
            
            # 提取预测答案
            pred_answer = self.extract_numeric_answer(response)
            
            # 标准化答案
            true_answer_norm = self.normalize_answer(true_answer)
            pred_answer_norm = self.normalize_answer(pred_answer)
            
            # 判断是否正确
            is_correct = None
            if pred_answer_norm is not None and true_answer_norm is not None:
                is_correct = (pred_answer_norm == true_answer_norm)
                if is_correct:
                    results["correct"] += 1
                else:
                    results["incorrect"] += 1
            else:
                results["unknown"] += 1
            
            # 保存详细结果
            results["details"].append({
                "id": idx,
                "question": question,
                "true_answer": true_answer,
                "true_answer_norm": true_answer_norm,
                "pred_answer": pred_answer,
                "pred_answer_norm": pred_answer_norm,
                "is_correct": is_correct,
                "cot_response": response,
                "full_answer": answer
            })
        
        # 计算准确率
        if results["correct"] + results["incorrect"] > 0:
            results["accuracy"] = results["correct"] / (results["correct"] + results["incorrect"]) * 100
        
        # 打印结果摘要
        print("\n" + "="*50)
        print(f"GSM8K评估结果摘要")
        print("="*50)
        print(f"总样本数: {results['total']}")
        print(f"正确数: {results['correct']}")
        print(f"错误数: {results['incorrect']}")
        print(f"未知数: {results['unknown']}")
        print(f"准确率: {results['accuracy']:.2f}%")
        print("="*50)
        
        # 保存结果
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"/root/autodl-tmp/ProtoCoT/baseline/ccot/llama/contrastive_cot_results/gsm8kresults.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n详细结果已保存到: {output_path}")
        
        return results

def main():
    """主函数"""
    # 配置参数
    model_path = "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct"
    dataset_path = "/root/autodl-tmp/ProtoCoT/database/gsm8k/train-00000-of-00001.parquet"
    max_samples = 500
    
    # 创建评估器并运行评估
    evaluator = GSM8KCoTEvaluator(model_path)
    results = evaluator.evaluate_dataset(dataset_path, max_samples=max_samples)

if __name__ == "__main__":
    main()