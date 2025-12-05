import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
from datetime import datetime
import re

class CommonsenseQACoTEvaluator:
    def __init__(self, model_path, device=None):
        """初始化CommonsenseQA的CoT评估器"""
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
    
    def generate_cot_response(self, question, choices, max_new_tokens=512):
        """生成CoT推理响应（加入CommonsenseQA正负样例）"""
        # 构建问题和选项
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        
        # 4个CommonsenseQA正负样例（2正2反，贴合常识推理场景，明确输出格式）
        commonsenseqa_examples = """
### Commonsense Reasoning Examples (Learn correct logic, avoid mistakes)
Example 1 (Correct Reasoning):
Question: What do people use to drink water?
Options:
A. Fork
B. Cup
C. Pen
D. Chair
E. Book
Reasoning: First, I need to identify the function of each option. A fork is for eating food, a pen is for writing, a chair is for sitting, and a book is for reading. A cup is specifically designed to hold liquids like water, making it the tool for drinking water.
Final Answer: B

Example 2 (Correct Reasoning):
Question: Which animal can fly?
Options:
A. Dog
B. Cat
C. Bird
D. Fish
E. Rabbit
Reasoning: Let's analyze each animal's ability. Dogs, cats, and rabbits are land animals with no wings to fly. Fish live in water and use fins to swim, not fly. Birds have wings and most can fly (e.g., sparrows, eagles), so this is the correct answer.
Final Answer: C

Example 3 (Incorrect vs Correct Reasoning):
Question: Where do fish live?
Options:
A. Desert
B. Ocean
C. Mountain
D. Forest
E. Sky
Incorrect Reasoning: Fish can move freely, so they might live in the forest where there are trees and animals.
Correct Reasoning: Fish have gills that allow them to breathe underwater, so they need an aquatic environment. The desert is dry, mountains are rocky, forests are land-based, and the sky is for flying animals. The ocean is a large body of water, which is the natural habitat for fish.
Final Answer: B

Example 4 (Incorrect vs Correct Reasoning):
Question: What do plants need to grow?
Options:
A. Chocolate
B. Sunlight
C. Television
D. Shoes
E. Toys
Incorrect Reasoning: People like chocolate, so plants might also need chocolate to grow well.
Correct Reasoning: Plants require specific conditions for growth: sunlight (for photosynthesis), water, and nutrients. Chocolate is food for humans, television is an electronic device, shoes and toys are human items—none of these help plants grow. Sunlight is essential for plants.
Final Answer: B
"""
        
        # 构建提示词（融入样例，明确要求）
        prompt = f"""{commonsenseqa_examples}

Now apply the same reasoning method to the following question. Think step by step, analyze the function, characteristic, or relationship of each option, and avoid the wrong logic in the examples. Finally, output your answer in the exact format: Final Answer: [A/B/C/D/E]

Question: {question}

Options:
{choices_text}

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
                option_match = re.search(r'[ABCDE]', response)
                if option_match:
                    response += f"\nFinal Answer: {option_match.group()}"
            return response
            
        except Exception as e:
            print(f"生成响应时出错: {e}")
            return f"Error: {str(e)}"
    
    def extract_answer_choice(self, response):
        """从响应中提取答案选项（A/B/C/D/E）"""
        # 首先查找明确的Final Answer格式
        final_answer_pattern = r'Final Answer:\s*([ABCDE])'
        match = re.search(final_answer_pattern, response)
        if match:
            return match.group(1)
        
        # 备选模式：Answer: X
        answer_pattern = r'Answer:\s*([ABCDE])'
        match = re.search(answer_pattern, response)
        if match:
            return match.group(1)
            
        # 备选模式：直接查找选项
        option_pattern = r'\b([ABCDE])\b'
        matches = re.findall(option_pattern, response)
        if matches:
            return matches[-1]  # 返回最后出现的选项
            
        return None
    
    def evaluate_dataset(self, dataset_path, max_samples=500, save_results=True):
        """评估CommonsenseQA数据集"""
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
        print("提示词已加入4个CommonsenseQA正负样例，强化常识推理逻辑和格式规范")
        
        # 逐样本评估
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="评估进度"):
            question = row.get("question", "")
            choices = row.get("choices", {}).get("text", [])
            answer_key = row.get("answerKey", "")
            
            # 生成CoT响应
            response = self.generate_cot_response(question, choices)
            
            # 提取预测答案
            pred_answer = self.extract_answer_choice(response)
            
            # 判断是否正确
            is_correct = None
            if pred_answer is not None and answer_key:
                is_correct = (pred_answer == answer_key)
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
                "choices": choices,
                "true_answer": answer_key,
                "pred_answer": pred_answer,
                "is_correct": is_correct,
                "cot_response": response
            })
        
        # 计算准确率
        if results["correct"] + results["incorrect"] > 0:
            results["accuracy"] = results["correct"] / (results["correct"] + results["incorrect"]) * 100
        
        # 打印结果摘要
        print("\n" + "="*50)
        print(f"CommonsenseQA评估结果摘要")
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
            output_path = f"/root/autodl-tmp/ProtoCoT/baseline/ccot/llama/contrastive_cot_results/commonsenseqa_ccot.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n详细结果已保存到: {output_path}")
        
        return results

def main():
    """主函数"""
    # 配置参数
    model_path = "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct"
    dataset_path = "/root/autodl-tmp/ProtoCoT/database/commonsenseQA/train-00000-of-00001.parquet"
    max_samples = 500
    
    # 创建评估器并运行评估
    evaluator = CommonsenseQACoTEvaluator(model_path)
    results = evaluator.evaluate_dataset(dataset_path, max_samples=max_samples)

if __name__ == "__main__":
    main()