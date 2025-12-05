import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
from datetime import datetime
import re
import numpy as np

class GSM8KSelfConsistencyEvaluator:
    def __init__(self, model_path, device=None):
        """初始化GSM8K的Self-Consistency评估器"""
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_samples = 5  # Self-Consistency采样数量
        
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
        
        # 创建文本生成pipeline（启用采样）
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        print(f"模型加载完成，使用设备: {self.device}")
        print(f"Self-Consistency采样数量: {self.num_samples}")
    
    def generate_cot_response(self, question, temperature=0.7):
        """生成单条CoT推理路径（针对数学问题）"""
        # 构建简洁的提示词
        prompt = f"{question}\n\nLet's think step by step."
        
        # 生成响应（启用采样以产生多样性）
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=1024,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            response = outputs[0]["generated_text"][len(prompt):].strip()
            return response
            
        except Exception as e:
            print(f"生成响应时出错: {e}")
            return f"Error: {str(e)}"
    
    def extract_numeric_answer(self, response):
        """从响应中提取数字答案"""
        # 尝试匹配常见的答案格式
        patterns = [
            r'Final Answer:\s*(\d+(?:\.\d+)?)',
            r'The answer is\s*(\d+(?:\.\d+)?)',
            r'Answer:\s*(\d+(?:\.\d+)?)',
            r'=\s*(\d+(?:\.\d+)?)\s*$',
            r'\b(\d+(?:\.\d+)?)\s*$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        
        # 提取所有数字，返回最后一个
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
    
    def self_consistency_voting(self, question):
        """通过Self-Consistency生成多条路径并投票"""
        answers = []
        responses = []
        normalized_answers = []
        
        # 生成多条推理路径
        for i in range(self.num_samples):
            # 使用不同的temperature增加多样性
            temperature = 0.7 + (i * 0.1)  # 0.7, 0.8, 0.9, 1.0, 1.1
            response = self.generate_cot_response(question, temperature)
            responses.append(response)
            
            # 提取并标准化答案
            answer = self.extract_numeric_answer(response)
            if answer is not None:
                norm_answer = self.normalize_answer(answer)
                answers.append(answer)
                normalized_answers.append(norm_answer)
        
        # 多数投票
        final_answer = None
        confidence = 0.0
        vote_counts = {}
        
        if normalized_answers:
            # 统计每个答案的票数
            for answer in set(normalized_answers):
                vote_counts[answer] = normalized_answers.count(answer)
            
            # 找到票数最多的答案
            max_votes = max(vote_counts.values())
            final_answer = max(vote_counts, key=vote_counts.get)
            confidence = max_votes / len(normalized_answers)
        
        return {
            "final_answer": final_answer,
            "confidence": confidence,
            "vote_counts": vote_counts,
            "responses": responses,
            "individual_answers": answers,
            "normalized_answers": normalized_answers
        }
    
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
            "avg_confidence": 0.0,
            "details": []
        }
        
        print(f"开始评估，共{len(df)}条样本...")
        
        # 逐样本评估
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="评估进度"):
            question = row.get("question", "")
            answer_text = row.get("answer", "")
            
            # 提取真实答案
            true_answer = self.extract_numeric_answer(answer_text)
            true_answer_norm = self.normalize_answer(true_answer)
            
            # 使用Self-Consistency进行推理
            sc_result = self.self_consistency_voting(question)
            pred_answer = sc_result["final_answer"]
            
            # 判断是否正确
            is_correct = None
            if pred_answer is not None and true_answer_norm is not None:
                is_correct = (pred_answer == true_answer_norm)
                if is_correct:
                    results["correct"] += 1
                else:
                    results["incorrect"] += 1
                results["avg_confidence"] += sc_result["confidence"]
            else:
                results["unknown"] += 1
            
            # 保存详细结果
            results["details"].append({
                "id": idx,
                "question": question,
                "true_answer": true_answer,
                "true_answer_norm": true_answer_norm,
                "pred_answer": pred_answer,
                "is_correct": is_correct,
                "confidence": sc_result["confidence"],
                "vote_counts": sc_result["vote_counts"],
                "individual_responses": sc_result["responses"],
                "individual_answers": sc_result["individual_answers"]
            })
        
        # 计算准确率和平均置信度
        if results["correct"] + results["incorrect"] > 0:
            results["accuracy"] = results["correct"] / (results["correct"] + results["incorrect"]) * 100
            results["avg_confidence"] = results["avg_confidence"] / (results["correct"] + results["incorrect"])
        
        # 打印结果摘要
        print("\n" + "="*60)
        print(f"GSM8K Self-Consistency评估结果摘要 (采样数={self.num_samples})")
        print("="*60)
        print(f"总样本数: {results['total']}")
        print(f"正确数: {results['correct']}")
        print(f"错误数: {results['incorrect']}")
        print(f"未知数: {results['unknown']}")
        print(f"准确率: {results['accuracy']:.2f}%")
        print(f"平均置信度: {results['avg_confidence']:.3f}")
        print("="*60)
        
        # 保存结果
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"/root/autodl-tmp/ProtoCoT/baseline/selfc/llama/self_consistency_gsm8k.json"
            
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
    evaluator = GSM8KSelfConsistencyEvaluator(model_path)
    results = evaluator.evaluate_dataset(dataset_path, max_samples=max_samples)

if __name__ == "__main__":
    main()