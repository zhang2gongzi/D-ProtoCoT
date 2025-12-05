import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
from datetime import datetime
import re
import numpy as np
from collections import Counter

class CommonsenseQASelfConsistencyEvaluator:
    def __init__(self, model_path, device=None):
        """初始化CommonsenseQA的Self-Consistency评估器"""
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
    
    def generate_cot_response(self, question, choices, temperature=0.7):
        """生成单条CoT推理路径"""
        # 构建问题和选项
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        
        # 使用简洁的提示词
        prompt = f"""{question}

Options:
{choices_text}

Let's think step by step."""
        
        # 生成响应（启用采样以产生多样性）
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=512,
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
    
    def extract_answer_choice(self, response):
        """从响应中提取答案选项（A/B/C/D/E）"""
        # 查找明确的答案格式
        patterns = [
            r'Final Answer:\s*([ABCDE])',
            r'Answer:\s*([ABCDE])',
            r'The answer is\s*([ABCDE])',
            r'\b([ABCDE])\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                return matches[-1]  # 返回最后出现的选项
        
        return None
    
    def convert_to_serializable(self, obj):
        """将对象转换为JSON可序列化的类型"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_to_serializable(item) for item in obj)
        else:
            return obj
    
    def self_consistency_voting(self, question, choices):
        """通过Self-Consistency生成多条路径并投票"""
        answers = []
        responses = []
        
        # 生成多条推理路径
        for i in range(self.num_samples):
            # 使用不同的temperature增加多样性
            temperature = 0.7 + (i * 0.1)  # 0.7, 0.8, 0.9, 1.0, 1.1
            response = self.generate_cot_response(question, choices, temperature)
            responses.append(response)
            
            # 提取答案
            answer = self.extract_answer_choice(response)
            if answer and answer in ['A', 'B', 'C', 'D', 'E']:
                answers.append(answer)
        
        # 多数投票
        final_answer = None
        confidence = 0.0
        vote_counts = {}
        
        if answers:
            # 统计每个选项的票数
            vote_counter = Counter(answers)
            vote_counts = dict(vote_counter)
            
            # 找到票数最多的选项
            max_votes = max(vote_counter.values())
            final_answer = max(vote_counter, key=vote_counter.get)
            confidence = max_votes / len(answers)
        
        return {
            "final_answer": final_answer,
            "confidence": confidence,
            "vote_counts": vote_counts,
            "responses": responses,
            "individual_answers": answers
        }
    
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
            "avg_confidence": 0.0,
            "details": []
        }
        
        print(f"开始评估，共{len(df)}条样本...")
        
        # 逐样本评估
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="评估进度"):
            question = row.get("question", "")
            
            # 安全地获取choices数据
            choices_data = row.get("choices", {})
            if isinstance(choices_data, dict):
                choices = choices_data.get("text", [])
            elif isinstance(choices_data, (list, np.ndarray)):
                choices = list(choices_data)
            else:
                choices = []
            
            # 确保choices是普通Python列表
            choices = self.convert_to_serializable(choices)
            answer_key = row.get("answerKey", "")
            
            # 使用Self-Consistency进行推理
            sc_result = self.self_consistency_voting(question, choices)
            pred_answer = sc_result["final_answer"]
            
            # 判断是否正确
            is_correct = None
            if pred_answer is not None and answer_key:
                is_correct = (pred_answer == answer_key)
                if is_correct:
                    results["correct"] += 1
                else:
                    results["incorrect"] += 1
                results["avg_confidence"] += sc_result["confidence"]
            else:
                results["unknown"] += 1
            
            # 保存详细结果
            sample_result = {
                "id": int(idx),
                "question": str(question),
                "choices": choices,
                "true_answer": str(answer_key),
                "pred_answer": str(pred_answer) if pred_answer is not None else None,
                "is_correct": bool(is_correct) if is_correct is not None else None,
                "confidence": float(sc_result["confidence"]),
                "vote_counts": sc_result["vote_counts"],
                "individual_responses": sc_result["responses"],
                "individual_answers": sc_result["individual_answers"]
            }
            
            results["details"].append(sample_result)
        
        # 计算准确率和平均置信度
        if results["correct"] + results["incorrect"] > 0:
            results["accuracy"] = float(results["correct"] / (results["correct"] + results["incorrect"]) * 100)
            results["avg_confidence"] = float(results["avg_confidence"] / (results["correct"] + results["incorrect"]))
        
        # 打印结果摘要
        print("\n" + "="*60)
        print(f"CommonsenseQA Self-Consistency评估结果摘要 (采样数={self.num_samples})")
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
            # 转换所有数据为可序列化格式
            serializable_results = self.convert_to_serializable(results)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"/root/autodl-tmp/ProtoCoT/baseline/selfc/llama/self_consistency_commonsenseqa_results_llama.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            print(f"\n详细结果已保存到: {output_path}")
        
        return results

def main():
    """主函数"""
    # 配置参数
    model_path = "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct"
    dataset_path = "/root/autodl-tmp/ProtoCoT/database/commonsenseQA/train-00000-of-00001.parquet"
    max_samples = 500
    
    # 创建评估器并运行评估
    evaluator = CommonsenseQASelfConsistencyEvaluator(model_path)
    results = evaluator.evaluate_dataset(dataset_path, max_samples=max_samples)

if __name__ == "__main__":
    main()