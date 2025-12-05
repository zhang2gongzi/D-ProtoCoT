import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
from datetime import datetime
import numpy as np

class SelfConsistencyEvaluator:
    def __init__(self, model_path, device=None):
        """初始化Self-Consistency评估器"""
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
    
    def generate_cot_response(self, question, description=None, temperature=0.7):
        """生成单条CoT推理路径"""
        # 构建提示词
        if description:
            prompt = f"""Background: {description}

Question: {question}

Let's think step by step and give the final answer as True or False."""
        else:
            prompt = f"""Question: {question}

Let's think step by step and give the final answer as True or False."""
        
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
    
    def extract_answer(self, response):
        """从响应中提取布尔答案"""
        lower_response = response.lower()
        
        # 查找明确的True/False
        if "final answer: true" in lower_response:
            return True
        elif "final answer: false" in lower_response:
            return False
        
        # 查找其他变体
        if any(word in lower_response for word in ["true", "yes", "correct", "affirmative"]):
            return True
        elif any(word in lower_response for word in ["false", "no", "incorrect", "negative"]):
            return False
        
        return None
    
    def self_consistency_voting(self, question, description=None):
        """通过Self-Consistency生成多条路径并投票"""
        answers = []
        responses = []
        
        # 生成多条推理路径
        for i in range(self.num_samples):
            # 使用不同的temperature增加多样性
            temperature = 0.7 + (i * 0.1)  # 0.7, 0.8, 0.9, 1.0, 1.1
            response = self.generate_cot_response(question, description, temperature)
            responses.append(response)
            
            # 提取答案
            answer = self.extract_answer(response)
            if answer is not None:
                answers.append(answer)
        
        # 多数投票
        if answers:
            # 统计True和False的数量
            true_count = sum(answers)
            false_count = len(answers) - true_count
            
            final_answer = true_count > false_count
            confidence = max(true_count, false_count) / len(answers)
            
            return {
                "final_answer": final_answer,
                "confidence": confidence,
                "vote_counts": {"True": true_count, "False": false_count},
                "responses": responses,
                "individual_answers": answers
            }
        else:
            return {
                "final_answer": None,
                "confidence": 0.0,
                "vote_counts": {"True": 0, "False": 0},
                "responses": responses,
                "individual_answers": answers
            }
    
    def evaluate_dataset(self, dataset_path, max_samples=500, save_results=True):
        """评估数据集"""
        # 加载数据集
        print(f"正在加载数据集: {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # 限制样本数量
        if max_samples and len(dataset) > max_samples:
            dataset = dataset[:max_samples]
        
        results = {
            "total": len(dataset),
            "correct": 0,
            "incorrect": 0,
            "unknown": 0,
            "accuracy": 0.0,
            "avg_confidence": 0.0,
            "details": []
        }
        
        print(f"开始评估，共{len(dataset)}条样本...")
        
        # 逐样本评估
        for sample in tqdm(dataset, desc="评估进度"):
            qid = sample.get("qid", "")
            question = sample.get("question", "")
            description = sample.get("description", "")
            true_answer = sample.get("answer", None)
            
            # 使用Self-Consistency进行推理
            sc_result = self.self_consistency_voting(question, description)
            pred_answer = sc_result["final_answer"]
            
            # 判断是否正确
            is_correct = None
            if pred_answer is not None and true_answer is not None:
                is_correct = (pred_answer == true_answer)
                if is_correct:
                    results["correct"] += 1
                else:
                    results["incorrect"] += 1
                results["avg_confidence"] += sc_result["confidence"]
            else:
                results["unknown"] += 1
            
            # 保存详细结果
            results["details"].append({
                "qid": qid,
                "question": question,
                "description": description,
                "true_answer": true_answer,
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
        print(f"Self-Consistency评估结果摘要 (采样数={self.num_samples})")
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
            output_path = f"/root/autodl-tmp/ProtoCoT/baseline/selfc/llama/self_consistency_strategyqa_llama.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n详细结果已保存到: {output_path}")
        
        return results

def main():
    """主函数"""
    # 配置参数
    model_path = "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct"
    dataset_path = "/root/autodl-tmp/ProtoCoT/database/StrategyQA/strategyqa_train_filtered.json"
    max_samples = 500
    
    # 创建评估器并运行评估
    evaluator = SelfConsistencyEvaluator(model_path)
    results = evaluator.evaluate_dataset(dataset_path, max_samples=max_samples)

if __name__ == "__main__":
    main()