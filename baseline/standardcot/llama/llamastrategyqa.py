import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import argparse
from datetime import datetime

class CoTEvaluator:
    def __init__(self, model_path, device=None):
        """初始化CoT评估器"""
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
    
    def generate_cot_response(self, question, description=None, max_new_tokens=512):
        """生成CoT推理响应"""
        # 构建CoT提示词
        if description:
            system_prompt = """你是一个擅长逻辑推理的助手。请先分析问题，逐步推理，最后给出明确的答案（True/False）。"""
            user_prompt = f"""背景信息：{description}

问题：{question}

请按照以下步骤回答：
1. 分析问题和背景信息
2. 逐步推理
3. 最后用"答案：True"或"答案：False"给出明确结论"""
        else:
            system_prompt = """你是一个擅长逻辑推理的助手。请先分析问题，逐步推理，最后给出明确的答案（True/False）。"""
            user_prompt = f"""问题：{question}

请按照以下步骤回答：
1. 分析问题
2. 逐步推理
3. 最后用"答案：True"或"答案：False"给出明确结论"""
        
        # 构建Llama格式的提示
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 生成响应
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.9,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            response = outputs[0]["generated_text"][len(prompt):].strip()
            return response
            
        except Exception as e:
            print(f"生成响应时出错: {e}")
            return f"Error: {str(e)}"
    
    def extract_answer(self, response):
        """从响应中提取最终答案"""
        if "答案：True" in response:
            return True
        elif "答案：False" in response:
            return False
        else:
            # 如果没有明确的答案标记，尝试从文本中提取
            lower_response = response.lower()
            if any(word in lower_response for word in ["true", "yes", "是", "对"]):
                return True
            elif any(word in lower_response for word in ["false", "no", "否", "错"]):
                return False
            else:
                return None
    
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
            "details": []
        }
        
        print(f"开始评估，共{len(dataset)}条样本...")
        
        # 逐样本评估
        for sample in tqdm(dataset, desc="评估进度"):
            qid = sample.get("qid", "")
            question = sample.get("question", "")
            description = sample.get("description", "")
            true_answer = sample.get("answer", None)
            
            # 生成CoT响应
            response = self.generate_cot_response(question, description)
            
            # 提取预测答案
            pred_answer = self.extract_answer(response)
            
            # 判断是否正确
            is_correct = None
            if pred_answer is not None and true_answer is not None:
                is_correct = (pred_answer == true_answer)
                if is_correct:
                    results["correct"] += 1
                else:
                    results["incorrect"] += 1
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
                "cot_response": response
            })
        
        # 计算准确率
        if results["correct"] + results["incorrect"] > 0:
            results["accuracy"] = results["correct"] / (results["correct"] + results["incorrect"]) * 100
        
        # 打印结果摘要
        print("\n" + "="*50)
        print(f"评估结果摘要")
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
            output_path = f"cot_evaluation_results_{timestamp}.json"
            
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
    evaluator = CoTEvaluator(model_path)
    results = evaluator.evaluate_dataset(dataset_path, max_samples=max_samples)

if __name__ == "__main__":
    main()