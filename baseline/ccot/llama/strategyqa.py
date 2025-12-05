import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import argparse
from datetime import datetime

class CoTEvaluator:
    def __init__(self, model_path, device=None):
        """Initialize CoT Evaluator"""
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        print(f"Loading model: {model_path}")
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
        
        # Create text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        print(f"Model loaded successfully, using device: {self.device}")
    
    def generate_cot_response(self, question, description=None, max_new_tokens=512):
        """Generate CoT reasoning response (with 4 StrategyQA positive/negative examples in English)"""
        # 4 StrategyQA examples (2 correct reasoning + 2 wrong vs correct contrast)
        strategyqa_examples = """
### Reasoning Examples (Learn correct logic, avoid wrong approaches)
Example 1 (Correct Reasoning - Fact Checking):
Background: Penguins are birds, but most penguins cannot fly; their wings have evolved into flippers for swimming.
Question: Can all birds fly?
Reasoning: First, the definition of birds is warm-blooded, egg-laying vertebrates with feathers and wings—flight is not a necessary characteristic. Second, penguins are classified as birds but cannot fly; they only use flippers for swimming. Therefore, there exist birds that cannot fly, so the statement is false.
Answer: False

Example 2 (Correct Reasoning - Causal Deduction):
Background: Plant photosynthesis requires carbon dioxide, water, and sunlight—all three are indispensable. There is no sunlight at night, so photosynthesis cannot occur.
Question: Do plants perform photosynthesis at night?
Reasoning: Step 1: Clarify the core condition for photosynthesis—sunlight is required to provide energy. Step 2: The key feature of the night environment is the lack of sunlight, which fails to meet the necessary condition for photosynthesis. Step 3: Conclude that plants cannot perform photosynthesis at night.
Answer: False

Example 3 (Incorrect vs Correct Reasoning - Missing Key Conditions):
Background: Humans need oxygen to survive, and oxygen mainly comes from plant photosynthesis.
Question: Can humans survive long-term in an airtight space without plants?
Incorrect Reasoning: Humans need oxygen, and as long as there is oxygen in the space, they can survive, so they can live long-term.
Correct Reasoning: First, human respiration consumes oxygen and releases carbon dioxide. Second, without plants in an airtight space to replenish oxygen through photosynthesis, oxygen will gradually be exhausted and carbon dioxide will accumulate continuously. Finally, humans cannot survive when oxygen is depleted, so they cannot live long-term.
Answer: False

Example 4 (Incorrect vs Correct Reasoning - Overgeneralization):
Background: Most mammals are viviparous, but platypuses and echidnas are oviparous mammals.
Question: Are all mammals viviparous?
Incorrect Reasoning: Common animals like cats, dogs, and cows are viviparous mammals, so all mammals are viviparous.
Correct Reasoning: First, the core characteristics of mammals are warm-blooded and lactating—viviparity is not an absolute standard. Second, there are exceptions (platypuses, echidnas) that lay eggs but are classified as mammals. Therefore, we cannot overgeneralize that "all mammals are viviparous."
Answer: False
"""
        
        # Build CoT prompt (English only)
        if description:
            system_prompt = """You are an assistant skilled in logical reasoning. Refer to the following StrategyQA reasoning examples, first analyze the question and background information, deduce step by step (avoid the wrong approaches in the examples), and finally give a clear answer (True/False)."""
            user_prompt = f"""{strategyqa_examples}

Background Information: {description}

Question: {question}

Please strictly follow these requirements:
1. Refer to the reasoning structure of the examples and first break down the core of the question.
2. Deduce step by step based on background information, avoiding mistakes such as missing key conditions and overgeneralization.
3. Ensure the reasoning process is clear, and finally give a clear conclusion in the format "Answer: True" or "Answer: False" (only these two formats are allowed)."""
        else:
            system_prompt = """You are an assistant skilled in logical reasoning. Refer to the following StrategyQA reasoning examples, first analyze the question, deduce step by step (avoid the wrong approaches in the examples), and finally give a clear answer (True/False)."""
            user_prompt = f"""{strategyqa_examples}

Question: {question}

Please strictly follow these requirements:
1. Refer to the reasoning structure of the examples and first break down the core of the question.
2. Deduce step by step, avoiding mistakes such as missing key conditions and overgeneralization.
3. Ensure the reasoning process is clear, and finally give a clear conclusion in the format "Answer: True" or "Answer: False" (only these two formats are allowed)."""
        
        # Build Llama-style prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate response
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
            print(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def extract_answer(self, response):
        """Extract final answer from response"""
        if "Answer: True" in response:
            return True
        elif "Answer: False" in response:
            return False
        else:
            # Extract from text if no explicit answer marker
            lower_response = response.lower()
            if any(word in lower_response for word in ["true", "yes"]):
                return True
            elif any(word in lower_response for word in ["false", "no"]):
                return False
            else:
                return None
    
    def evaluate_dataset(self, dataset_path, max_samples=500, save_results=True):
        """Evaluate dataset"""
        # Load dataset
        print(f"Loading dataset: {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Limit sample number
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
        
        print(f"Starting evaluation with {len(dataset)} samples...")
        print("Prompt includes 4 StrategyQA positive/negative examples (English) to enhance logical reasoning.")
        
        # Evaluate each sample
        for sample in tqdm(dataset, desc="Evaluation Progress"):
            qid = sample.get("qid", "")
            question = sample.get("question", "")
            description = sample.get("description", "")
            true_answer = sample.get("answer", None)
            
            # Generate CoT response
            response = self.generate_cot_response(question, description)
            
            # Extract predicted answer
            pred_answer = self.extract_answer(response)
            
            # Determine correctness
            is_correct = None
            if pred_answer is not None and true_answer is not None:
                is_correct = (pred_answer == true_answer)
                if is_correct:
                    results["correct"] += 1
                else:
                    results["incorrect"] += 1
            else:
                results["unknown"] += 1
            
            # Save detailed results
            results["details"].append({
                "qid": qid,
                "question": question,
                "description": description,
                "true_answer": true_answer,
                "pred_answer": pred_answer,
                "is_correct": is_correct,
                "cot_response": response
            })
        
        # Calculate accuracy
        if results["correct"] + results["incorrect"] > 0:
            results["accuracy"] = results["correct"] / (results["correct"] + results["incorrect"]) * 100
        
        # Print summary
        print("\n" + "="*50)
        print(f"Evaluation Summary")
        print("="*50)
        print(f"Total Samples: {results['total']}")
        print(f"Correct: {results['correct']}")
        print(f"Incorrect: {results['incorrect']}")
        print(f"Unknown: {results['unknown']}")
        print(f"Accuracy: {results['accuracy']:.2f}%")
        print("="*50)
        
        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"/root/autodl-tmp/ProtoCoT/baseline/ccot/llama/contrastive_cot_results/ccot_evaluation_results.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\nDetailed results saved to: {output_path}")
        
        return results

def main():
    """Main function"""
    # Configuration
    model_path = "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct"
    dataset_path = "/root/autodl-tmp/ProtoCoT/database/StrategyQA/strategyqa_train_filtered.json"
    max_samples = 500
    
    # Create evaluator and run evaluation
    evaluator = CoTEvaluator(model_path)
    results = evaluator.evaluate_dataset(dataset_path, max_samples=max_samples)

if __name__ == "__main__":
    main()