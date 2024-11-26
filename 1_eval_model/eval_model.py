import gc
import re
import os
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

class HuggingFaceModel:
    def __init__(self, model_name, device="cuda"):
        print(f"Loading model: {model_name}")
        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Pad token set to: {self.tokenizer.pad_token}")

        # Load model with device_map="auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    def generate(self, prompt, max_new_tokens=200, do_sample=False):
        # Tokenize input and move tensors to device
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample
            )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def extract_answer(self, generated_text, options=["A", "B", "C", "D"]):
        pattern = r"The correct answer is\s*([A-D])"
        match = re.search(pattern, generated_text)
        
        if match:
            option = match.group(1)  # 提取括号内的匹配结果
            #print(f"The extracted answer is: {option}")
        else:
            print("No match found.")

        if option in generated_text:
            return option
        return "Unknown"


def load_data(dataset="openlifescienceai/medmcqa", split="train", num_examples=100):
    if dataset == "openlifescienceai/medmcqa":
        return _load_medmcqa_data(split, num_examples) 
    elif dataset == "qiaojin/PubMedQA":
        return _load_medqa_data(split, num_examples)
    else:
        raise ValueError("This dataset is not available.")
    
    
def _load_medqa_data(split="train", num_examples=100):
        print(f"Loading MedQA dataset ({split} split)...")
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split=split).select(range(num_examples))
        return dataset

def _load_medmcqa_data(split="train", num_examples=100):
        print(f"Loading MedMCQA dataset ({split} split)...")
        dataset = load_dataset("openlifescienceai/medmcqa", split=split).select(range(num_examples))
        return dataset


def preprocess_dataset(dataset):
    # Preprocess dataset for evaluation
    processed = []
    num_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    for item in dataset:
        processed.append({
            "id": item["id"] if item["id"] is not None else "Unknown ID",
            "question": item["question"] if item["question"] is not None else "No question provided",
            "choice_a": "A. " + (item["opa"] if item["opa"] is not None else "No option provided"),
            "choice_b": "B. " + (item["opb"] if item["opb"] is not None else "No option provided"),
            "choice_c": "C. " + (item["opc"] if item["opc"] is not None else "No option provided"),
            "choice_d": "D. " + (item["opd"] if item["opd"] is not None else "No option provided"),
            "context": item["exp"] if item["exp"] is not None else "e",  # Use empty string if evidence is None
            "ref_ans": num_to_letter[item["cop"]] if item["cop"] is not None else "Unknown",
        })
    return processed


def main(args):
    # Configuration
    batch_size = args.batch_size
    num_examples = args.num_examples
    model_name = args.model_name
    split = args.split
    use_context = args.use_context  # Whether to use context in prompts

    # Check GPU availability
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Switching to CPU...")
        device = "cpu"
    else:
        device = "cuda"

    # Prompt template
    if use_context:
        prompt_template = """
You are a medical assistant.
Context: {}
Question: {}
Choice: 
{}
Think about the question step by step, then answer it with one of the choices
Answer: The correct answer is
"""

    else:
        prompt_template = """
You are a medical assistant.
Question: {}
Choice: 
{}
Think about the question step by step, then answer it with one of the choices
Answer: The correct answer is
"""

    # Initialize language model
    lm = HuggingFaceModel(model_name, device=device)

    # Load and preprocess dataset
    dataset = load_data(split=split, num_examples=num_examples)
    dataset_ = preprocess_dataset(dataset)
    dataloader = DataLoader(dataset_, batch_size=batch_size, shuffle=False)

    # Initialize metrics
    correct_answers = 0
    question_counter = 0
    results = []

    # F1-score components
    tp, fp, fn = 0, 0, 0

    # Process dataset
    for batch in tqdm(dataloader, desc="Processing Batches"):
        # 确保 batch 是字典格式
        idx = batch['id']
        questions = batch['question']
        contexts = batch['context'] if use_context else [None] * len(batch['question'])
        ref_answers = batch['ref_ans']

        # 构造选项列表
        choices_list = [
            f"{batch['choice_a'][i]}\n{batch['choice_b'][i]}\n{batch['choice_c'][i]}\n{batch['choice_d'][i]}" 
            for i in range(len(batch['question']))
        ]

        # Build prompts
        if use_context:
            prompts = [
                prompt_template.format(context, question, choices)
                for context, question, choices in zip(contexts, questions, choices_list)
            ]
        else:
            prompts = [
                prompt_template.format(question, choices)
                for question, choices in zip(questions, choices_list)
            ]

        # Generate model responses
        generated_texts = lm.generate(prompts)

        # Evaluate results
        for i, (generated_text, question, context, ref_answer, prompt) in enumerate(
            zip(generated_texts, questions, contexts, ref_answers, prompts)
        ):
            try:
                generated_answer = lm.extract_answer(generated_text)
            except Exception as e:
                print(f"[Error] Failed to extract answer for question {question_counter}: {e}")
                generated_answer = "Unknown"

            question_counter += 1

            # Check if the answer is correct
            is_correct = generated_answer == ref_answer
            if is_correct:
                correct_answers += 1

            # Update TP, FP, FN for metrics calculation
            if generated_answer == ref_answer:
                tp += 1
            elif generated_answer != "Unknown" and generated_answer != ref_answer:
                fp += 1
            elif generated_answer == "Unknown" or generated_answer != ref_answer:
                fn += 1

            # Save result
            results.append({
                "index": idx,
                "prompt": prompt,
                "generated_answer": generated_answer,
                "reference_answer": ref_answer,
                "is_correct": is_correct
            })
            
            if question_counter % 5 == 0:
                print("\n--- Debug Info ---")
                print(f"Prompt: {prompt}")
                print(f"Generated Output: {generated_text}")
                print(f"Extracted Answer: {generated_answer}")
                print(f"Reference Answer: {ref_answer}")
                print("------------------\n")

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()

    # Calculate metrics
    accuracy = correct_answers / len(dataset) 
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    # Print metrics
    print(f"\nAccuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(args.root, exist_ok=True)
    output_filename = os.path.join(
        args.root, f"results_{model_name.replace('/', '_')}_{num_examples}_{split}_use_context_{use_context}_{timestamp}.json"
    )
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--model_name", type=str, default="themanas021/phi-3-medical-instruct-themanas")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--root", type=str, default="./logs")
    parser.add_argument("--use_context", type=lambda x: x.lower() in ["true", "1", "yes"], default="1")
    args = parser.parse_args()

    main(args)