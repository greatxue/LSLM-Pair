import gc
import re
import os
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM



class HuggingFaceModel:
    def __init__(self, model_name, device="cuda"):
        print(f"Loading model: {model_name}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Ensure padding token is set
        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                          device_map="auto").to(self.device)

    def generate(self, prompt, max_new_tokens=50, do_sample=False):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample
            )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def extract_answer(self, generated_text, options=["yes", "no"]):
        for option in options:
            if option in generated_text:
                return option
        return "Unknown"

def load_medqa_data(split="test", num_examples=100):

    print(f"Loading MedQA dataset ({split} split)...")
    print(10 * "=")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", 
                           split=split).select(range(num_examples))
    return dataset

def main(args):
    # Configuration
    batch_size = args.batch_size
    num_examples = args.num_examples
    model_name = args.model_name
    split = args.split

    # Judgment-style prompt template
    prompt_template = """
    You are a medical assistant. Think step by step, and answer the following question with yes or no:
    Question: {}
    Answer:
    """

    # Initialize language model
    lm = HuggingFaceModel(model_name)

    # Load dataset
    dataset = load_medqa_data(split=split, num_examples=num_examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize metrics
    correct_answers = 0
    question_counter = 0
    results = [] 

    # Process dataset in batches
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing Batches")):
        idx = batch["pubid"]
        questions = batch["question"]
        answers = batch["final_decision"]  # Expected answers should be "yes" or "no"
        prompts = [prompt_template.format(question) for question in questions]

        # Generate outputs
        generated_texts = lm.generate(prompts)

        # Evaluate results
        for i, (generated_text, question, reference_answer) in enumerate(
            zip(generated_texts, questions, answers)
        ):
            try:
                generated_answer = lm.extract_answer(generated_text, options=["yes", "no"])
            except Exception as e:
                print(f"[Error] Failed to extract answer for question {question_counter}: {e}")
                generated_answer = "Unknown"

            question_counter += 1

            # Compare answers
            is_correct = generated_answer == reference_answer
            if is_correct:
                correct_answers += 1
                
            results.append({
                "index": idx,
                "question": question,
                "generated_answer": generated_answer,
                "reference_answer": reference_answer,
                "is_correct": is_correct
            })

            # Log every 8 questions
            if question_counter % 8 == 0:
                print(f"\n[Question {question_counter}]")
                print(f"Question: {question}")
                print(f"Generated Answer: {generated_answer}")
                print(f"Reference Answer: {reference_answer}")
                print("Correct!" if generated_answer == reference_answer else "Incorrect!")
                print("-" * 60)

        # Clear memory to prevent GPU OOM
        gc.collect()
        torch.cuda.empty_cache()

    # Calculate and display accuracy
    accuracy = correct_answers / len(dataset) * 100
    print(f"\nAccuracy on the {split} set: {accuracy:.2f}%")
    
    # Save results to JSON file
    os.makedirs(args.root, exist_ok=True)
    output_filename = f"results_{model_name}_{num_examples}_{split}.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_filename}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--root", type=str, default="/home/wenhao/Project/greatxue/LSLM_Pair/logs")
    args = parser.parse_args()

    # Run main logic
    main(args)