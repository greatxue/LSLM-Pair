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

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Pad token set to: {self.tokenizer.pad_token}")

        # Load model with device_map="auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    def generate(self, prompt, max_new_tokens=50, do_sample=False):
        # Tokenize input and move tensors to device
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample
            )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def extract_answer(self, generated_text, options=["yes", "no"]):
        # Extract answers from generated text
        for option in options:
            if option in generated_text:
                return option
        return "Unknown"


def load_medqa_data(split="test", num_examples=100):
    print(f"Loading MedQA dataset ({split} split)...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split=split).select(range(num_examples))
    return dataset


def preprocess_dataset(dataset):
    # Preprocess dataset for evaluation
    processed = []
    for item in dataset:
        processed.append({
            "pubid": item["pubid"],
            "question": item["question"],
            "context": " ".join(item["context"]["contexts"]),  # Combine context texts
            "long_answer": item["long_answer"],
            "final_decision": item["final_decision"]
        })
    return processed


def clean_pubid(pubid):
    # Clean or process pubid for compatibility
    if isinstance(pubid, torch.Tensor):
        if pubid.numel() == 1:  # Single element Tensor
            return int(pubid.item())
        else:  # Multiple elements Tensor
            return [int(x) for x in pubid.tolist()]
    elif isinstance(pubid, str):
        return pubid.replace(",", "")
    else:
        return pubid


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
        You are a medical assistant. Think step by step, and answer the following question with yes or no:
        Context: {}
        Question: {}
        Answer:
        """
    else:
        prompt_template = """
        You are a medical assistant. Think step by step, and answer the following question with yes or no:
        Question: {}
        Answer:
        """

    # Initialize language model
    lm = HuggingFaceModel(model_name, device=device)

    # Load and preprocess dataset
    dataset = load_medqa_data(split=split, num_examples=num_examples)
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
        idx = batch["pubid"]
        questions = batch["question"]
        contexts = batch["context"] if use_context else [None] * len(batch["question"])
        answers = batch["final_decision"]

        # Build prompts
        if use_context:
            prompts = [prompt_template.format(context, question) for context, question in zip(contexts, questions)]
        else:
            prompts = [prompt_template.format(question) for question in questions]

        # Generate model responses
        generated_texts = lm.generate(prompts)

        # Evaluate results
        for i, (generated_text, question, context, reference_answer) in enumerate(
            zip(generated_texts, questions, contexts, answers)
        ):
            try:
                generated_answer = lm.extract_answer(generated_text, options=["yes", "no"])
            except Exception as e:
                print(f"[Error] Failed to extract answer for question {question_counter}: {e}")
                generated_answer = "Unknown"

            question_counter += 1
            is_correct = generated_answer == reference_answer
            if is_correct:
                correct_answers += 1

            # Update F1 components
            if reference_answer == "yes":
                if generated_answer == "yes":
                    tp += 1
                elif generated_answer == "no":
                    fn += 1
            elif reference_answer == "no" and generated_answer == "yes":
                fp += 1

            # Save result
            results.append({
                "index": clean_pubid(idx),
                "question": question,
                "context": context,
                "generated_answer": generated_answer,
                "reference_answer": reference_answer,
                "is_correct": is_correct
            })

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()

    # Calculate metrics
    accuracy = correct_answers / len(dataset) * 100
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    # Print metrics
    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")

    # Save results
    os.makedirs(args.root, exist_ok=True)
    output_filename = os.path.join(
        args.root, f"results_{model_name.replace('/', '_')}_{num_examples}_{split}_use_context_{use_context}.json"
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
    parser.add_argument("--use_context", type=lambda x: x.lower() in ["true", "1", "yes"], default="true")
    args = parser.parse_args()

    main(args)