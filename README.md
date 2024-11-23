## Experimental Plan for Enhancing Large Model Performance Using Evidence Generation

### TODO:

+ [X] 1=Download Qwen 1.5B/7B, manage inferences
+ [ ] 2=Test the Qwen 7B with MedQA (with GT), record the initial and improved performance
+ [ ] 3=Finetune the SLM, MLP-head-1-PRE

### STAT:

|               | Qwen 1.5B | Qwen 7B | Phi 3 |
| ------------- | --------- | ------- | ----- |
| acc/MedQA:100 |           |         |       |

### 1. Define the Models, Dataset, and Scoring Criteria

- **Small Model**:
  - Qwen 1.5B, used for evidence generation.
  - Also try MedLLM `themanas021/phi-3-medical-instruct-themanas.`
- **Large Model**: Qwen 7B, used for final task execution.
- **Dataset**:
  - Domain-specific knowledge dataset where the large model shows poor performance.
  - MedQA: https://huggingface.co/datasets/qiaojin/PubMedQA/viewer/pqa_artificial
- **Scoring Metric**: Assign a score to the generated evidence based on the improvement in large model performance:
  - The improvement score is determined by the change in performance metrics (e.g., Accuracy/F1 Score).
  - Optionally, the score can be adjusted based on evidence length, content coverage, and actual contribution.

### 2. Evaluate Baseline Performance, Pre-train MLP Head 1, and Fine-tune the Small Model

- **Baseline 1**: Evaluate the large model (Qwen 7B) on the original dataset and record its initial performance.
- **Baseline 2**: Evaluate the large model using ground truth (GT) facts as evidence and record the performance. Calculate the improvement score.
- Analyze the error types of the large model, identifying categories where it performs poorly (e.g., lack of domain knowledge, insufficient reasoning capabilities).
- Fine-tune the small model using `(question + instruction, GT fact)` pairs:
  - **Loss 1**: For evidence generation using the GT fact.
  - **Loss 2**: Pre-train MLP Head 1 (MLP head 1 PRE) to output the improvement score.
  - Jointly optimize both losses during training.

### 3. Generate Evidence and Enhance Large Model Performance with Prompting

- Use the fine-tuned small model to generate evidence for each test sample.
- Prompt the large model with the generated evidence and record its performance improvement. Assign an improvement score based on the observed change.

### 4. Train Two MLP Heads

- **Continue Training MLP Head 1**:

  - **Input**: Extract hidden state from the final layer of the small model (representing the evidence generated).
  - **Output**: Evidence score prediction.
  - **Loss Function**: Mean Squared Error (MSE Loss).
- **Train MLP Head 2**:

  - **Input**: Original question and the response from the large model.
  - **Output**: A binary label indicating whether the large model's response is correct.
  - **Loss Function**: Cross-Entropy Loss.

### 5. Iterative Evaluation and Enhancement of the Major Model

- During inference, generate multiple pieces of evidence using the small model.
- Rank the evidence using scores from MLP Head 1, and select high-quality evidence to prompt the large model.
- Pass the large model’s response through MLP Head 2 for quality assessment:
  - If the response quality is below a certain threshold, retry with different evidence.
- Iterate through this process, dynamically refining the selection of evidence and improving the overall response quality of the large model.
