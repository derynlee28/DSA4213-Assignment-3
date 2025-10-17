import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score
import time
import random
import pandas as pd

from utils import compute_metrics, run_experiment

# For reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load tweet eval dataset

dataset = load_dataset("tweet_eval", "irony")
print(dataset)


# First, run the baseline experiments for distilbert and roberta

baseline_results = []
baseline_models = ["distilbert-base-uncased", "roberta-base"]

for model_name in baseline_models:
    for strategy in ["full", "lora"]:
        lora_cfg = None
        if strategy == "lora":
            lora_cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1, bias="none")
        result = run_experiment(model_name, strategy, learning_rate=2e-5, batch_size=16, epochs=3, lora_config=lora_cfg)
        baseline_results.append(result)

# Save baseline results
df_baseline = pd.DataFrame(baseline_results)
df_baseline.to_csv("results/baseline_results.csv", index=False)
print("\nBaseline results saved to results/baseline_results.csv")
print(df_baseline.head())

# Run RobERTA

roberta_results = []

# Baseline
baseline_lr = 2e-5
baseline_full_bs = 16
baseline_lora_bs = 16
baseline_epochs = 3
baseline_lora_dropout = 0.1

# Hyperparameters to tune

learning_rates = [2e-5, 3e-5, 4e-5]
full_batch_sizes = [8, 16, 32]
lora_batch_sizes = [8, 16]
epochs_list = [3, 4]
lora_dropouts = [0.05, 0.1]

# For fine-tuning, vary fine-tuning one factor at a time first
# For full fine-tuning
print("Full fine-tuning")

# 1. Learning rate
for lr in learning_rates:
    print(f"FULL: lr={lr}, bs={baseline_full_bs}, epochs={baseline_epochs}")
    result = run_experiment(
        model_name="roberta-base",
        strategy="full",
        learning_rate=lr,
        batch_size=baseline_full_bs,
        epochs=baseline_epochs,
        lora_config=None
    )
    roberta_results.append(result)

# 2. Batch size (use best LR from previous step manually)
best_lr_full = 4e-5
full_batch_sizes = [8, 32] # Already have for batch size 16

for bs in full_batch_sizes:
    print(f"FULL: lr={best_lr_full}, bs={bs}, epochs={baseline_epochs}")
    result = run_experiment(
        model_name="roberta-base",
        strategy="full",
        learning_rate=best_lr_full,
        batch_size=bs,
        epochs=baseline_epochs,
        lora_config=None
    )
    roberta_results.append(result)

# 3. Epochs (use best LR & batch size)
best_bs_full = 32
result = run_experiment(
        model_name="roberta-base",
        strategy="full",
        learning_rate=best_lr_full,
        batch_size=best_bs_full,
        # Just need for epoch 4
        epochs=4,
        lora_config=None)

roberta_results.append(result)

# Results
df_roberta_results = pd.DataFrame(roberta_results)
df_roberta_results

df_roberta_summary = df_roberta_results[["learning_rate", "batch_size", "epochs", "accuracy", "precision", "recall", "f1_macro", 
                                         "train_time_s", "confusion_matrix"]].sort_values(by="f1_macro", ascending=False).reset_index(drop=True)

df_roberta_summary

df_roberta_summary.to_csv("results/roberta_tuning_results.csv", index=False)
print("\nRobERTA tuning results saved to results/roberta_tuning_results.csv")
print(df_roberta.head())

# Run the best model again

best_model = run_experiment(
        model_name="roberta-base",
        strategy="full",
        learning_rate=4e-4,
        batch_size=32,
        epochs=3,
        lora_config=None)

# Extract the error cases from the best model run

best_model_results = []
best_model_results.append(best_model)
best_model_df = pd.DataFrame(best_model_results)
error_cases = best_model_df.loc[0, 'error_cases']
pd.set_option('display.max_colwidth', None)
error_cases.style.set_properties(**{'text-align': 'left'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])

