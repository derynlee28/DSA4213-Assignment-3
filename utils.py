# Define metrics for evaluation

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    probs = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=-1)[:, 1].numpy()

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "roc_auc": roc_auc_score(labels, probs)
    }


# Experiment setup

def run_experiment(model_name, strategy, learning_rate=2e-5, batch_size=16, epochs=3, lora_config=None):
    print(f"\nTraining {model_name} | {strategy} | lr={learning_rate} | batch_size={batch_size} | epochs={epochs}")

    # Initialize the tokenizer for this model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize dataset and convert to pytorch
    tokenized_data = dataset.map(
        lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=128),
        batched=True
    )
    tokenized_data = tokenized_data.rename_column('label', 'labels')
    tokenized_data.set_format('torch')

    # Load models
    if strategy == "full":
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    elif strategy == "lora":
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        # Have to determine target modules for LoRA
        if model_name.startswith("distilbert"):
            target_modules = ["q_lin", "v_lin"]
        elif model_name.startswith("roberta"):
            target_modules = ["query", "value"]
        else:
            target_modules = None

        # LoRA configurations (create default ones if theres no lora_config)
        if lora_config is None:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                target_modules=target_modules)
        else:
            # If target_modules not set in the provided config, fill it
            if getattr(lora_config, "target_modules", None) is None:
                lora_config.target_modules = target_modules

        model = get_peft_model(base_model, lora_config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params:,}/{total_params:,}")

    # Move model to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{model_name}_{strategy}_lr{learning_rate}_bs{batch_size}_ep{epochs}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=100,
        warmup_steps=100,
        report_to="none",
        dataloader_num_workers=0)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['validation'],
        compute_metrics=compute_metrics)

    # Train & evaluate + track training time
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time

    eval_results = trainer.evaluate(tokenized_data['test'])
    preds = trainer.predict(tokenized_data['test'])
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids
    cm = confusion_matrix(y_true, y_pred)

    return {
        "model": model_name,
        "strategy": strategy,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "train_time_s": train_time,
        "accuracy": eval_results.get('eval_accuracy'),
        "precision": eval_results.get('eval_precision'),
        "recall": eval_results.get('eval_recall'),
        "f1": eval_results.get('eval_f1'),
        "f1_macro": eval_results.get('eval_f1_macro'),
        "roc_auc": eval_results.get('eval_roc_auc'),
        "confusion_matrix": cm
    }
