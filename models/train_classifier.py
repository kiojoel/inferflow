import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

dataset = load_dataset("fancyzhx/ag_news")

train_dataset = dataset["train"].select(range(2000))
eval_dataset = dataset["test"].select(range(500))

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_eval = tokenized_eval.remove_columns(["text"])

tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_eval = tokenized_eval.rename_column("label", "labels")

tokenized_train.set_format("torch")
tokenized_eval.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4  # AG_NEWS has 4 classes
)

if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("GPU not available.")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return {"accuracy": (predictions == eval_pred.label_ids).astype(np.float32).mean().item()}

training_args = TrainingArguments(
    output_dir="./results",          
    eval_strategy="epoch",     
    save_strategy="epoch",           
    num_train_epochs=1,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,
    logging_steps=10,               
    report_to="none",
    load_best_model_at_end=True,     
    save_total_limit=1,              
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

print("Evaluating...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# save model
print("Saving model to ./models/ag_news_classifier ...")
trainer.save_model("./models/ag_news_classifier")
tokenizer.save_pretrained("./models/ag_news_classifier") 

print("Done!")