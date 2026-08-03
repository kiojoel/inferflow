from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np

model_path = "./models/ag_news_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

dataset = load_dataset("fancyzhx/ag_news")
eval_dataset = dataset["test"].select(range(500))

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
tokenized_eval = tokenized_eval.remove_columns(["text"]).rename_column("label", "labels")
tokenized_eval.set_format("torch")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return {"accuracy": (predictions == eval_pred.label_ids).astype(np.float32).mean().item()}

trainer = Trainer(model=model, args=TrainingArguments(output_dir="./tmp_eval", report_to="none"), compute_metrics=compute_metrics)
results = trainer.evaluate(eval_dataset=tokenized_eval)
print(results)