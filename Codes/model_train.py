from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import evaluate
import numpy as np

model_checkpoint = "facebook/bart-large"
data_files = {
    "train": "train_set2.csv",
    "validation": "val_set2.csv"
}

text_column = "content"
summary_column = "summary"
batch_size = 8
max_input_length = 512
max_target_length = 64
output_dir = "./bart-large-tldr-model"

dataset = load_dataset("csv", data_files=data_files)
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

def preprocess(example):
    inputs = example[text_column]
    targets = example[summary_column]

    if "t5" in model_checkpoint:
        inputs = ["summarize: " + inp for inp in inputs]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        [str(target) for target in targets],
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess, batched=True)
val_dataset = val_dataset.map(preprocess, batched=True)

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    return rouge.compute(predictions=decoded_preds, references=decoded_labels)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    num_train_epochs=5,
    logging_dir="./logs",
    max_grad_norm=1.0,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    fp16=True,
    weight_decay=0.01,
    warmup_steps=100,

)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train(resume_from_checkpoint=False)
trainer.save_model("./bart-large-final")