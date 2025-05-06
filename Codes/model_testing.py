import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from tqdm import tqdm


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model


def generate_summaries(posts, tokenizer, model, model_type="t5",
                       max_input_length=512, max_target_length=100):
    model.eval()
    generated = []

    for post in tqdm(posts, desc="Generating Summaries"):
        input_text = f"summarize: {post}" if model_type.lower().startswith("t5") else post

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding="max_length"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()} 

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_target_length,
                num_beams=4,
                early_stopping=True
            )

        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated.append(summary)

    return generated


def evaluate_rouge(preds, refs):
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=preds, references=refs)
    for key, value in results.items():
        print(f"{key}: {value:.4f}")


model_path = "./bart-large-final"     
model_type = "bart"                    
input_csv = "test_data.csv"


df = pd.read_csv(input_csv)
posts = df["clean_content"].astype(str).tolist()
references = df["summary"].astype(str).tolist()

tokenizer, model = load_model(model_path)

device = torch.device("cuda")  
model = model.to(device)

generated = generate_summaries(posts, tokenizer, model, model_type=model_type)

df["generated_summary"] = generated
df.to_csv("test_generated_summaries_bartlarge.csv", index=False)

print("\n--- ROUGE Evaluation ---")
evaluate_rouge(generated, references)