# pip install pandas transformers datasets

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import json
from transformers import pipeline

# Load the CSV
data = pd.read_csv("poiTrainingData.csv")

# Process the CSV into prompt-response format for fine-tuning
# Example: {"prompt": "User is interested in glaciers near California.", "completion": "Mount Fiske Glacier"}
def preprocess_data(row):
    prompt = f"Recommend a place for someone interested in {', '.join(row['categories'].split(';'))} near a location at latitude {row['latitude_radian']} and longitude {row['longitude_radian']}."
    completion = row['name']
    return {"prompt": prompt, "completion": completion}

# Apply preprocessing
training_data = data.apply(preprocess_data, axis=1).tolist()


# Save as JSONL
with open("training_data.jsonl", "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\n")


# Load a pre-trained model and tokenizer
model_name = "gpt2"  # Replace with a preferred model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load your training data
dataset = load_dataset("json", data_files="training_data.jsonl")

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["prompt"] + tokenizer.eos_token + examples["completion"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Fine-tuning settings
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")


# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

# Create a pipeline for inference
nlp = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Query the model
location = {"latitude": 0.648, "longitude": -2.071}  # Example user location
interest = "glaciers"
prompt = f"Recommend a place for someone interested in {interest} near a location at latitude {location['latitude']} and longitude {location['longitude']}."

response = nlp(prompt, max_length=100)
print(response[0]["generated_text"])
