from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# I have decided this is too expensive in computational cost.

# Load the dataset
dataset = load_dataset('json', data_files='data/cleaned/extracted_email_data.json')

# Define system instruction
system_instruction = """
You are reviewing an email using its subject and body. Based on the text in its subject and body, please categorize this email into one of the following categories:
1. 'Company Business/Strategy'
2. 'Purely Personal'
3. 'Personal but in a professional context'
4. 'Logistic Arrangements'
5. 'Employment arrangements'
6. 'Document editing/checking/collaboration'
7. 'Empty message (due to missing attachment)'
8. 'Empty message'
Please provide only one category (e.g., 'Purely Personal').
"""

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")
model = AutoModelForCausalLM.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")


# Functions to format and tokenize entries
def format_entry(entry):
    formatted_text = f"<system> {system_instruction}\n<user> Subject: {entry['Subject']}, Body: {entry['Body']}\n<assistant> "
    return {"formatted_text": formatted_text}


def tokenize_function(examples):
    return tokenizer(examples["formatted_text"], padding="max_length", truncation=True)


# Apply formatting and tokenization
formatted_dataset = dataset.map(format_entry)
tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)


# Function to predict categories and compare with actual categories
def predict_and_evaluate(entry):
    input_ids = tokenizer(entry["formatted_text"], return_tensors="pt").input_ids
    attention_mask = tokenizer(entry["formatted_text"], return_tensors="pt").attention_mask
    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)
    prediction = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract category from prediction (you might need to adjust this based on your model's output)
    predicted_category = prediction.split('\n')[-1].strip()
    # Compare predicted category with actual category
    actual_category = entry["Category"]
    return predicted_category, actual_category


for entry in formatted_dataset["train"]:  # Adjust this if your dataset has a different split or structure
    predicted_category, actual_category = predict_and_evaluate(entry)
    print(f"Predicted Category: {predicted_category}, Actual Category: {actual_category}")