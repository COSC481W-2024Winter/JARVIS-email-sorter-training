import logging
import os
import sys
from datetime import datetime

import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# start logging

log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")

logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


# Redirect stdout and stderr to log file
stdout_logger = logging.getLogger('STDOUT')
sl = StreamToLogger(stdout_logger, logging.INFO)
sys.stdout = sl

stderr_logger = logging.getLogger('STDERR')
sl = StreamToLogger(stderr_logger, logging.ERROR)
sys.stderr = sl

# Define categories and their mapping to numerical labels
categories = [
    'Company Business/Strategy', 'Purely Personal', 'Personal but in a professional context',
    'Logistic Arrangements', 'Employment arrangements', 'Document editing/checking/collaboration',
    'Empty message (due to missing attachment)', 'Empty message'
]
category2id = {category: i for i, category in enumerate(categories)}
id2label = {i: label for i, label in enumerate(categories)}


def preprocess_function(examples):
    texts = [f"{subj} [SEP] {body}" for subj, body in zip(examples['Subject'], examples['Body'])]
    labels = [category2id[cat] for cat in examples['Category']]
    tokenized_inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=512)
    # Ensure this function returns a dictionary with tokenized inputs and labels
    return {**tokenized_inputs, 'labels': labels}


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

train_dataset = load_dataset('json', data_files='data/cleaned/extracted_email_data_1300.json', split='train')
test_dataset = load_dataset('json', data_files='data/cleaned/extracted_email_data_-100.json', split='train')

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(categories))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


def display_predictions(eval_pred, id2label):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    for pred, actual in zip(predictions, labels):
        pred_label = id2label[pred]
        actual_label = id2label[actual]
        print(f"Predicted: '{pred_label}', Actual: '{actual_label}'")


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
results = trainer.evaluate()
print(results)

# After training and evaluation
predictions = trainer.predict(tokenized_test_dataset)
# Convert logits to predicted labels
predicted_labels = np.argmax(predictions.predictions, axis=-1)

# Display predictions and actual labels
for pred, actual in zip(predicted_labels, predictions.label_ids):
    print(f"Predicted: '{id2label[pred]}', Actual: '{id2label[actual]}'")

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_directory = f"./model_{current_time}"
os.makedirs(save_directory, exist_ok=True)
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
print(f"Model and tokenizer saved in {save_directory}")
