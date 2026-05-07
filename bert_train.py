import pandas as pd

from datasets import Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.model_selection import train_test_split


# Load datasets
fake_df = pd.read_csv("dataset/Fake.csv")

true_df = pd.read_csv("dataset/True.csv")


# Labels
fake_df["label"] = 0

true_df["label"] = 1


# Combine title + text
fake_df["content"] = fake_df["title"] + " " + fake_df["text"]

true_df["content"] = true_df["title"] + " " + true_df["text"]


# Merge
data = pd.concat([fake_df, true_df])


# Keep only needed columns
data = data[["content", "label"]]


# Reduce dataset for faster training
data = data.sample(5000, random_state=42)


# Split dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["content"].tolist(),
    data["label"].tolist(),
    test_size=0.2,
    random_state=42
)


# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)


# Tokenize
train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True
)

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True
)


# Create datasets
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "label": train_labels
})

test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "label": test_labels
})


# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)


# Training arguments
training_args = TrainingArguments(

    output_dir="./results",

    num_train_epochs=1,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    warmup_steps=100,

    weight_decay=0.01,

    logging_dir="./logs",

    logging_steps=10
)


# Trainer
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset
)


# Train model
trainer.train()


# Save model
model.save_pretrained("./bert_model")

tokenizer.save_pretrained("./bert_model")


print("\nBERT Model Training Complete")