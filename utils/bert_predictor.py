from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

import torch


# Load trained BERT model
model = DistilBertForSequenceClassification.from_pretrained(
    "./bert_model"
)

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "./bert_model"
)


# Prediction function
def predict_news_bert(text):

    # Tokenize text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )


    # Predict
    outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(
        outputs.logits,
        dim=1
    )


    prediction = torch.argmax(probabilities).item()

    confidence = round(
        torch.max(probabilities).item() * 100,
        2
    )


    if prediction == 1:

        return "REAL NEWS", confidence

    else:

        return "FAKE NEWS", confidence