import pickle
import re


# Load model
model = pickle.load(open("model.pkl", "rb"))

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z ]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text


def predict_news(text):

    text = clean_text(text)

    vector_input = vectorizer.transform([text])

    prediction = model.predict(vector_input)[0]

    probabilities = model.predict_proba(vector_input)[0]

    confidence = round(max(probabilities) * 100, 2)


    if prediction == 1:

        return "REAL NEWS", confidence

    else:

        return "FAKE NEWS", confidence