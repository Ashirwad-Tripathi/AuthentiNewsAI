import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Clean text
def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z ]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text


# Load datasets
fake_df = pd.read_csv("dataset/Fake.csv")

true_df = pd.read_csv("dataset/True.csv")


# Labels
fake_df["label"] = 0

true_df["label"] = 1


# Combine title + text
fake_df["content"] = fake_df["title"] + " " + fake_df["text"]

true_df["content"] = true_df["title"] + " " + true_df["text"]


# Merge datasets
data = pd.concat([fake_df, true_df])


# Shuffle
data = data.sample(frac=1, random_state=42)


# Clean content
data["content"] = data["content"].apply(clean_text)


# Features and labels
X = data["content"]

y = data["label"]


# Train test split
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

xv_train = vectorizer.fit_transform(x_train)

xv_test = vectorizer.transform(x_test)


# Model
model = LogisticRegression(max_iter=2000)

model.fit(xv_train, y_train)


# Accuracy
predictions = model.predict(xv_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)


# Save files
pickle.dump(model, open("model.pkl", "wb"))

pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))


print("Training Complete")