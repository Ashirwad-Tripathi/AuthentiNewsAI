import pickle

from lime.lime_text import LimeTextExplainer


# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# Prediction probabilities
def predict_proba(texts):

    transformed_text = vectorizer.transform(texts)

    return model.predict_proba(transformed_text)


# Main explanation function
def explain_prediction(text, prediction):

    # =========================
    # AI SUMMARY
    # =========================

    if prediction == "FAKE NEWS":

        summary = """
        The AI model detected language patterns commonly associated
        with misleading or sensational news content. Certain phrases
        in the article resemble structures frequently found in
        misinformation and viral fake news posts.
        """

    else:

        summary = """
        The article contains structured and balanced reporting
        patterns commonly found in legitimate news sources.
        The wording and sentence structure align with factual
        journalistic content.
        """


    # =========================
    # LIME EXPLANATION
    # =========================

    explainer = LimeTextExplainer(
        class_names=["FAKE", "REAL"]
    )

    explanation = explainer.explain_instance(
        text,
        predict_proba,
        num_features=6
    )

    influence_words = []

    for word, score in explanation.as_list():

        if score > 0:

            impact = "Supports REAL news"

        else:

            impact = "Supports FAKE news"

        influence_words.append({

            "word": word,

            "impact": impact

        })

    return {

        "summary": summary,

        "influence_words": influence_words
    }