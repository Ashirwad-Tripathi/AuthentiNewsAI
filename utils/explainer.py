import pickle

from lime.lime_text import LimeTextExplainer


# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# Prediction function
def predict_proba(texts):

    transformed_text = vectorizer.transform(texts)

    return model.predict_proba(transformed_text)


# Generate explanation
def explain_prediction(text):

    try:

        # Handle short text
        if len(text.split()) < 5:

            return [
                ("Text too short for explanation", 0)
            ]


        explainer = LimeTextExplainer(
            class_names=["FAKE", "REAL"]
        )


        explanation = explainer.explain_instance(
            text,
            predict_proba,
            num_features=10
        )


        return explanation.as_list()


    except Exception as e:

        return [
            ("Explanation Error", str(e))
        ]
    