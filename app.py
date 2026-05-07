from flask import Flask, render_template, request, redirect, session

from flask_sqlalchemy import SQLAlchemy

import bcrypt

from utils.bert_predictor import predict_news_bert
from utils.scraper import extract_news_from_url
from utils.explainer import explain_prediction


app = Flask(__name__)

app.secret_key = "authentinews_secret_key"


# =========================
# DATABASE CONFIGURATION
# =========================
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"

db = SQLAlchemy(app)


# =========================
# USER MODEL
# =========================
class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(100))

    email = db.Column(db.String(100), unique=True)

    password = db.Column(db.String(200))


# =========================
# PREDICTION HISTORY MODEL
# =========================
class Prediction(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    news_text = db.Column(db.Text)

    prediction = db.Column(db.String(50))

    confidence = db.Column(db.String(50))

    user_email = db.Column(db.String(100))


# =========================
# CREATE DATABASE
# =========================
with app.app_context():

    db.create_all()


# =========================
# HOME PAGE
# =========================
@app.route("/")
def home():

    return render_template("index.html")


# =========================
# REGISTER ROUTE
# =========================
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form["name"]

        email = request.form["email"]

        password = request.form["password"]


        # Check existing user
        existing_user = User.query.filter_by(
            email=email
        ).first()


        if existing_user:

            return """
            <h1>Email Already Registered</h1>
            """


        # Hash password
        hashed_password = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt()
        )


        # Create new user
        new_user = User(
            name=name,
            email=email,
            password=hashed_password
        )


        db.session.add(new_user)

        db.session.commit()


        return redirect("/login")


    return render_template("register.html")


# =========================
# LOGIN ROUTE
# =========================
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]

        password = request.form["password"]


        user = User.query.filter_by(
            email=email
        ).first()


        if user and bcrypt.checkpw(
            password.encode("utf-8"),
            user.password
        ):

            # Store session
            session["user"] = user.email

            return redirect("/dashboard")


        else:

            return """
            <h1>Invalid Email or Password</h1>
            """


    return render_template("login.html")


# =========================
# DASHBOARD ROUTE
# =========================
@app.route("/dashboard")
def dashboard():

    if "user" not in session:

        return redirect("/login")


    history = Prediction.query.filter_by(
        user_email=session["user"]
    ).all()


    # Analytics
    total_predictions = len(history)

    real_count = len([
        item for item in history
        if item.prediction == "REAL NEWS"
    ])

    fake_count = len([
        item for item in history
        if item.prediction == "FAKE NEWS"
    ])


    return render_template(
        "dashboard.html",
        history=history,
        user=session["user"],
        total_predictions=total_predictions,
        real_count=real_count,
        fake_count=fake_count
    )

# =========================
# LOGOUT ROUTE
# =========================
@app.route("/logout")
def logout():

    session.pop("user", None)

    return redirect("/login")


# =========================
# PREDICTION ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    news_text = request.form["news"]

    news_url = request.form["url"]


    # Extract article from URL
    if news_url.strip() != "":

        extracted_text = extract_news_from_url(news_url)

        if extracted_text:

            news_text = extracted_text

        else:

            return """
            <h1>Could Not Extract Article</h1>
            """


    # Predict news
    result, confidence = predict_news_bert(news_text)


    # Generate LIME explanation
    explanation = explain_prediction(news_text)


    # Save history
    if "user" in session:

        history = Prediction(

            news_text=news_text[:500],

            prediction=result,

            confidence=str(confidence),

            user_email=session["user"]
        )

        db.session.add(history)

        db.session.commit()


    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence,
        explanation=explanation
    )


# =========================
# RUN FLASK APP
# =========================
if __name__ == "__main__":

    app.run(debug=True)