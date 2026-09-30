"""
app.py
-------
Flask backend for the DuplicateIQ web app.

Loads the trained TF-IDF vectorizer + Logistic Regression model, plus the
same pretrained semantic embeddings used during training, and serves:
  GET  /         -> the frontend page
  POST /predict  -> duplicate-question prediction
"""

import os

import joblib
from flask import Flask, jsonify, render_template, request

from nlp_utils import build_pair_features, get_embedding_model

app = Flask(__name__)

MODEL_PATH = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

model = None
vectorizer = None
embedding_model = None
load_error = None

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    # Loaded once at startup so the first prediction isn't slow.
    embedding_model = get_embedding_model()
else:
    load_error = (
        "Model files not found. Please run 'python train_model.py' first "
        "to train the model and generate model/model.pkl and model/vectorizer.pkl."
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if load_error:
        return jsonify({"error": load_error}), 500

    data = request.get_json(silent=True) or {}
    question1 = (data.get("question1") or "").strip()
    question2 = (data.get("question2") or "").strip()

    if not question1 or not question2:
        return jsonify({"error": "Please enter both questions."}), 400

    # Same feature logic (TF-IDF cosine, semantic similarity, word overlap,
    # length difference) as used in train_model.py — see nlp_utils.py.
    features = build_pair_features(question1, question2, vectorizer, embedding_model)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[int(prediction)])
    semantic_similarity = float(features[0][1])

    response = {
        "is_duplicate": bool(prediction),
        "label": "Duplicate" if prediction == 1 else "Not Duplicate",
        "confidence": round(confidence, 4),
        "similarity": round(semantic_similarity, 4),
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
