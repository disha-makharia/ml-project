"""
app.py
------
Flask backend for DuplicateIQ.

Uses:
    - TF-IDF cosine similarity
    - GloVe semantic similarity
    - Word overlap
    - Length difference
    - Logistic Regression

A hybrid score is used for the final duplicate decision.
"""

import os
import joblib

from flask import Flask, jsonify, render_template, request

from nlp_utils import (
    build_pair_features,
    get_embedding_model,
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "vectorizer.pkl"
)


# ============================================================
# MODEL VARIABLES
# ============================================================

model = None
vectorizer = None
embedding_model = None
load_error = None


# ============================================================
# LOAD MODEL
# ============================================================

try:

    # Check model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # Check vectorizer
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Vectorizer not found: {VECTORIZER_PATH}"
        )

    print("Loading trained Logistic Regression model...")
    model = joblib.load(MODEL_PATH)

    print("Loading TF-IDF vectorizer...")
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("Loading GloVe embedding model...")
    embedding_model = get_embedding_model()

    print("All models loaded successfully.")

except Exception as error:

    load_error = str(error)

    print("\nMODEL LOADING ERROR:")
    print(load_error)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


# ============================================================
# PREDICT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------------------------
    # Check whether model loaded correctly
    # --------------------------------------------------------

    if load_error:

        return jsonify({
            "error": load_error
        }), 500


    # --------------------------------------------------------
    # Get JSON data
    # --------------------------------------------------------

    data = request.get_json(
        silent=True
    ) or {}


    question1 = (
        data.get("question1") or ""
    ).strip()

    question2 = (
        data.get("question2") or ""
    ).strip()


    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not question1 or not question2:

        return jsonify({
            "error": "Please enter both questions."
        }), 400


    # ========================================================
    # BUILD FEATURES
    # ========================================================

    try:

        features = build_pair_features(
            question1,
            question2,
            vectorizer,
            embedding_model
        )

    except Exception as error:

        return jsonify({
            "error":
                f"Feature generation failed: {error}"
        }), 500


    # ========================================================
    # EXTRACT FEATURES
    # ========================================================

    # Feature 1:
    # TF-IDF cosine similarity

    tfidf_similarity = float(
        features[0][0]
    )


    # Feature 2:
    # GloVe semantic similarity

    semantic_similarity = float(
        features[0][1]
    )


    # Feature 3:
    # Jaccard word overlap

    word_overlap = float(
        features[0][2]
    )


    # Feature 4:
    # Difference in question length
    #
    # This is a difference, not a percentage.

    length_difference = float(
        features[0][3]
    )


    # ========================================================
    # LOGISTIC REGRESSION PREDICTION
    # ========================================================

    try:

        # Model prediction
        ml_prediction = int(
            model.predict(features)[0]
        )


        # Probability for each class
        probabilities = (
            model.predict_proba(features)[0]
        )


        # Classes normally are [0, 1]
        classes = model.classes_


        # ----------------------------------------------------
        # Get probability of DUPLICATE = class 1
        # ----------------------------------------------------

        duplicate_probability = 0.0


        for class_value, probability in zip(
            classes,
            probabilities
        ):

            if int(class_value) == 1:

                duplicate_probability = float(
                    probability
                )

                break


        # ----------------------------------------------------
        # Confidence of ML prediction
        # ----------------------------------------------------

        ml_confidence = float(
            probabilities[
                list(classes).index(
                    ml_prediction
                )
            ]
        )


    except Exception as error:

        return jsonify({
            "error":
                f"Prediction failed: {error}"
        }), 500


    # ========================================================
    # HYBRID SIMILARITY SCORE
    # ========================================================
    #
    # The final similarity score combines:
    #
    # 50% semantic similarity
    # 30% TF-IDF similarity
    # 20% word overlap
    #
    # This helps detect paraphrased questions that may use
    # different wording but have the same meaning.
    #
    # Example:
    #
    # "How do I install Python?"
    #
    # "What are the steps to install Python?"
    #
    # ========================================================

    hybrid_score = (
        (0.50 * semantic_similarity)
        +
        (0.30 * tfidf_similarity)
        +
        (0.20 * word_overlap)
    )


    # Keep score between 0 and 1

    hybrid_score = min(
        1.0,
        max(
            0.0,
            hybrid_score
        )
    )


    # ========================================================
    # FINAL DUPLICATE DECISION
    # ========================================================
    #
    # First condition:
    # If overall hybrid similarity is 50% or more,
    # classify as duplicate.
    #
    # Second condition:
    # Very high semantic similarity combined with
    # reasonable TF-IDF similarity also indicates duplicate.
    #
    # Otherwise use the Logistic Regression prediction.
    #
    # ========================================================

    if hybrid_score >= 0.50:

        final_prediction = 1


    elif (
        semantic_similarity >= 0.85
        and tfidf_similarity >= 0.20
    ):

        final_prediction = 1


    else:

        final_prediction = ml_prediction


    # ========================================================
    # FINAL CONFIDENCE
    # ========================================================

    if final_prediction == 1:

        confidence = max(
            duplicate_probability,
            hybrid_score
        )

    else:

        confidence = max(
            1.0 - duplicate_probability,
            1.0 - hybrid_score
        )


    # Keep confidence between 0 and 1

    confidence = min(
        1.0,
        max(
            0.0,
            confidence
        )
    )


    # ========================================================
    # TERMINAL DEBUG INFORMATION
    # ========================================================

    print("\n======================================")
    print("             DUPLICATEIQ")
    print("======================================")

    print("\nQuestion 1:")
    print(question1)

    print("\nQuestion 2:")
    print(question2)

    print("\nFEATURES")
    print("--------------------------------------")

    print(
        f"TF-IDF Similarity   : "
        f"{tfidf_similarity:.4f}"
    )

    print(
        f"Semantic Similarity : "
        f"{semantic_similarity:.4f}"
    )

    print(
        f"Word Overlap        : "
        f"{word_overlap:.4f}"
    )

    print(
        f"Length Difference   : "
        f"{length_difference:.4f}"
    )

    print(
        f"\nML Duplicate Prob.  : "
        f"{duplicate_probability:.4f}"
    )

    print(
        f"Hybrid Score        : "
        f"{hybrid_score:.4f}"
    )

    print(
        f"ML Prediction       : "
        f"{'DUPLICATE' if ml_prediction == 1 else 'NOT DUPLICATE'}"
    )

    print(
        f"Final Prediction    : "
        f"{'DUPLICATE' if final_prediction == 1 else 'NOT DUPLICATE'}"
    )

    print(
        f"Final Confidence    : "
        f"{confidence:.4f}"
    )

    print("======================================\n")


    # ========================================================
    # RESPONSE TO FRONTEND
    # ========================================================

    response = {

        # Final result
        "is_duplicate":
            bool(final_prediction),


        # Human-readable label
        "label":
            (
                "Duplicate"
                if final_prediction == 1
                else "Not Duplicate"
            ),


        # Final confidence
        "confidence":
            round(
                confidence,
                4
            ),


        # IMPORTANT:
        # Your script.js uses data.similarity.
        # Therefore this field MUST be present.

        "similarity":
            round(
                hybrid_score,
                4
            ),


        # Keep hybrid_score too, for debugging
        # and future frontend use.

        "hybrid_score":
            round(
                hybrid_score,
                4
            ),


        # Logistic Regression probability

        "duplicate_probability":
            round(
                duplicate_probability,
                4
            ),


        # Individual feature values

        "tfidf_similarity":
            round(
                tfidf_similarity,
                4
            ),


        "semantic_similarity":
            round(
                semantic_similarity,
                4
            ),


        "word_overlap":
            round(
                word_overlap,
                4
            ),


        "length_difference":
            round(
                length_difference,
                4
            ),
    }


    return jsonify(response)


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )