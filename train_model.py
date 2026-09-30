"""
train_model.py
--------------
Train a Duplicate Question Detection model using:

    TF-IDF cosine similarity
    +
    GloVe semantic similarity
    +
    Word overlap
    +
    Length similarity
    +
    StandardScaler
    +
    Logistic Regression

Dataset:
    Heliosoph/Quora-Question-Pairs

Output:
    model/model.pkl
    model/vectorizer.pkl
"""

import os

import joblib
import numpy as np
import pandas as pd

from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nlp_utils import (
    FEATURE_NAMES,
    get_embedding_model,
    preprocess,
    semantic_similarity,
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

RANDOM_STATE = 42

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "model.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "vectorizer.pkl"
)


# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# 1. Load dataset
# --------------------------------------------------------------------------

def load_data():

    print(
        "Loading dataset from Hugging Face:"
        " Heliosoph/Quora-Question-Pairs ..."
    )

    dataset = load_dataset(
        "Heliosoph/Quora-Question-Pairs"
    )

    if "train" in dataset:
        split_name = "train"
    else:
        split_name = list(dataset.keys())[0]

    df = dataset[split_name].to_pandas()

    print("\nRaw dataset inspection")
    print("----------------------")

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    return df


# --------------------------------------------------------------------------
# 2. Normalize column names
# --------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame):

    rename_map = {}

    for column in df.columns:

        lower = column.strip().lower()

        if lower in (
            "question1",
            "q1",
            "question_1"
        ):
            rename_map[column] = "question1"

        elif lower in (
            "question2",
            "q2",
            "question_2"
        ):
            rename_map[column] = "question2"

        elif lower in (
            "is_duplicate",
            "isduplicate",
            "label",
            "duplicate"
        ):
            rename_map[column] = "is_duplicate"

    df = df.rename(
        columns=rename_map
    )

    required_columns = {
        "question1",
        "question2",
        "is_duplicate",
    }

    missing = required_columns - set(df.columns)

    if missing:

        raise ValueError(
            f"Dataset is missing expected columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df[
        [
            "question1",
            "question2",
            "is_duplicate"
        ]
    ]


# --------------------------------------------------------------------------
# 3. Clean dataset
# --------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame):

    print("\nCleaning data...")

    before = len(df)

    # Remove missing rows
    df = df.dropna(
        subset=[
            "question1",
            "question2",
            "is_duplicate"
        ]
    )

    # Remove empty questions
    df = df[
        (
            df["question1"]
            .astype(str)
            .str.strip()
            != ""
        )
        &
        (
            df["question2"]
            .astype(str)
            .str.strip()
            != ""
        )
    ]

    print(
        "Cleaning + lemmatizing text..."
    )

    # Preprocess Question 1
    q1_processed = df[
        "question1"
    ].apply(preprocess)

    # Preprocess Question 2
    q2_processed = df[
        "question2"
    ].apply(preprocess)

    df["question1_clean"] = (
        q1_processed
        .apply(lambda x: x[0])
    )

    df["question1_lemmas"] = (
        q1_processed
        .apply(lambda x: x[1])
    )

    df["question2_clean"] = (
        q2_processed
        .apply(lambda x: x[0])
    )

    df["question2_lemmas"] = (
        q2_processed
        .apply(lambda x: x[1])
    )

    # Make labels integers
    df["is_duplicate"] = (
        df["is_duplicate"]
        .astype(int)
    )

    after = len(df)

    print(
        f"Rows before cleaning: {before}"
    )

    print(
        f"Rows after cleaning: {after}"
    )

    print("\nLabel distribution:")

    print(
        df["is_duplicate"]
        .value_counts()
    )

    return df.reset_index(
        drop=True
    )


# --------------------------------------------------------------------------
# 4. Build features
# --------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    vectorizer,
    embedding_model,
    fit=False
):

    """
    Build pair-level features:

        1. TF-IDF cosine similarity
        2. Semantic similarity
        3. Word overlap
        4. Length similarity
    """

    q1 = df[
        "question1_clean"
    ].tolist()

    q2 = df[
        "question2_clean"
    ].tolist()

    # --------------------------------------------------------------
    # Fit vectorizer only on training data
    # --------------------------------------------------------------

    if fit:

        vectorizer.fit(
            q1 + q2
        )

    # --------------------------------------------------------------
    # Transform questions
    # --------------------------------------------------------------

    q1_vectors = vectorizer.transform(q1)
    q2_vectors = vectorizer.transform(q2)

    # --------------------------------------------------------------
    # TF-IDF cosine similarity
    # --------------------------------------------------------------

    numerator = np.asarray(
        q1_vectors
        .multiply(q2_vectors)
        .sum(axis=1)
    ).flatten()

    q1_norm = np.sqrt(
        np.asarray(
            q1_vectors
            .multiply(q1_vectors)
            .sum(axis=1)
        ).flatten()
    )

    q2_norm = np.sqrt(
        np.asarray(
            q2_vectors
            .multiply(q2_vectors)
            .sum(axis=1)
        ).flatten()
    )

    denominator = q1_norm * q2_norm

    tfidf_cosine = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0
    )

    # --------------------------------------------------------------
    # Other features
    # --------------------------------------------------------------

    semantic_sim = []
    word_overlap = []
    length_similarity = []

    for (
        q1_clean,
        q2_clean,
        q1_lemmas,
        q2_lemmas
    ) in zip(
        q1,
        q2,
        df["question1_lemmas"],
        df["question2_lemmas"]
    ):

        # Semantic similarity
        semantic_sim.append(
            semantic_similarity(
                q1_lemmas,
                q2_lemmas,
                embedding_model
            )
        )

        # Word overlap
        set_a = set(q1_lemmas)
        set_b = set(q2_lemmas)

        union = set_a | set_b

        if union:

            overlap = (
                len(set_a & set_b)
                / len(union)
            )

        else:

            overlap = 0.0

        word_overlap.append(
            overlap
        )

        # Length similarity
        length1 = len(
            q1_clean.split()
        )

        length2 = len(
            q2_clean.split()
        )

        difference = abs(
            length1 - length2
        )

        similarity = 1.0 / (
            1.0 + difference
        )

        length_similarity.append(
            similarity
        )

    # --------------------------------------------------------------
    # Final matrix
    # --------------------------------------------------------------

    features = np.column_stack(
        [
            tfidf_cosine,
            semantic_sim,
            word_overlap,
            length_similarity,
        ]
    )

    return features


# --------------------------------------------------------------------------
# 5. Main training pipeline
# --------------------------------------------------------------------------

def main():

    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------------
    # Normalize
    # --------------------------------------------------------------

    df = normalize_columns(df)

    # --------------------------------------------------------------
    # Clean
    # --------------------------------------------------------------

    df = clean_dataframe(df)

    # --------------------------------------------------------------
    # Load GloVe
    # --------------------------------------------------------------

    embedding_model = (
        get_embedding_model()
    )

    # --------------------------------------------------------------
    # Train/Test split
    # --------------------------------------------------------------

    X_train_df, X_test_df, y_train, y_test = (
        train_test_split(
            df,
            df["is_duplicate"],
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=df["is_duplicate"],
        )
    )

    print("\nTraining rows:", len(X_train_df))
    print("Testing rows:", len(X_test_df))

    # --------------------------------------------------------------
    # TF-IDF Vectorizer
    # --------------------------------------------------------------

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    # --------------------------------------------------------------
    # Build training features
    # --------------------------------------------------------------

    print(
        "\nBuilding training features..."
    )

    X_train = build_features(
        X_train_df,
        vectorizer,
        embedding_model,
        fit=True
    )

    # --------------------------------------------------------------
    # Build testing features
    # --------------------------------------------------------------

    print(
        "Building testing features..."
    )

    X_test = build_features(
        X_test_df,
        vectorizer,
        embedding_model,
        fit=False
    )

    print(
        "\nFeature matrix shape:"
    )

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    # --------------------------------------------------------------
    # Logistic Regression Pipeline
    # --------------------------------------------------------------
    #
    # StandardScaler is important because:
    #
    # TF-IDF similarity       -> 0 to 1
    # Semantic similarity     -> 0 to 1
    # Word overlap            -> 0 to 1
    # Length similarity       -> 0 to 1
    #
    # Scaling makes the features comparable.
    # --------------------------------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    print(
        "\nEvaluating model..."
    )

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(
        "\n=============================="
    )

    print(
        "     MODEL EVALUATION"
    )

    print(
        "=============================="
    )

    print(
        f"Accuracy :  {accuracy:.4f}"
    )

    print(
        f"Precision:  {precision:.4f}"
    )

    print(
        f"Recall   :  {recall:.4f}"
    )

    print(
        f"F1 Score :  {f1:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    # --------------------------------------------------------------
    # Feature coefficients
    # --------------------------------------------------------------

    classifier = (
        model
        .named_steps["classifier"]
    )

    print(
        "\nFeature coefficients:"
    )

    for name, coefficient in zip(
        FEATURE_NAMES,
        classifier.coef_[0]
    ):

        print(
            f"{name:22s}: "
            f"{coefficient:+.4f}"
        )

    # --------------------------------------------------------------
    # Save model and vectorizer
    # --------------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH
    )

    print(
        "\n=============================="
    )

    print(
        "Training completed successfully!"
    )

    print(
        f"Model saved to: {MODEL_PATH}"
    )

    print(
        f"Vectorizer saved to: "
        f"{VECTORIZER_PATH}"
    )

    print(
        "=============================="
    )


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

if __name__ == "__main__":
    main()