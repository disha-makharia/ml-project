"""
train_model.py
----------------
Trains a Logistic Regression model on TF-IDF + semantic-embedding features
for duplicate question detection.

Pipeline:
    Load dataset -> Clean & lemmatize -> Build features
    (TF-IDF cosine similarity, semantic embedding similarity,
     lemmatized word overlap, length difference)
    -> Train/test split -> Train Logistic Regression -> Evaluate
    -> Save model + vectorizer

Run:
    python train_model.py
"""

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_utils import get_embedding_model, preprocess, semantic_similarity

RANDOM_STATE = 42
MODEL_PATH = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"


# --------------------------------------------------------------------------
# 1. Load dataset
# --------------------------------------------------------------------------
def load_data():
    print("Loading dataset from Hugging Face: Heliosoph/Quora-Question-Pairs ...")
    ds = load_dataset("Heliosoph/Quora-Question-Pairs")

    split_name = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split_name].to_pandas()

    print("\nRaw dataset inspection")
    print("-----------------------")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    return df


# --------------------------------------------------------------------------
# 2. Normalize column names (dataset columns can vary in casing/naming)
# --------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        lower = col.strip().lower()
        if lower in ("question1", "q1", "question_1"):
            rename_map[col] = "question1"
        elif lower in ("question2", "q2", "question_2"):
            rename_map[col] = "question2"
        elif lower in ("is_duplicate", "isduplicate", "label", "duplicate"):
            rename_map[col] = "is_duplicate"

    df = df.rename(columns=rename_map)

    required = {"question1", "question2", "is_duplicate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return df[["question1", "question2", "is_duplicate"]]


# --------------------------------------------------------------------------
# 3. Cleaning + lemmatization
# --------------------------------------------------------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("\nCleaning data...")
    before = len(df)

    df = df.dropna(subset=["question1", "question2", "is_duplicate"])
    df = df[(df["question1"].astype(str).str.strip() != "") &
            (df["question2"].astype(str).str.strip() != "")]

    print("Cleaning + lemmatizing text (this can take a few minutes on the full dataset)...")
    q1_processed = df["question1"].apply(preprocess)
    q2_processed = df["question2"].apply(preprocess)

    df["question1_clean"] = q1_processed.apply(lambda t: t[0])
    df["question1_lemmas"] = q1_processed.apply(lambda t: t[1])
    df["question2_clean"] = q2_processed.apply(lambda t: t[0])
    df["question2_lemmas"] = q2_processed.apply(lambda t: t[1])

    df["is_duplicate"] = df["is_duplicate"].astype(int)

    after = len(df)
    print(f"Rows before cleaning: {before}")
    print(f"Rows after cleaning:  {after}")
    print("\nLabel distribution:")
    print(df["is_duplicate"].value_counts())

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. Feature engineering (batch, for speed over the whole dataset)
# --------------------------------------------------------------------------
def build_features(df: pd.DataFrame, vectorizer: TfidfVectorizer, embedding_model, fit: bool):
    """
    Builds pair-level features:
      - TF-IDF cosine similarity (lexical overlap, weighted by term rarity)
      - semantic embedding similarity (captures meaning, not just shared words)
      - lemmatized word overlap (Jaccard, stopwords removed)
      - question length difference
    """
    q1 = df["question1_clean"].tolist()
    q2 = df["question2_clean"].tolist()

    if fit:
        vectorizer.fit(q1 + q2)

    q1_vec = vectorizer.transform(q1)
    q2_vec = vectorizer.transform(q2)

    numerator = np.asarray(q1_vec.multiply(q2_vec).sum(axis=1)).flatten()
    q1_norm = np.sqrt(np.asarray(q1_vec.multiply(q1_vec).sum(axis=1)).flatten())
    q2_norm = np.sqrt(np.asarray(q2_vec.multiply(q2_vec).sum(axis=1)).flatten())
    denom = q1_norm * q2_norm
    tfidf_cosine = np.divide(
        numerator, denom, out=np.zeros_like(numerator), where=denom != 0
    )

    semantic_sim = []
    word_overlap = []
    length_diff = []

    for q1_clean, q2_clean, q1_lemmas, q2_lemmas in zip(
        q1, q2, df["question1_lemmas"], df["question2_lemmas"]
    ):
        semantic_sim.append(semantic_similarity(q1_lemmas, q2_lemmas, embedding_model))

        set_a, set_b = set(q1_lemmas), set(q2_lemmas)
        union = set_a | set_b
        word_overlap.append(len(set_a & set_b) / len(union) if union else 0.0)

        length_diff.append(abs(len(q1_clean.split()) - len(q2_clean.split())))

    features = np.column_stack([tfidf_cosine, semantic_sim, word_overlap, length_diff])
    return features


# --------------------------------------------------------------------------
# 5. Main pipeline
# --------------------------------------------------------------------------
def main():
    df = load_data()
    df = normalize_columns(df)
    df = clean_dataframe(df)

    embedding_model = get_embedding_model()

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df,
        df["is_duplicate"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["is_duplicate"],
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    print("\nBuilding features (TF-IDF + semantic embeddings)...")
    X_train = build_features(X_train_df, vectorizer, embedding_model, fit=True)
    X_test = build_features(X_test_df, vectorizer, embedding_model, fit=False)

    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nEvaluation Results")
    print("-------------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    print("\nFeature importance (Logistic Regression coefficients):")
    for name, coef in zip(
        ["tfidf_cosine", "semantic_similarity", "word_overlap", "length_diff"],
        model.coef_[0],
    ):
        print(f"  {name:22s} {coef:+.4f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
