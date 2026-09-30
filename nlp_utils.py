"""
nlp_utils.py
------------
Shared text cleaning, lemmatization, TF-IDF, semantic similarity,
word-overlap, and feature-building utilities.

The SAME feature logic is used during training and prediction.
"""

import re
import string

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# --------------------------------------------------------------------------
# NLTK resources
# --------------------------------------------------------------------------

_NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}

for package, path in _NLTK_RESOURCES.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(package, quiet=True)


STOPWORDS = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Convert text to lowercase, remove punctuation,
    normalize whitespace.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove punctuation
    text = re.sub(
        r"[%s]" % re.escape(string.punctuation),
        " ",
        text
    )

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def lemmatize_tokens(tokens):
    """
    Lemmatize tokens first as verbs and then as nouns.
    """

    lemmas = []

    for token in tokens:

        lemma = _lemmatizer.lemmatize(token, pos="v")
        lemma = _lemmatizer.lemmatize(lemma, pos="n")

        lemmas.append(lemma)

    return lemmas


def preprocess(text: str):
    """
    Returns:

        cleaned_text
        content_lemmas

    Stopwords are removed from content_lemmas.
    """

    cleaned = clean_text(text)

    tokens = cleaned.split()

    content_tokens = [
        token
        for token in tokens
        if token not in STOPWORDS
    ]

    lemmas = lemmatize_tokens(content_tokens)

    return cleaned, lemmas


# --------------------------------------------------------------------------
# GloVe semantic embeddings
# --------------------------------------------------------------------------

_EMBEDDING_MODEL_NAME = "glove-wiki-gigaword-50"

_embedding_model = None


def get_embedding_model():
    """
    Load pretrained GloVe word vectors.

    The model is loaded only once and cached by gensim.
    """

    global _embedding_model

    if _embedding_model is None:

        import gensim.downloader as gensim_api

        print(
            f"\nLoading pretrained word vectors "
            f"'{_EMBEDDING_MODEL_NAME}'..."
        )

        _embedding_model = gensim_api.load(
            _EMBEDDING_MODEL_NAME
        )

        print("GloVe model loaded successfully.")

    return _embedding_model


def sentence_vector(tokens, model):
    """
    Create a sentence vector by averaging the word vectors
    of all known tokens.
    """

    vectors = [
        model[token]
        for token in tokens
        if token in model
    ]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


def semantic_similarity(tokens_a, tokens_b, model):
    """
    Calculate cosine similarity between two averaged
    GloVe sentence vectors.
    """

    vector_a = sentence_vector(tokens_a, model)
    vector_b = sentence_vector(tokens_b, model)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = np.dot(vector_a, vector_b) / (
        norm_a * norm_b
    )

    # Keep value safely between 0 and 1
    similarity = max(0.0, min(1.0, float(similarity)))

    return similarity


# --------------------------------------------------------------------------
# Feature names
# --------------------------------------------------------------------------

FEATURE_NAMES = [
    "tfidf_cosine",
    "semantic_similarity",
    "word_overlap",
    "length_similarity",
]


# --------------------------------------------------------------------------
# Feature building
# --------------------------------------------------------------------------

def build_pair_features(
    question1: str,
    question2: str,
    vectorizer,
    embedding_model
):
    """
    Build the four features for one question pair.

    Features:

    1. TF-IDF cosine similarity
    2. GloVe semantic similarity
    3. Lemmatized word overlap
    4. Length similarity

    Returns:
        numpy array with shape (1, 4)
    """

    # --------------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------------

    q1_clean, q1_lemmas = preprocess(question1)
    q2_clean, q2_lemmas = preprocess(question2)

    # --------------------------------------------------------------
    # 1. TF-IDF cosine similarity
    # --------------------------------------------------------------

    q1_vector = vectorizer.transform([q1_clean])
    q2_vector = vectorizer.transform([q2_clean])

    numerator = float(
        q1_vector.multiply(q2_vector).sum()
    )

    q1_norm = float(
        np.sqrt(
            q1_vector.multiply(q1_vector).sum()
        )
    )

    q2_norm = float(
        np.sqrt(
            q2_vector.multiply(q2_vector).sum()
        )
    )

    denominator = q1_norm * q2_norm

    if denominator != 0:
        tfidf_cosine = numerator / denominator
    else:
        tfidf_cosine = 0.0

    # --------------------------------------------------------------
    # 2. Semantic similarity
    # --------------------------------------------------------------

    semantic_sim = semantic_similarity(
        q1_lemmas,
        q2_lemmas,
        embedding_model
    )

    # --------------------------------------------------------------
    # 3. Word overlap
    # --------------------------------------------------------------

    set_a = set(q1_lemmas)
    set_b = set(q2_lemmas)

    union = set_a | set_b

    if union:
        word_overlap = (
            len(set_a & set_b) / len(union)
        )
    else:
        word_overlap = 0.0

    # --------------------------------------------------------------
    # 4. Length similarity
    #
    # Instead of using raw length difference such as:
    #
    # 0, 1, 2, 5, 10...
    #
    # convert it into a 0-1 similarity value.
    #
    # Same length -> 1.0
    # Difference of 1 -> 0.5
    # Difference of 2 -> 0.33
    # etc.
    # --------------------------------------------------------------

    length1 = len(q1_clean.split())
    length2 = len(q2_clean.split())

    length_difference = abs(length1 - length2)

    length_similarity = 1.0 / (
        1.0 + length_difference
    )

    # --------------------------------------------------------------
    # Final feature vector
    # --------------------------------------------------------------

    features = np.array([
        [
            tfidf_cosine,
            semantic_sim,
            word_overlap,
            length_similarity,
        ]
    ])

    return features