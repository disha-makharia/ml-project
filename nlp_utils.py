"""
nlp_utils.py
-------------
Shared text-cleaning, lemmatization, and semantic-similarity utilities.

Both train_model.py (training) and app.py (inference) import this module,
so the exact same preprocessing and feature logic runs at train time and at
prediction time. This matters: if training and inference clean/feature-ize
text even slightly differently, the model effectively sees different inputs
than it was trained on, which is a common cause of "trained fine but gives
wrong results in the app" bugs.
"""

import re
import string

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------------------------------------------------------
# One-time NLTK data downloads (no-op if already present)
# --------------------------------------------------------------------------
_NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}
for _pkg, _path in _NLTK_RESOURCES.items():
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------
def clean_text(text) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_tokens(tokens):
    """Lemmatize as verb first, then noun, so 'learning' -> 'learn',
    'cities' -> 'city', etc."""
    lemmas = []
    for tok in tokens:
        lemma = _lemmatizer.lemmatize(tok, pos="v")
        lemma = _lemmatizer.lemmatize(lemma, pos="n")
        lemmas.append(lemma)
    return lemmas


def preprocess(text: str):
    """Returns (cleaned_text, content_lemmas) where content_lemmas has
    stopwords removed and remaining words lemmatized. Used for the word
    overlap feature and for embedding lookups."""
    cleaned = clean_text(text)
    tokens = cleaned.split()
    content_tokens = [t for t in tokens if t not in STOPWORDS]
    lemmas = lemmatize_tokens(content_tokens)
    return cleaned, lemmas


# --------------------------------------------------------------------------
# Semantic similarity via pretrained word embeddings
# --------------------------------------------------------------------------
# We use pretrained GloVe vectors (looked up, not trained) so that words with
# similar meaning end up with similar vectors even when they share no
# characters at all (e.g. "purchase" vs "buy"). This is NOT a neural network
# we train — it's a fixed lookup table plus simple averaging, which keeps the
# project within "no deep learning / no complicated architectures".
_EMBEDDING_MODEL_NAME = "glove-wiki-gigaword-50"
_embedding_model = None


def get_embedding_model():
    """Lazily loads pretrained GloVe vectors. Downloaded once via gensim's
    downloader and cached locally (~/gensim-data) on subsequent runs."""
    global _embedding_model
    if _embedding_model is None:
        import gensim.downloader as gensim_api

        print(
            f"Loading pretrained word vectors '{_EMBEDDING_MODEL_NAME}' "
            "(first run downloads ~65MB, then it's cached locally)..."
        )
        _embedding_model = gensim_api.load(_EMBEDDING_MODEL_NAME)
    return _embedding_model


def sentence_vector(tokens, model) -> np.ndarray:
    """Average word-vector for a list of tokens. Unknown words are skipped;
    if none are known, returns a zero vector."""
    vectors = [model[t] for t in tokens if t in model]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def semantic_similarity(tokens_a, tokens_b, model) -> float:
    """Cosine similarity between the two questions' averaged word vectors."""
    vec_a = sentence_vector(tokens_a, model)
    vec_b = sentence_vector(tokens_b, model)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# --------------------------------------------------------------------------
# Feature vector for a single pair (used identically in training & app.py)
# --------------------------------------------------------------------------
FEATURE_NAMES = ["tfidf_cosine", "semantic_similarity", "word_overlap", "length_diff"]


def build_pair_features(question1: str, question2: str, vectorizer, embedding_model):
    """Builds the 4-feature vector for ONE question pair. Used at inference
    time in app.py (training builds the same features in batch for speed —
    see train_model.py — but the definitions below must match exactly)."""
    q1_clean, q1_lemmas = preprocess(question1)
    q2_clean, q2_lemmas = preprocess(question2)

    # 1. Lexical similarity: TF-IDF cosine similarity
    q1_vec = vectorizer.transform([q1_clean])
    q2_vec = vectorizer.transform([q2_clean])
    numerator = float(q1_vec.multiply(q2_vec).sum())
    q1_norm = float(np.sqrt(q1_vec.multiply(q1_vec).sum()))
    q2_norm = float(np.sqrt(q2_vec.multiply(q2_vec).sum()))
    denom = q1_norm * q2_norm
    tfidf_cosine = numerator / denom if denom != 0 else 0.0

    # 2. Semantic similarity: averaged word-embedding cosine similarity
    sem_sim = semantic_similarity(q1_lemmas, q2_lemmas, embedding_model)

    # 3. Word overlap: Jaccard similarity of lemmatized, stopword-free tokens
    set_a, set_b = set(q1_lemmas), set(q2_lemmas)
    union = set_a | set_b
    word_overlap = len(set_a & set_b) / len(union) if union else 0.0

    # 4. Length difference (in words, on the raw cleaned text)
    length_diff = abs(len(q1_clean.split()) - len(q2_clean.split()))

    return np.array([[tfidf_cosine, sem_sim, word_overlap, length_diff]])
