# Duplicate Question Pairs Detection

**DuplicateIQ** — a simple college project that predicts whether two questions
mean the same thing, using classic NLP (TF-IDF) and Logistic Regression.

## Problem Statement

Platforms like Quora receive many questions that ask the same thing in
different words. Automatically detecting these duplicate pairs helps reduce
redundancy and points users toward answers that already exist.

## Dataset

The project uses the **Quora Question Pairs** dataset from Hugging Face:

```python
from datasets import load_dataset

ds = load_dataset("Heliosoph/Quora-Question-Pairs")
```

Each row contains `question1`, `question2`, and an `is_duplicate` label.
`train_model.py` inspects the actual column names at load time so it adapts
if they differ slightly (e.g. casing).

## Technologies

- HTML, CSS, JavaScript (frontend — no frameworks)
- Python, Flask (backend)
- Scikit-learn, Pandas, NumPy (machine learning)

## Machine Learning Approach

```
Data preprocessing (clean + lemmatize)
      ↓
Pair-level feature extraction
(TF-IDF cosine similarity, semantic embedding similarity,
 lemmatized word overlap, length difference)
      ↓
Logistic Regression
      ↓
Prediction (Duplicate / Not Duplicate)
```

Preprocessing lowercases the text, strips punctuation/whitespace, removes
stopwords, and lemmatizes the remaining words (so "learning" and "learn"
are treated the same).

Two pairs of questions can be duplicates while sharing almost no words
("How do I buy a used car?" vs "What's the best way to purchase a
second-hand vehicle?"), and two pairs can share many words while meaning
different things. Relying on token overlap alone misses the first case and
can be fooled by the second, so the model is trained on **four**
interpretable pair-level features instead of just one:

1. **TF-IDF cosine similarity** — lexical overlap, weighted by how rare/
   informative each word (and word pair, via bigrams) is.
2. **Semantic embedding similarity** — cosine similarity between each
   question's averaged pretrained GloVe word vectors. This is what lets the
   model recognize that "purchase" and "buy" mean the same thing even
   though they're completely different tokens.
3. **Lemmatized word overlap** — Jaccard similarity of content words
   (stopwords removed, words lemmatized).
4. **Length difference** — absolute difference in word count.

All four features, plus the exact same cleaning/lemmatization logic, live in
`nlp_utils.py` and are imported by both `train_model.py` and `app.py`, so
training and the live app can never drift out of sync.

## Installation

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model (this downloads the dataset, downloads the pretrained GloVe
word vectors used for the semantic-similarity feature the first time it
runs — about 65MB, cached afterwards — and creates `model/model.pkl` and
`model/vectorizer.pkl`):

```bash
python train_model.py
```

Run the app:

```bash
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

## Project Structure

```
duplicate-question-pairs/
│
├── app.py               # Flask backend (routes: / and /predict)
├── train_model.py        # Loads data, engineers features, trains and saves the model
├── nlp_utils.py          # Shared cleaning/lemmatization/feature logic (train + app)
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   ├── model.pkl          # Trained Logistic Regression model (generated)
│   └── vectorizer.pkl     # Fitted TF-IDF vectorizer (generated)
│
├── templates/
│   └── index.html         # Frontend page
│
└── static/
    ├── style.css
    └── script.js
```

## Evaluation

`train_model.py` prints the following metrics on a held-out 20% test split
(`random_state=42`):

- **Accuracy** — overall proportion of correct predictions
- **Precision** — of the pairs predicted as duplicates, how many actually were
- **Recall** — of the true duplicate pairs, how many were correctly found
- **F1 Score** — harmonic mean of precision and recall
- **Confusion Matrix** — full breakdown of true/false positives and negatives

None of these are hard-coded — they're computed fresh from `sklearn.metrics`
each time the script runs.

## Future Scope

- Use BERT or Sentence Transformers for deeper semantic similarity
- Train on the full, larger version of the Quora dataset
- Add n-gram or fuzzy-matching features
- Deploy the app (e.g. to a small cloud VM or PaaS)
