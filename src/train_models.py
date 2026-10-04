import json
from pathlib import Path

import numpy as np
from gensim.models import FastText, Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def _load_split(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_metrics(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _write_predictions(path, true_labels, pred_labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"true_label": true_label, "pred_label": pred_label}
        for true_label, pred_label in zip(true_labels, pred_labels)
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def run_tfidf_baseline(train_df, val_df, test_df, results_dir):
    train_texts = [item["text"] for item in train_df]
    train_labels = [item["label"] for item in train_df]
    test_texts = [item["text"] for item in test_df]
    test_labels = [item["label"] for item in test_df]

    vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, train_labels)
    pred = clf.predict(X_test)

    res = {
        "method": "tfidf_logistic",
        "accuracy": accuracy_score(test_labels, pred),
        "macro_f1": f1_score(test_labels, pred, average="macro"),
        "labels": sorted(set(train_labels) | set(test_labels)),
    }
    _write_metrics(results_dir / "tfidf_metrics.json", res)
    _write_predictions(results_dir / "tfidf_predictions.json", test_labels, pred)
    print(res)


def run_word2vec_experiment(train_df, val_df, test_df, results_dir):
    train_texts = [item["text"].split() for item in train_df]
    test_texts = [item["text"].split() for item in test_df]
    labels = [item["label"] for item in train_df]
    test_labels = [item["label"] for item in test_df]

    model = Word2Vec(
        sentences=train_texts,
        vector_size=64,
        window=5,
        min_count=2,
        epochs=10,
        sg=1,
        seed=42,
        workers=1,
    )

    def vec_of(doc):
        vectors = []
        for token in doc:
            if token in model.wv:
                vectors.append(model.wv[token])
        if not vectors:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)

    X_train = np.vstack([vec_of(doc) for doc in train_texts])
    X_test = np.vstack([vec_of(doc) for doc in test_texts])

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, labels)
    pred = clf.predict(X_test)

    res = {
        "method": "word2vec_mean",
        "accuracy": accuracy_score(test_labels, pred),
        "macro_f1": f1_score(test_labels, pred, average="macro"),
    }
    _write_metrics(results_dir / "word2vec_metrics.json", res)
    _write_predictions(results_dir / "word2vec_predictions.json", test_labels, pred)
    print(res)


def run_fasttext_experiment(train_df, val_df, test_df, results_dir):
    train_texts = [f"__label__{item['label']} {item['text']}" for item in train_df]
    test_texts = [item["text"] for item in test_df]
    test_labels = [item["label"] for item in test_df]

    fasttext_model = FastText(
        sentences=[text.split() for text in [item['text'] for item in train_df]],
        vector_size=64,
        window=5,
        min_count=2,
        epochs=10,
        sg=1,
        seed=42,
        workers=1,
    )

    # A lightweight FastText-classification style experiment:
    # using the learned vectors as features to a classifier.
    train_vectors = []
    for item in train_df:
        tokens = item["text"].split()
        vecs = [fasttext_model.wv[token] for token in tokens if token in fasttext_model.wv]
        if not vecs:
            train_vectors.append(np.zeros(fasttext_model.vector_size))
        else:
            train_vectors.append(np.mean(vecs, axis=0))

    test_vectors = []
    for item in test_df:
        tokens = item["text"].split()
        vecs = [fasttext_model.wv[token] for token in tokens if token in fasttext_model.wv]
        if not vecs:
            test_vectors.append(np.zeros(fasttext_model.vector_size))
        else:
            test_vectors.append(np.mean(vecs, axis=0))

    X_train = np.vstack(train_vectors)
    X_test = np.vstack(test_vectors)
    y_train = [item["label"] for item in train_df]

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    res = {
        "method": "fasttext_mean_vector",
        "accuracy": accuracy_score(test_labels, pred),
        "macro_f1": f1_score(test_labels, pred, average="macro"),
    }
    _write_metrics(results_dir / "fasttext_metrics.json", res)
    _write_predictions(results_dir / "fasttext_predictions.json", test_labels, pred)
    print(res)
