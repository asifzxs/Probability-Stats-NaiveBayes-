"""
Core logic for the Probabilistic SMS Spam Detector.
BAMAT207 Probability and Statistics - Project Activity.

Used by both the notebook (spam_detector.ipynb) and the web app (app.py).
"""
import io
import os
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA = os.path.join(HERE, "data", "SMSSpamCollection")
UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SUSPICIOUS_WORDS = ["free", "prize", "win", "urgent", "offer"]


# ---------------------------------------------------------------- 1. Dataset
def load_raw():
    """Load the UCI SMS Spam Collection (local copy first, then UCI, then ucimlrepo)."""
    if os.path.exists(LOCAL_DATA):
        return pd.read_csv(LOCAL_DATA, sep="\t", names=["label", "text"], quoting=3)
    try:
        req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        with z.open("SMSSpamCollection") as f:
            return pd.read_csv(f, sep="\t", names=["label", "text"], quoting=3)
    except Exception:
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=228)
        df = d.data.original
        df.columns = ["text", "label"] if df.columns[0] != "label" else ["label", "text"]
        return df[["label", "text"]]


def clean(df):
    """Missing values, duplicates and label encoding. Returns (clean_df, summary)."""
    summary = {
        "raw_rows": len(df),
        "missing": df.isnull().sum().to_dict(),
    }
    df = df.dropna()
    summary["duplicates"] = int(df.duplicated(subset=["text"]).sum())
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["label_num"] = df["label"].map({"spam": 1, "ham": 0})
    df["length"] = df["text"].str.len()
    df["words"] = df["text"].str.split().str.len()
    summary["clean_rows"] = len(df)
    return df, summary


# ----------------------------------------------------------- 2. Priors
def priors(df):
    n = len(df)
    n_spam = int((df["label_num"] == 1).sum())
    n_ham = n - n_spam
    return {"n": n, "n_spam": n_spam, "n_ham": n_ham,
            "p_spam": n_spam / n, "p_ham": n_ham / n}


# ----------------------------------------------------------- 3. Words
def contains_word(text, word):
    return bool(re.search(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE))


def word_probabilities(df, words=SUSPICIOUS_WORDS):
    """P(Spam|word) directly from counts, and again via Bayes' theorem."""
    pr = priors(df)
    rows = []
    for w in words:
        has = df["text"].apply(lambda t: contains_word(t, w))
        n_word = int(has.sum())
        n_spam_word = int((has & (df["label_num"] == 1)).sum())
        p_word = n_word / pr["n"]
        p_word_spam = n_spam_word / pr["n_spam"]
        p_word_ham = (n_word - n_spam_word) / pr["n_ham"]
        direct = n_spam_word / n_word if n_word else float("nan")
        bayes = p_word_spam * pr["p_spam"] / p_word if p_word else float("nan")
        rows.append({
            "word": w,
            "msgs_with_word": n_word,
            "spam_with_word": n_spam_word,
            "ham_with_word": n_word - n_spam_word,
            "P(word)": p_word,
            "P(word|Spam)": p_word_spam,
            "P(word|Ham)": p_word_ham,
            "P(Spam|word) direct": direct,
            "P(Spam|word) Bayes": bayes,
            "lift": direct / pr["p_spam"] if n_word else float("nan"),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------- 4. Random variable
def random_variable(df):
    y = df["label_num"].to_numpy()
    p = y.mean()
    return {"p": p, "E(Y)": p, "Var(Y)": p * (1 - p),
            "SD(Y)": np.sqrt(p * (1 - p)),
            "sample_mean": float(np.mean(y)), "sample_var": float(np.var(y))}


# ----------------------------------------------------------- 5-7. Model
def train(df, test_size=0.2, seed=42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["label_num"], test_size=test_size,
        random_state=seed, stratify=df["label_num"])
    vec = CountVectorizer(lowercase=True)
    Xtr = vec.fit_transform(X_tr)
    Xte = vec.transform(X_te)
    model = MultinomialNB()
    model.fit(Xtr, y_tr)
    y_pred = model.predict(Xte)
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    test = pd.DataFrame({"text": X_te, "actual": y_te, "pred": y_pred,
                         "p_spam": model.predict_proba(Xte)[:, 1]})
    metrics = {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred),
        "recall": recall_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred),
        "cm": cm, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "report": classification_report(y_te, y_pred, target_names=["Ham", "Spam"], digits=4),
        "n_train": len(X_tr), "n_test": len(X_te),
        "train_spam_ratio": float(y_tr.mean()), "test_spam_ratio": float(y_te.mean()),
        "vocab": len(vec.vocabulary_),
    }
    return model, vec, metrics, test


# ----------------------------------------------------------- 8. Prediction
def predict(model, vec, message):
    proba = model.predict_proba(vec.transform([message]))[0]
    return {"label": "Spam" if proba[1] >= 0.5 else "Ham",
            "p_spam": float(proba[1]), "p_ham": float(proba[0])}


def explain(model, vec, message):
    """Per-word evidence: log P(w|Spam) - log P(w|Ham). Positive -> pushes toward spam."""
    analyzer = vec.build_analyzer()
    vocab = vec.vocabulary_
    lp = model.feature_log_prob_  # row 0 = ham, row 1 = spam
    out = []
    for tok in re.findall(r"\S+", message):
        toks = analyzer(tok)
        score = sum(lp[1, vocab[t]] - lp[0, vocab[t]] for t in toks if t in vocab)
        known = any(t in vocab for t in toks)
        out.append((tok, float(score), known))
    return out


def top_words(model, vec, n=20):
    lp = model.feature_log_prob_
    ratio = lp[1] - lp[0]
    names = np.array(vec.get_feature_names_out())
    idx_spam = np.argsort(ratio)[::-1][:n]
    idx_ham = np.argsort(ratio)[:n]
    return (pd.DataFrame({"word": names[idx_spam], "log_ratio": ratio[idx_spam]}),
            pd.DataFrame({"word": names[idx_ham], "log_ratio": ratio[idx_ham]}))


def run_all():
    raw = load_raw()
    df, summary = clean(raw)
    model, vec, metrics, test = train(df)
    return {"raw": raw, "df": df, "summary": summary, "priors": priors(df),
            "words": word_probabilities(df), "rv": random_variable(df),
            "model": model, "vec": vec, "metrics": metrics, "test": test}
