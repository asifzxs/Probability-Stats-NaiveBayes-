# 🛡️ SpamShield — Probabilistic SMS Spam Detection

**BAMAT207 Probability and Statistics · Project Activity** — Topic: Probability and Random Variables

**Mohammad Asif Shaik** · Reg. No. 25BDS0107 · Faculty: Dhivyaa C R

A Python system that classifies an SMS as **Spam (1)** or **Genuine / Ham (0)** using dataset probabilities,
Bayes' theorem, a Bernoulli random variable and a **Multinomial Naive Bayes** classifier, with an
interactive web app for live demos.

![Word probabilities](figures/word_probabilities.png)

## Results

| Metric | Value |
|---|---|
| Messages (raw → after removing duplicates) | 5,574 → 5,171 |
| P(Spam) / P(Ham) | 0.1263 / 0.8737 |
| E(Y) / Var(Y) for Y ~ Bernoulli(p) | 0.1263 / 0.1103 |
| Test accuracy | **98.07%** |
| Precision / Recall / F1 (spam) | 95.12% / 89.31% / 92.13% |
| Spam missed (FN) / genuine wrongly flagged (FP) | 14 / 6 (of 1,035 test messages) |

| Word | Msgs with word | P(Spam \| word) direct | P(Spam \| word) via Bayes |
|---|---|---|---|
| free | 203 | 0.7291 | 0.7291 |
| prize | 73 | 1.0000 | 1.0000 |
| win | 55 | 0.8909 | 0.8909 |
| urgent | 62 | 0.9194 | 0.9194 |
| offer | 30 | 0.8000 | 0.8000 |

## Project files

| File | What it is |
|---|---|
| `spam_detector.ipynb` | Notebook covering all 8 activities (also runs on Google Colab) |
| `app.py` | SpamShield web app — live scanner, charts, Bayes lab |
| `spam_core.py` | Shared probability and model code used by the app and report |
| `make_report.py` | Regenerates `Project_Report.docx` from live results |
| `figures/` | Class distribution, word probabilities, random variable, confusion matrix, top words |
| `data/SMSSpamCollection` | Local copy of the UCI dataset, so the demo works offline |

## Run it

```bash
git clone https://github.com/asifzxs/Probability-Stats-NaiveBayes-.git
cd Probability-Stats-NaiveBayes-
pip install -r requirements.txt
streamlit run app.py          # web app (or double-click run_app.bat on Windows)
jupyter notebook spam_detector.ipynb
```

## What the web app shows

1. **⚡ Live Scanner** — type any SMS: verdict, P(Spam | Message), P(Ham | Message), a spam gauge,
   every word coloured by its evidence, and a log-odds waterfall showing how the prior and each word add up to the decision.
2. **📦 Dataset** — cleaning summary, class distribution, message-length distributions.
3. **🔎 Word Probabilities** — P(Spam | word) by direct counting vs Bayes' theorem, plus a "try any word" box.
4. **🎲 Random Variable** — Bernoulli PMF, E(Y), Var(Y) and a Law of Large Numbers simulation.
5. **🧠 Model Evaluation** — accuracy, precision, recall, F1, confusion matrix, and the exact misclassified messages.
6. **🧪 Bayes Lab** — sliders for P(Spam), P(word | Spam), P(word | Ham) showing the prior → posterior update.

## Method

1. **Dataset handling** — load, check missing values, drop 403 duplicate messages, encode spam = 1, ham = 0.
2. **Dataset analysis** — P(Spam) = #spam / #total, P(Ham) = #ham / #total, bar chart.
3. **Word probabilities** — P(Spam | w) = #spam containing w / #messages containing w, verified with
   P(Spam | w) = P(w | Spam) P(Spam) / P(w).
4. **Random variable** — Y = 1 if spam else 0, so E(Y) = p and Var(Y) = p(1 − p).
5. **Data preparation** — stratified 80/20 split (`random_state=42`), `CountVectorizer` word counts.
6. **Model** — `MultinomialNB()`: P(Spam | m) ∝ P(Spam) Π P(w | Spam)^count(w).
7. **Evaluation** — accuracy, precision, recall, F1, confusion matrix, false positives and false negatives.
8. **Real-time testing** — `predict_sms()` in the notebook and the Live Scanner in the app.

## Dataset

Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection*. UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/228/sms+spam+collection
