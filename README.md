# Sentiment Analysis of Drug Reviews Using NLP

Analyzes patient drug reviews to classify sentiment and extract key factors behind negative medication experiences. Uses classical ML models (SVM, Logistic Regression) and a fine-tuned DistilBERT transformer.

---

## Dataset

Uses the [UCI Drug Review Dataset](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com) (`drugsComTrain_raw.tsv`). Download it separately and place it in the project root before running.

Sentiment labels are derived from ratings: **positive** (≥ 7), **neutral** (5–6), **negative** (< 5).

---

## Installation

```bash
pip install pandas numpy matplotlib scikit-learn nltk spacy textblob wordcloud vaderSentiment transformers datasets torch
pip install numpy==1.26.4  # required for transformers compatibility
python -m spacy download en_core_web_sm
```

NLTK resources are downloaded automatically when the notebook runs.

---

## Project Overview

### Part 1 — Sentiment Analysis

Preprocessing pipeline covers tokenization, lemmatization, stopword removal, and raw text cleaning (HTML entities, URLs, abbreviations). Reviews are vectorized with TF-IDF and classified using LinearSVC and Logistic Regression with 5-fold stratified cross-validation. A DistilBERT model is also fine-tuned on a 50K sample with weighted cross-entropy loss to handle class imbalance, achieving its best validation loss at Epoch 2 before early stopping.

Additional analysis includes opinion mining (spaCy + VADER) and aspect-based sentiment analysis (ABSA) to identify what patients feel positively or negatively about.

### Part 2 — Negative Review Factor Extraction

Negative reviews are analyzed in isolation to surface root causes of dissatisfaction through n-gram frequency and TF-IDF analysis, LDA topic modeling (5 topics), word cloud visualizations, and sentiment polarity distribution.

Recurring themes identified: mental health side effects, hormonal/menstrual issues, birth control complaints, pain management, and sleep disruption.

---

## Results

| Model | Test Accuracy |
|---|---|
| LinearSVC | ~0.79 |
| Logistic Regression | ~0.80 |
| DistilBERT | best at epoch 2 (val loss 0.667) |

