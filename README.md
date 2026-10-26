# Review Text Analytics: Sentiment Classification & Topic Modeling

Classical NLP pipeline on customer review text, covering **(1) sentiment classification** with a rigorous ablation, significance testing and calibration analysis, and **(2) topic modeling** with coherence-based K selection, NMF vs LDA comparison and stability checks.

> Built with scikit-learn, no deep learning required. The goal is a well-validated, interpretable baseline.

---

## Highlights

- **Sentiment classification** (binary and 3-class) using TF-IDF + Logistic Regression
- **Ablation study** across n-gram range, sublinear TF, stopword removal, count vs TF-IDF, character n-grams and class weighting
- **Statistical comparison** of Logistic Regression vs Naive Bayes: exact McNemar-style sign test + paired bootstrap CI on macro-F1
- **Probability calibration**: ROC-AUC, PR-AUC, Brier score, ECE, reliability bins, isotonic calibration
- **Learning curve** to judge whether more data would help
- **Interpretability**: top positive/negative features with coefficients
- **Topic modeling**: NMF K-sweep scored by NPMI coherence, NMF vs LDA at K=8
- **Topic stability**: seed stability and bootstrap stability via best-match Jaccard

---

## Repository Structure

```
.
├── data/
│   └── reviews.csv                  # input data (not included, see Data)
├── technical_nlp_analysis.py        # sentiment classification + evaluation
├── topic_model_validation.py        # topic modeling + validation
├── ablation.json                    # generated: ablation F1 scores
├── requirements.txt
└── README.md
```

---

## Data

Place your dataset at `data/reviews.csv` with at least these columns:

| Column | Description |
|---|---|
| `Title` | Review title (nulls allowed) |
| `Review Text` | Review body (rows with null text are dropped) |
| `Rating` | Integer star rating, 1 to 5 |

**Preprocessing:** title and body are concatenated into one `text` field.

**Labels:**
- Binary: Rating ≤ 2 → negative, Rating ≥ 4 → positive (3-star reviews excluded)
- 3-class: ≤ 2 negative, 3 neutral, ≥ 4 positive

Add your dataset source and license here: `[dataset name / link]`.

---

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
scikit-learn
scipy
```

Python 3.9+ recommended.

---

## Usage

### 1. Sentiment classification & evaluation

```bash
python technical_nlp_analysis.py
```

Prints the following and writes `ablation.json`:
- Ablation table (binary macro-F1 | 3-class macro-F1) for each feature/model variant
- LR vs NB significance test and bootstrap CI
- AUC, PR-AUC (negative class), Brier score
- ECE (raw vs isotonic-calibrated) and calibration bins
- Learning curve across training fractions
- Top 12 positive and negative features with coefficients

### 2. Topic model validation

```bash
python topic_model_validation.py
```

Prints the following:
- NMF K-sweep (K = 4, 6, 8, 10, 12, 14) with NPMI coherence and reconstruction error
- NMF vs LDA coherence at K=8, plus LDA top words
- Seed stability (5 random initializations) and bootstrap stability (4 resamples)

---

## Methodology

### Sentiment model
| Component | Setting |
|---|---|
| Split | 80/20 stratified, `random_state=42` |
| Features | TF-IDF, unigrams + bigrams, `min_df=3`, sublinear TF |
| Classifier | Logistic Regression, `C=4`, `class_weight='balanced'` |
| Metric | Macro-F1 (robust to class imbalance) |

### Topic model
| Component | Setting |
|---|---|
| Features | TF-IDF, English stopwords removed, `min_df=10`, `max_df=0.4`, 6,000 features |
| Models | NMF (`nndsvda` init) and LDA (batch variational Bayes) |
| Coherence | NPMI over top-10 words per topic, using document co-occurrence |
| Stability | Mean best-match Jaccard of top-10 words vs the reference model |

---

## Results

Fill in after running the scripts.

### Sentiment classification (macro-F1, %)

| Variant | Binary | 3-class |
|---|---|---|
| TF-IDF unigram | – | – |
| **TF-IDF uni+bigram (chosen)** | – | – |
| TF-IDF uni+bi+trigram | – | – |
| Counts uni+bigram | – | – |
| TF-IDF no sublinear | – | – |
| TF-IDF + stopword removal | – | – |
| Char n-grams 2–5 | – | – |
| No class weighting | – | – |

### Model comparison & calibration

| Metric | Value |
|---|---|
| LR vs NB macro-F1 difference (95% CI) | – |
| Sign test p-value | – |
| ROC-AUC | – |
| PR-AUC (negative class) | – |
| Brier score | – |
| ECE raw → isotonic | – → – |

### Topics

| K | NPMI coherence |
|---|---|
| 4 | – |
| 6 | – |
| 8 | – |
| 10 | – |
| 12 | – |
| 14 | – |

| Model (K=8) | NPMI |
|---|---|
| NMF | – |
| LDA | – |

Seed stability: `–` · Bootstrap stability: `–`

**Discovered topics (K=8):** `[add topic labels and top words]`

---

## Known Limitations

- Ablation variants are scored on the held-out test set. For strict model selection, choose on cross-validation and report test once.
- `class_weight='balanced'` shifts predicted probabilities, so raw ECE is not directly comparable to an unweighted model.
- NPMI is computed on the training corpus (intrinsic coherence), and `max_df=0.4` can inflate it.
- Binary and 3-class results use different test splits, so they are not directly comparable.
- Stability uses one-directional best-match Jaccard (reference → candidate).

## Future Work

- Cross-validated model selection and nested evaluation
- Transformer baselines (e.g., DistilBERT) for comparison
- Aspect-based sentiment linking topics to ratings
- BERTopic comparison against NMF/LDA

---

## License

`[MIT / Apache-2.0 / etc.]`

## Author

**Soumya**: M.Sc. Data Science & AI, NMIMS Mumbai
`[LinkedIn / email / portfolio link]`
