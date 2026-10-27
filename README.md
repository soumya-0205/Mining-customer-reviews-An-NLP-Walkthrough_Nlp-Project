# Mining Customer Reviews: Sentiment, Topics & Aspects

An end-to-end classical NLP study of **22,641 women's clothing reviews**: sentiment classification, topic discovery and aspect-level sentiment, with ablations, significance tests, calibration and error analysis.

> **Headline:** A calibrated TF-IDF + Logistic Regression model reaches **94.9% binary accuracy (5-fold CV)** and **89.2 macro-F1** on held-out reviews. Neural baselines did not beat it, and the remaining errors look mostly like label noise (mixed or ambiguous reviews), not model capacity.

---

## Highlights

- **3 NLP tasks on one corpus:** sentiment classification (binary and 3-class), topic discovery (NMF vs LDA), aspect sentiment (fabric, fit, color, price, etc.)
- **7 candidate models** compared on the same split: Majority, Multinomial NB, Complement NB, Linear SVM, Logistic Regression, Word2vec average + LR, MLP on TF-IDF SVD
- **Ablation:** n-grams, sublinear TF, stop words, counts vs TF-IDF, char n-grams, class weights
- **Statistical testing:** McNemar test + 1,000-sample paired bootstrap on macro-F1
- **Calibration:** isotonic calibration cut ECE from 0.058 to 0.007
- **Learning curve** and **error analysis** (by rating, length, negation, contrast)
- **Topic validation:** NPMI coherence, LDA comparison, seed and bootstrap stability

---

## Repository Structure

```
.
├── data/
│   └── reviews.csv                  # input data (not included, see Data)
├── technical_nlp_analysis.py        # sentiment ablation, significance, calibration, learning curve, top features
├── topic_model_validation.py        # NMF K-sweep, NMF vs LDA, stability checks
├── ablation.json                    # generated: ablation macro-F1 scores
├── docs/
│   └── review_nlp_detailed_deck.pptx  # 31-slide technical walkthrough
├── requirements.txt
└── README.md
```

> The model comparison (SVM, Complement NB, Word2vec, MLP), per-class results, error analysis and aspect sentiment appear in the deck. Add those scripts here if they live in separate files.

---

## Data

Women's clothing e-commerce reviews, expected at `data/reviews.csv`.

| Column | Description |
|---|---|
| `Title` | Review title (13% missing) |
| `Review Text` | Review body (rows with no text are dropped) |
| `Rating` | Integer star rating, 1 to 5 |

**Corpus:** 23,486 rows → **22,641** reviews with text. Mean length 63 words (median 62, max 121), 1.31M tokens, 14,679 distinct words (42% appear only once).

**Labels (from star ratings, no human gold standard):**

| Class | Rating | Reviews |
|---|---|---|
| Negative | 1–2 | 2,370 |
| Neutral | 3 | 2,823 |
| Positive | 4–5 | 17,448 (77%) |

Binary task = negative vs positive (3-star excluded). Class imbalance is why **macro-F1** is the headline metric.

Add source and license: `[dataset link / license]`

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

Python 3.9+ recommended. The Word2vec baseline additionally needs `gensim`.

---

## Usage

```bash
python technical_nlp_analysis.py     # sentiment: ablation, McNemar + bootstrap, calibration, learning curve, features
python topic_model_validation.py     # topics: K-sweep, NMF vs LDA, stability
```

---

## Methodology

| Component | Setting |
|---|---|
| Split | Stratified 80/20, seed 42 (4,529 test reviews; 3,964 for binary). 5-fold CV for stability |
| Features | TF-IDF, words + pairs, `min_df=3`, sublinear TF, L2 norm → 45,767 sparse features |
| Logistic Regression | `C=4`, balanced class weights, `max_iter=2000` |
| Naive Bayes | `alpha=0.3` (multinomial and complement) |
| Linear SVM | `C=0.5`, balanced class weights |
| Word2vec | 100-d, window 5, 10 epochs, trained on training reviews only |
| NMF | 8 components, `nndsvda` init, seed 0, 400 iterations; 6,000 TF-IDF terms, `min_df=10`, `max_df=0.4` |
| Metrics | Accuracy, macro-F1, ROC-AUC, average precision, ECE, Brier |

**Preprocessing is deliberately minimal:** lowercase, words of 2+ characters, stop words kept (removing them cost 2.0 F1), no stemming.

---

## Results

### Model comparison (macro-F1, %)

| Model | Binary | 3-class |
|---|---|---|
| **Logistic regression (chosen)** | **89.2** | 65.3 |
| Linear SVM | 88.8 | 65.3 |
| Complement NB | 88.3 | **65.6** |
| MLP on TF-IDF SVD | 86.1 | 62.6 |
| Word2vec average + LR | 79.2 | 60.5 |
| Multinomial NB | 71.2 | 46.2 |
| Majority class | 46.8 | 29.0 |

Logistic regression, Linear SVM and Complement NB are within one point of each other and effectively tied. LR is kept for its calibratable probabilities and readable coefficients.

### Ablation (binary macro-F1, one change at a time)

| Variant | Macro-F1 |
|---|---|
| No sublinear TF | 89.8 |
| **Words + pairs (chosen)** | **89.2** |
| Words + pairs + triples | 89.2 |
| Raw counts, words + pairs | 88.1 |
| Stop words removed | 87.2 |
| Character n-grams | 87.1 |
| Words only | 87.0 |
| No class weights | 85.8 |

Word pairs add 2.2 points and class weights add 3.4. Triples and the sublinear switch change little (differences under ~0.5–0.6 are near the noise level on a 3,964-review test set).

### Is the winner real?

| Comparison | Result |
|---|---|
| LR vs Multinomial NB | +18 macro-F1 points, McNemar p = 2e-13, bootstrap 95% CI 15 to 21 |
| LR vs SVM / Complement NB | < 1 point: treat as tied |

### Discrimination & per-class performance

- **ROC-AUC 0.977**, average precision for the negative (minority) class **0.870**
- Binary confusion matrix: 410 TN, 64 FP, 129 FN, 3,361 TP
- 5-fold CV accuracy: **94.9% ± 0.3**

| Class (3-class LR) | Precision | Recall | F1 | Test reviews |
|---|---|---|---|---|
| Negative | 0.572 | 0.601 | 0.586 | 474 |
| Neutral | 0.413 | 0.469 | 0.439 | 565 |
| Positive | 0.946 | 0.919 | 0.932 | 3,490 |

Neutral (3-star) is the weak class: those reviews mix praise and complaint.

### Calibration

Raw probabilities are **underconfident** (balanced class weights shift scores). Isotonic calibration (5-fold cross-fitting): **ECE 0.058 → 0.007**, Brier 0.041 → 0.034, AUC unchanged.

### Learning curve

Binary macro-F1 rises from 84.3 (1.6k reviews) to 89.1 (15.9k) and flattens near 12k. The last 4,000 reviews add only +0.2. More labels of the same kind will not help; better labels would.

### What the model learned

| Strongest positive | Coef | Strongest negative | Coef |
|---|---|---|---|
| love | 8.7 | disappointed | -9.0 |
| little | 8.3 | not | -7.6 |
| great | 7.6 | cheap | -7.2 |
| perfect | 7.0 | unflattering | -7.2 |
| comfortable | 7.0 | poor | -6.8 |
| beautiful | 6.1 | huge | -6.6 |

"little" looks odd but checks out: it appears in 16.6% of positive vs 5.7% of negative reviews, mostly in hedges ("a little bit", "a little big").

### Error analysis

- Overall 3-class error rate: **17%**. **39% of all errors** come from 3-star reviews; only 2% of errors flip between the two extremes.
- Error rate by rating: 1★ 27%, 2★ 46%, 3★ 53%, 4★ 23%, 5★ 2%
- Longer reviews fail more (11.2% for ≤30 words → 21.2% for 91+)
- Contrast words (but, however): 22.5% error vs 8.7% without. Negation words: 21.5% vs 11.0%
- Many errors are mixed content or arguably label noise (e.g., a complaint about delivery in a 5-star review)

**Three fixes that did not help:** negation tagging (65.3 → 65.5), clause-level scoring (65.3 → 65.2), grid search over n-gram range / `min_df` / `C` (89.2 → 89.1). The ceiling looks like label noise, not model capacity.

### Topic model (NMF, K=8)

NPMI coherence is flat from K=6 to 14, so **K=8 was chosen for interpretability**, not by metric. NMF coherence **0.076 vs 0.055 for LDA**. Stability is moderate: top-word overlap (Jaccard) of **0.74** across 5 seeds and **0.73** across bootstrap resamples.

| NPMI coherence by K | 4 | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|
| NMF | 0.065 | 0.076 | 0.076 | 0.070 | 0.073 | 0.078 |

| Topic (manual label) | Top words | Reviews | Avg stars | Negative |
|---|---|---|---|---|
| General fabric and look | like, just, fabric, look, really | 6,551 | 3.70 | 19.3% |
| Casual tops | shirt, cute, super, comfy, little | 1,881 | 4.10 | 11.1% |
| Sweaters and coats | sweater, soft, warm, cozy, sleeves | 1,560 | 4.28 | 9.4% |
| Sizing and fit | size, small, large, runs, medium | 2,042 | 4.12 | 9.1% |
| Dresses | dress, perfect, flattering, summer, wear | 3,561 | 4.27 | 8.7% |
| Skirts and blouses | beautiful, skirt, gorgeous, color, blouse | 1,627 | 4.39 | 6.7% |
| Pants and jeans | great, jeans, pants, perfect, fit | 3,020 | 4.66 | 2.8% |
| Color and softness | love, comfortable, colors, soft, fits | 2,399 | 4.70 | 2.6% |

Average negative share across all reviews: 10.5%.

### Aspect sentiment

The binary model scores ~106,000 sentences; a sentence counts toward an aspect if it contains one of the aspect's keywords.

| Aspect | Negative sentences |
|---|---|
| Price | 40.5% |
| Fabric and quality | 31.6% |
| Look and style | 20.9% |
| Length | 18.6% |
| Fit and size | 18.2% |
| Color | 10.8% |

Price is rare (9% of reviews) but harsh. Fit is mentioned in 63% of reviews and only 18.2% negative. Keyword overlap makes these figures approximate.

### Business takeaways

- **Fabric and quality** are over-represented in complaints; fit is not
- **20 items** hold 19% of negative reviews with 11% of reviews: inspect these first
- Hidden unhappy reviews (79) plus negative-leaning 3-star reviews (844) grow the follow-up queue by 44%

---

## Limitations

- Labels are star ratings with no human gold standard
- Aspect scores use keywords and a model trained on whole reviews
- Topic words overlap only 0.74 across seeds, and K=8 is not metric-driven
- Ablation variants and models are compared on the same held-out test set; for strict model selection, choose on cross-validation and report test once
- Recurrent networks and transformers were not tested
- One retailer, one split, one time period

## Next Steps

- Hand-label ~200 reviews to measure label noise and annotator agreement
- Fine-tune a pretrained transformer and test the gap
- Train a BiLSTM or CNN to complete the neural comparison
- Replace keyword aspects with a trained aspect-based model
- Deploy with isotonic-calibrated probabilities and monitor drift

---

## License

`[MIT / Apache-2.0 / etc.]`

## Author

**Soumya**: M.Sc. Data Science & AI, NMIMS Mumbai
`[LinkedIn / email / portfolio link]`
