# Breast Cancer Classification — Random Forest Model Evaluation

A machine learning evaluation project classifying breast tumors as benign or malignant with a Random Forest classifier, built around rigorous evaluation: a dummy baseline, stratified cross-validation, and model selection driven by recall rather than accuracy.

## Dataset

- **Wisconsin Breast Cancer dataset** (via scikit-learn): 569 samples, 30 numeric features
- 357 benign / 212 malignant
- Labels are flipped so **malignant is the positive class** — recall therefore measures how many cancers the model catches, which is the error that matters most in a medical setting

## Method

1. **Split:** stratified 60/20/20 train / validation / test split, so each set preserves the benign–malignant ratio
2. **Cross-validation:** 5-fold StratifiedKFold on the training set, scoring accuracy, precision, recall, F1, and ROC-AUC
3. **Baseline:** a `DummyClassifier` that always predicts the majority class (benign)
4. **Model selection:** four Random Forest configurations (A–D) compared on cross-validation, varying `n_estimators` (50–200), `max_depth` (3, 5, unlimited), and `class_weight`
5. **Final model:** the selected configuration retrained on train + validation and evaluated **once** on the held-out test set

## Results

**Why accuracy is not enough:** the dummy baseline scores **62.3% accuracy with 0.0 recall** — it looks passable while missing every single malignant case.

**Model selection:** configuration D (200 trees, unlimited depth, `class_weight="balanced"`) was selected for the highest cross-validation recall (0.937) among the four configurations.

**Final held-out test results** (114 samples):

| Metric | Score |
|---|---|
| Accuracy | 0.956 |
| Precision | 0.930 |
| Recall | 0.952 |
| F1 | 0.941 |
| ROC-AUC | 0.995 |

Confusion matrix: 69 true negatives, 3 false positives, 2 false negatives, 40 true positives — the model missed 2 of 42 malignant cases.

**Top predictive features:** worst perimeter, mean concave points, worst area.

## Files

- `Henderson_Brice_ML_Evaluation.ipynb` — full notebook: exploration, cross-validation, baseline, model comparison, final evaluation, feature importances
- `breastCancerRF.py` — the same pipeline as a standalone script

## Run it

```bash
pip install scikit-learn pandas matplotlib seaborn
python breastCancerRF.py
```

All random seeds are fixed (`random_state=42`), so results reproduce exactly.

## Tech

Python, scikit-learn (RandomForest, StratifiedKFold, cross-validation), pandas, matplotlib / seaborn
