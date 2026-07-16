# Adaptive Trust-Based Ensemble Defense Framework — XGBoost Module

Final-year B.E. Cyber Security project. This module implements the **XGBoost base classifier** for the fraud detection ensemble, built as a standalone, reusable component that a future stacking ensemble (Random Forest, Autoencoder, meta-learner) will plug into without modification.

## Project Structure

```
project/
├── data/
│   ├── raw/                # Place train_transaction.csv and train_identity.csv here
│   └── processed/          # Reserved for future cached intermediate datasets
├── models/
│   └── xgboost_model.py    # Sklearn-compatible XGBoost wrapper (fit/predict/predict_proba/save/load)
├── preprocessing/
│   ├── data_loader.py      # Dataset loaders (IEEE-CIS implemented; PaySim/ULB stubbed)
│   └── preprocessor.py     # Leakage-safe preprocessing pipeline
├── training/
│   └── train_xgboost.py    # Orchestrates load -> preprocess -> train -> evaluate -> save
├── evaluation/
│   ├── evaluate.py         # Metrics computation, JSON export, confusion matrix plot
│   ├── metrics/            # JSON metrics output (created at runtime)
│   └── plots/              # Confusion matrix images (created at runtime)
├── utils/
│   ├── logger.py           # Centralized logging (console + timestamped file)
│   ├── config_loader.py    # YAML config loading + path resolution
│   └── persistence.py      # Saves preprocessing artifacts separately, with manifest
├── saved_models/           # Trained model (.joblib) + preprocessing artifacts land here
├── configs/
│   └── config.yaml         # ALL paths, hyperparameters, preprocessing rules
├── logs/                   # Timestamped run logs
└── main.py                 # CLI entry point
```

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Setup

Download the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection) from Kaggle and place these two files in `data/raw/`:

```
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

## Running the Pipeline

```bash
python main.py
```

Optional flags:

```bash
python main.py --config configs/config.yaml   # explicit config path
python main.py --dataset ieee_cis              # override active dataset
```

## What Happens on Each Run

1. **Load** — `IEEECISLoader` merges `train_transaction.csv` and `train_identity.csv` on `TransactionID`, standardizes the target column to `target`.
2. **Preprocess** (leakage-safe order):
   - Train / validation / test split (validation carved from train, pre-SMOTE, for honest early stopping)
   - Drop columns exceeding the configured missing-value threshold (learned from train only)
   - Median imputation (numeric), constant imputation (categorical) — fit on train only
   - Ordinal encoding of categorical columns — fit on train only
   - Standard scaling of numeric columns — fit on train only
   - SMOTE oversampling — **training split only**, never validation/test
3. **Train** — `XGBoostFraudModel` trains with early stopping monitored on the real validation split.
4. **Evaluate** — Accuracy, Precision, Recall, F1, ROC-AUC, and full confusion matrix computed on the untouched test split.
5. **Persist**:
   - Trained model → `saved_models/xgboost_ieee_cis.joblib`
   - Preprocessing artifacts (imputers, encoder, scaler) → `saved_models/preprocessing/ieee_cis/*.joblib` + `manifest.json`
   - Metrics → `evaluation/metrics/xgboost_ieee_cis_metrics.json`
   - Confusion matrix plot → `evaluation/plots/xgboost_ieee_cis_confusion_matrix.png`
   - Full run log → `logs/run_<timestamp>.log`

## Design Notes for Future Ensemble Integration

- `XGBoostFraudModel` exposes `fit` / `predict` / `predict_proba`, matching the interface scikit-learn's `StackingClassifier` expects from every base estimator.
- Preprocessing is dtype-driven, not column-name-driven, so the same `FraudPreprocessor` class will work unchanged on PaySim/ULB once their loaders are implemented.
- All preprocessing artifacts are saved **individually** (not bundled) so other models in the ensemble (e.g., a Logistic Regression meta-learner) can reuse the exact same encoder/scaler without deserializing everything.
- `configs/config.yaml`'s `active_dataset` key is the single switch for changing datasets — no code changes required once `PaySimLoader`/`ULBLoader` are implemented.

## Extending to PaySim / ULB (Future Work)

1. Implement `PaySimLoader.load()` / `ULBLoader.load()` in `preprocessing/data_loader.py`, returning a DataFrame with a `target` column.
2. Set `active_dataset: paysim` (or `ulb`) in `config.yaml`.
3. Run `python main.py` — no other code changes required.
