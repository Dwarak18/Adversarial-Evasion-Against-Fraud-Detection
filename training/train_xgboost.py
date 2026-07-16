"""
training/train_xgboost.py

Orchestrates the full XGBoost training pipeline:

    1. Load raw data via the configured dataset loader.
    2. Run it through the leakage-safe preprocessing pipeline.
    3. Train XGBoostFraudModel with early stopping on the real
       (non-synthetic) validation split.
    4. Evaluate on the held-out test set.
    5. Persist the trained model, preprocessing artifacts, and metrics.

This module contains NO argument parsing or CLI concerns -- that lives
in main.py. Keeping orchestration logic separate from the entry point
means this pipeline can also be imported and called directly from a
future ensemble-training script.
"""

from pathlib import Path
from typing import Any, Dict

from evaluation.evaluate import ModelEvaluator
from models.xgboost_model import XGBoostFraudModel
from preprocessing.data_loader import get_loader
from preprocessing.preprocessor import FraudPreprocessor
from utils.config_loader import get_active_dataset_config
from utils.logger import get_logger
from utils.persistence import save_preprocessing_artifacts

logger = get_logger(__name__)


def run_training_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the end-to-end XGBoost training pipeline for the currently
    active dataset (as configured in config.yaml).

    Returns
    -------
    dict containing the computed evaluation metrics.
    """
    dataset_config = get_active_dataset_config(config)
    dataset_name = dataset_config["_dataset_name"]
    logger.info(f"=== Starting training pipeline for dataset: {dataset_name} ===")

    paths = config["_resolved_paths"]

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    loader = get_loader(raw_data_dir=paths["raw_data_dir"], dataset_config=dataset_config)
    df = loader.load()

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    preprocessor = FraudPreprocessor(config)
    X_train, X_val, X_test, y_train, y_val, y_test, artifacts = preprocessor.run(df)

    save_preprocessing_artifacts(
        artifacts=artifacts,
        output_dir=paths["saved_models_dir"],
        dataset_name=dataset_name,
    )

    # ------------------------------------------------------------------
    # 3. Train model
    # ------------------------------------------------------------------
    model_name = f"xgboost_{dataset_name}"
    xgb_wrapper = XGBoostFraudModel(
        hyperparameters=config["xgboost"],
        random_seed=config["project"]["random_seed"],
    )
    xgb_wrapper.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    model_path = Path(paths["saved_models_dir"]) / f"{model_name}.joblib"
    xgb_wrapper.save(model_path)

    # ------------------------------------------------------------------
    # 4. Evaluate on held-out test set
    # ------------------------------------------------------------------
    y_pred = xgb_wrapper.predict(X_test)
    y_proba = xgb_wrapper.predict_proba(X_test)[:, 1]

    evaluator = ModelEvaluator(
        model_name=model_name,
        metrics_dir=paths["metrics_dir"],
        plots_dir=paths["plots_dir"],
    )
    metrics = evaluator.compute_metrics(y_test.to_numpy(), y_pred, y_proba)

    # Feature importance is stored alongside metrics -- useful both for
    # the report and for later feature-selection work in the ensemble.
    metrics["top_20_feature_importance"] = dict(
        list(xgb_wrapper.get_feature_importance(artifacts.final_feature_order).items())[:20]
    )

    evaluator.save_metrics(metrics)
    evaluator.plot_confusion_matrix(y_test.to_numpy(), y_pred)

    logger.info(f"=== Training pipeline completed for dataset: {dataset_name} ===")
    return metrics
