"""
evaluation/evaluate.py

Computes standard binary-classification evaluation metrics for the
fraud detector, persists them as JSON (for programmatic comparison
against future models -- Random Forest, Autoencoder, ensemble), and
saves a confusion matrix plot for inclusion in the project report.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless/script execution
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Computes and persists evaluation metrics for a trained fraud model."""

    def __init__(self, model_name: str, metrics_dir: Path, plots_dir: Path):
        self.model_name = model_name
        self.metrics_dir = Path(metrics_dir)
        self.plots_dir = Path(plots_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute accuracy, precision, recall, F1, ROC-AUC, and confusion
        matrix. `y_proba` should be the probability of the positive
        (fraud) class.
        """
        logger.info("Computing evaluation metrics")

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            "support": {
                "total_samples": int(len(y_true)),
                "fraud_samples": int(np.sum(y_true == 1)),
                "legitimate_samples": int(np.sum(y_true == 0)),
            },
        }

        logger.info(
            f"Accuracy={metrics['accuracy']:.4f} | Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | F1={metrics['f1_score']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )
        logger.info(
            f"Confusion Matrix -> TP={tp}, TN={tn}, FP={fp}, FN={fn}"
        )

        return metrics

    # ------------------------------------------------------------------
    def save_metrics(self, metrics: Dict[str, Any], filename: str = None) -> Path:
        """Persist metrics dictionary to a JSON file."""
        filename = filename or f"{self.model_name}_metrics.json"
        output_path = self.metrics_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Metrics saved to {output_path}")
        return output_path

    # ------------------------------------------------------------------
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Path:
        """Generate and save a confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix - {self.model_name}")
        plt.tight_layout()

        output_path = self.plots_dir / f"{self.model_name}_confusion_matrix.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Confusion matrix plot saved to {output_path}")
        return output_path
