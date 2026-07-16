"""
models/xgboost_model.py

XGBoost classifier wrapper for fraud detection.

Design rationale
-----------------
This class wraps `xgboost.XGBClassifier` behind a minimal, sklearn-
compatible interface (fit / predict / predict_proba). This is a
deliberate design choice: scikit-learn's `StackingClassifier` and
`VotingClassifier` -- which this project's ensemble module will use in
a future phase -- require every base estimator to expose exactly this
interface. Building that compatibility in now means the ensemble module
can consume this class directly with zero refactoring.

The class also owns its own persistence (save/load) so a trained model
is a fully self-contained, reproducible artifact.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostFraudModel:
    """
    Sklearn-compatible wrapper around XGBClassifier, tailored for
    binary fraud classification on tabular financial data.
    """

    def __init__(self, hyperparameters: Dict[str, Any], random_seed: int = 42):
        self.hyperparameters = dict(hyperparameters)
        self.random_seed = random_seed
        self.model: Optional[XGBClassifier] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    def build(self) -> XGBClassifier:
        """Instantiate the underlying XGBClassifier from config hyperparameters."""
        early_stopping_rounds = self.hyperparameters.pop("early_stopping_rounds", None)

        model = XGBClassifier(
            n_estimators=self.hyperparameters.get("n_estimators", 500),
            max_depth=self.hyperparameters.get("max_depth", 6),
            learning_rate=self.hyperparameters.get("learning_rate", 0.05),
            subsample=self.hyperparameters.get("subsample", 0.8),
            colsample_bytree=self.hyperparameters.get("colsample_bytree", 0.8),
            gamma=self.hyperparameters.get("gamma", 0.1),
            min_child_weight=self.hyperparameters.get("min_child_weight", 1),
            reg_alpha=self.hyperparameters.get("reg_alpha", 0.1),
            reg_lambda=self.hyperparameters.get("reg_lambda", 1.0),
            eval_metric=self.hyperparameters.get("eval_metric", "aucpr"),
            tree_method=self.hyperparameters.get("tree_method", "hist"),
            objective=self.hyperparameters.get("objective", "binary:logistic"),
            n_jobs=self.hyperparameters.get("n_jobs", -1),
            random_state=self.random_seed,
            early_stopping_rounds=early_stopping_rounds,
        )
        return model

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBoostFraudModel":
        """
        Train the model. If a validation set is provided, early stopping
        is used to prevent overfitting -- important given XGBoost's
        capacity to memorize noise in high-dimensional fraud data.
        """
        self.model = self.build()

        logger.info(
            f"Training XGBoost on {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} features"
        )

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            logger.info(f"Using validation set of {X_val.shape[0]} samples for early stopping")

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        self._is_fitted = True
        logger.info("XGBoost training completed")

        if eval_set is not None:
            best_iteration = getattr(self.model, "best_iteration", None)
            if best_iteration is not None:
                logger.info(f"Best iteration (early stopping): {best_iteration}")

        return self

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability estimates; column 1 is the fraud-class probability."""
        self._check_fitted()
        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        """Return feature importances as a name -> score dictionary, sorted descending."""
        self._check_fitted()
        importances = self.model.feature_importances_
        importance_dict = dict(zip(feature_names, importances.tolist()))
        return dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Persist the trained model to disk via joblib."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path, hyperparameters: Optional[Dict[str, Any]] = None) -> "XGBoostFraudModel":
        """Load a previously trained model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No model found at {path}")

        instance = cls(hyperparameters=hyperparameters or {})
        instance.model = joblib.load(path)
        instance._is_fitted = True
        logger.info(f"Model loaded from {path}")
        return instance

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._is_fitted or self.model is None:
            raise RuntimeError(
                "XGBoostFraudModel has not been fitted yet. Call .fit() or .load() first."
            )
