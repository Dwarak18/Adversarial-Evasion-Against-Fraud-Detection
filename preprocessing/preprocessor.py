"""
preprocessing/preprocessor.py

Dataset-agnostic preprocessing pipeline.

This module operates purely on pandas dtypes (numeric vs categorical),
never on hardcoded column names, so the exact same class works unchanged
on IEEE-CIS, PaySim, or ULB once their loaders hand it a DataFrame with
a standardized "target" column.

Pipeline order (leakage-safe):
    1. Separate features (X) from target (y).
    2. Train/test split FIRST, using stratification on the target.
    3. Drop high-missing-ratio columns, learned from TRAIN ONLY.
    4. Impute missing values, fit on TRAIN ONLY, applied to both splits.
    5. Encode categorical columns, fit on TRAIN ONLY.
    6. Scale numeric columns, fit on TRAIN ONLY.
    7. Apply SMOTE to the TRAINING SPLIT ONLY (test data must reflect
       real-world class imbalance, never synthetic oversampling).

Every fitted object (imputers, encoder, scaler, column manifests) is
returned so the calling module can persist them independently -- this
is what makes the exact same transformation reproducible at inference
time or inside a future stacking ensemble.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingArtifacts:
    """
    Container for every fitted preprocessing object plus the column
    manifests needed to reproduce the exact same transformation on new,
    unseen data (e.g., at inference time or for a different model in
    the future stacking ensemble).
    """
    numeric_imputer: SimpleImputer
    categorical_imputer: SimpleImputer
    ordinal_encoder: OrdinalEncoder
    scaler: StandardScaler
    numeric_columns: List[str]
    categorical_columns: List[str]
    dropped_columns: List[str]
    final_feature_order: List[str] = field(default_factory=list)


class FraudPreprocessor:
    """
    Encapsulates the full preprocessing pipeline for tabular fraud data.

    Usage
    -----
    preprocessor = FraudPreprocessor(config)
    X_train, X_test, y_train, y_test, artifacts = preprocessor.run(df)
    """

    def __init__(self, config: dict):
        self.config = config["preprocessing"]
        self.random_seed = config["project"]["random_seed"]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame):
        """
        Execute the full leakage-safe preprocessing pipeline.

        Returns
        -------
        X_train_resampled, X_val, X_test, y_train_resampled, y_val, y_test, artifacts

        Notes
        -----
        X_val/y_val is carved out of the training split BEFORE SMOTE is
        applied, so it contains only real (non-synthetic) samples. It is
        intended for early stopping during model training -- monitoring
        performance on synthetic data would give a falsely optimistic
        stopping signal.
        """
        logger.info("Starting preprocessing pipeline")

        X = df.drop(columns=["target"])
        y = df["target"].astype(int)

        X_train, X_test, y_train, y_test = self._split(X, y)
        logger.info(
            f"Train/test split -> train: {X_train.shape}, test: {X_test.shape}"
        )

        # Carve a validation set out of the training split, BEFORE SMOTE,
        # so early-stopping is evaluated on real, non-synthetic data.
        val_size = self.config.get("validation_size", 0.1)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            random_state=self.random_seed,
            stratify=y_train if self.config.get("stratify", True) else None,
        )
        logger.info(f"Carved validation split from training data: {X_val.shape}")

        X_train, X_val, X_test, dropped_columns = self._drop_high_missing_columns(
            X_train, X_val, X_test
        )

        numeric_columns, categorical_columns = self._identify_column_types(X_train)
        logger.info(
            f"Identified {len(numeric_columns)} numeric and "
            f"{len(categorical_columns)} categorical columns"
        )

        X_train, X_val, X_test, numeric_imputer, categorical_imputer = self._impute_missing_values(
            X_train, X_val, X_test, numeric_columns, categorical_columns
        )

        X_train, X_val, X_test, ordinal_encoder = self._encode_categorical(
            X_train, X_val, X_test, categorical_columns
        )

        X_train, X_val, X_test, scaler = self._scale_numeric(
            X_train, X_val, X_test, numeric_columns
        )

        X_train_resampled, y_train_resampled = self._apply_smote(X_train, y_train)

        artifacts = PreprocessingArtifacts(
            numeric_imputer=numeric_imputer,
            categorical_imputer=categorical_imputer,
            ordinal_encoder=ordinal_encoder,
            scaler=scaler,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            dropped_columns=dropped_columns,
            final_feature_order=list(X_train.columns),
        )

        logger.info("Preprocessing pipeline completed successfully")
        return (
            X_train_resampled,
            X_val,
            X_test,
            y_train_resampled,
            y_val,
            y_test,
            artifacts,
        )

    # ------------------------------------------------------------------
    # Step 1: Train/test split
    # ------------------------------------------------------------------
    def _split(self, X: pd.DataFrame, y: pd.Series):
        stratify_arg = y if self.config.get("stratify", True) else None
        return train_test_split(
            X,
            y,
            test_size=self.config.get("test_size", 0.2),
            random_state=self.random_seed,
            stratify=stratify_arg,
        )

    # ------------------------------------------------------------------
    # Step 2: Drop columns with excessive missingness (fit on train only)
    # ------------------------------------------------------------------
    def _drop_high_missing_columns(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
    ):
        threshold = self.config["missing_values"].get("drop_columns_missing_threshold", 0.95)
        missing_ratio = X_train.isnull().mean()
        columns_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

        if columns_to_drop:
            logger.info(
                f"Dropping {len(columns_to_drop)} columns exceeding "
                f"{threshold * 100:.0f}% missing values (learned from train split)"
            )
            X_train = X_train.drop(columns=columns_to_drop)
            X_val = X_val.drop(columns=columns_to_drop)
            X_test = X_test.drop(columns=columns_to_drop)

        return X_train, X_val, X_test, columns_to_drop

    # ------------------------------------------------------------------
    # Step 3: Identify numeric vs categorical columns
    # ------------------------------------------------------------------
    @staticmethod
    def _identify_column_types(X_train: pd.DataFrame) -> Tuple[List[str], List[str]]:
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        return numeric_columns, categorical_columns

    # ------------------------------------------------------------------
    # Step 4: Missing value imputation (fit on train only)
    # ------------------------------------------------------------------
    def _impute_missing_values(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        numeric_columns: List[str],
        categorical_columns: List[str],
    ):
        cfg = self.config["missing_values"]

        numeric_imputer = SimpleImputer(strategy=cfg.get("numeric_strategy", "median"))
        categorical_imputer = SimpleImputer(
            strategy=cfg.get("categorical_strategy", "constant"),
            fill_value=cfg.get("categorical_fill_value", "missing"),
        )

        # NOTE: columns are replaced via direct assignment (X[col] = array),
        # never via `.loc[:, cols] = array`. As of pandas 3.x, object/string
        # columns default to the strict nullable "str" dtype, which raises
        # a TypeError when `.loc` assignment tries to write numeric values
        # in place. Direct column replacement creates a fresh Series with
        # the correct inferred dtype instead, sidestepping the issue.
        if numeric_columns:
            logger.info(f"Imputing {len(numeric_columns)} numeric columns "
                        f"using strategy='{cfg.get('numeric_strategy', 'median')}'")
            X_train = X_train.copy()
            X_val = X_val.copy()
            X_test = X_test.copy()
            train_values = numeric_imputer.fit_transform(X_train[numeric_columns])
            val_values = numeric_imputer.transform(X_val[numeric_columns])
            test_values = numeric_imputer.transform(X_test[numeric_columns])
            for i, col in enumerate(numeric_columns):
                X_train[col] = train_values[:, i].astype(float)
                X_val[col] = val_values[:, i].astype(float)
                X_test[col] = test_values[:, i].astype(float)

        if categorical_columns:
            logger.info(f"Imputing {len(categorical_columns)} categorical columns "
                        f"using strategy='{cfg.get('categorical_strategy', 'constant')}'")
            train_values = categorical_imputer.fit_transform(X_train[categorical_columns])
            val_values = categorical_imputer.transform(X_val[categorical_columns])
            test_values = categorical_imputer.transform(X_test[categorical_columns])
            for i, col in enumerate(categorical_columns):
                X_train[col] = pd.Series(train_values[:, i], index=X_train.index, dtype=object)
                X_val[col] = pd.Series(val_values[:, i], index=X_val.index, dtype=object)
                X_test[col] = pd.Series(test_values[:, i], index=X_test.index, dtype=object)

        return X_train, X_val, X_test, numeric_imputer, categorical_imputer

    # ------------------------------------------------------------------
    # Step 5: Categorical encoding (fit on train only)
    # ------------------------------------------------------------------
    def _encode_categorical(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        categorical_columns: List[str],
    ):
        if not categorical_columns:
            logger.info("No categorical columns to encode")
            return X_train, X_val, X_test, None

        unknown_value = self.config["encoding"].get("unknown_value", -1)
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=unknown_value
        )

        logger.info(
            f"Encoding {len(categorical_columns)} categorical columns using "
            f"OrdinalEncoder (tree-model-friendly; avoids high-cardinality "
            f"one-hot explosion typical of IEEE-CIS's card/email/device fields)"
        )
        train_encoded = encoder.fit_transform(X_train[categorical_columns])
        val_encoded = encoder.transform(X_val[categorical_columns])
        test_encoded = encoder.transform(X_test[categorical_columns])

        # Direct column replacement, not `.loc` assignment -- see note in
        # _impute_missing_values regarding pandas 3.x strict string dtype.
        for i, col in enumerate(categorical_columns):
            X_train[col] = train_encoded[:, i].astype(float)
            X_val[col] = val_encoded[:, i].astype(float)
            X_test[col] = test_encoded[:, i].astype(float)

        return X_train, X_val, X_test, encoder

    # ------------------------------------------------------------------
    # Step 6: Numeric feature scaling (fit on train only)
    # ------------------------------------------------------------------
    def _scale_numeric(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        numeric_columns: List[str],
    ):
        if not self.config["scaling"].get("apply", True) or not numeric_columns:
            logger.info("Skipping feature scaling")
            return X_train, X_val, X_test, None

        # Note: XGBoost's tree-based splits are scale-invariant, so scaling
        # is not strictly required for THIS model. It is included here
        # because the exact same preprocessed arrays will later feed a
        # Logistic Regression meta-learner / Autoencoder in the stacking
        # ensemble, both of which are scale-sensitive. Keeping preprocessing
        # shared and consistent across all base models avoids maintaining
        # divergent pipelines per model.
        scaler = StandardScaler()
        logger.info(f"Scaling {len(numeric_columns)} numeric columns using StandardScaler")
        train_scaled = scaler.fit_transform(X_train[numeric_columns])
        val_scaled = scaler.transform(X_val[numeric_columns])
        test_scaled = scaler.transform(X_test[numeric_columns])

        for i, col in enumerate(numeric_columns):
            X_train[col] = train_scaled[:, i]
            X_val[col] = val_scaled[:, i]
            X_test[col] = test_scaled[:, i]

        return X_train, X_val, X_test, scaler

    # ------------------------------------------------------------------
    # Step 7: SMOTE (training split only)
    # ------------------------------------------------------------------
    def _apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series):
        smote_cfg = self.config["imbalance"]
        if not smote_cfg.get("apply", True):
            logger.info("SMOTE disabled in config; returning original training split")
            return X_train, y_train

        original_counts = y_train.value_counts().to_dict()
        logger.info(f"Class distribution before SMOTE: {original_counts}")

        smote = SMOTE(
            sampling_strategy=smote_cfg.get("sampling_strategy", 0.3),
            k_neighbors=smote_cfg.get("k_neighbors", 5),
            random_state=self.random_seed,
        )
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        resampled_counts = pd.Series(y_resampled).value_counts().to_dict()
        logger.info(f"Class distribution after SMOTE: {resampled_counts}")

        return X_resampled, y_resampled
