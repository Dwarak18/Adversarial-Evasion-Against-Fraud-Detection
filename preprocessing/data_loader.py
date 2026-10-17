"""
preprocessing/data_loader.py

Dataset loading layer.

Design rationale
-----------------
This module defines an abstract `BaseDatasetLoader` so that the rest of
the pipeline (preprocessor, trainer, evaluator) never needs to know
which raw dataset it is working with. Every loader's `load()` method
returns a single pandas DataFrame with:

    1. All raw feature columns intact.
    2. A standardized binary target column, always renamed to "target".
    3. Identifier / non-predictive columns already dropped.

Today only `IEEECISLoader` is implemented, matching this task's scope.
`PaySimLoader` and `ULBLoader` are stubbed with NotImplementedError so
the architecture is visibly ready for them without pretending they
exist yet.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class BaseDatasetLoader(ABC):
    """
    Abstract base class every dataset loader must implement.

    Enforcing this interface is what allows preprocessing, training, and
    evaluation code to remain dataset-agnostic: they only ever call
    `.load()` and expect a DataFrame with a "target" column.
    """

    def __init__(self, raw_data_dir: Path, dataset_config: Dict[str, Any]):
        self.raw_data_dir = Path(raw_data_dir)
        self.dataset_config = dataset_config

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return a single merged DataFrame with a standardized 'target' column."""
        raise NotImplementedError


class IEEECISLoader(BaseDatasetLoader):
    """
    Loader for the IEEE-CIS Fraud Detection dataset (Kaggle).

    The raw dataset ships as two files that must be merged on
    TransactionID:
        - train_transaction.csv  (transaction-level features + label)
        - train_identity.csv     (device/identity features, sparse join)

    Not every transaction has a matching identity record, so a LEFT JOIN
    on the transaction table is used to avoid silently dropping rows.
    """

    def load(self) -> pd.DataFrame:
        transaction_path = self.raw_data_dir / self.dataset_config["transaction_file"]
        identity_path = self.raw_data_dir / self.dataset_config["identity_file"]
        merge_key = self.dataset_config["merge_key"]
        target_column = self.dataset_config["target_column"]
        drop_columns = self.dataset_config.get("drop_columns", [])

        if not transaction_path.exists():
            raise FileNotFoundError(
                f"IEEE-CIS transaction file not found at {transaction_path}. "
                f"Download it from Kaggle and place it under data/raw/."
            )

        logger.info(f"Loading transaction data from {transaction_path}")
        transactions = pd.read_csv(transaction_path)
        logger.info(f"Transaction data shape: {transactions.shape}")

        if identity_path.exists():
            logger.info(f"Loading identity data from {identity_path}")
            identity = pd.read_csv(identity_path)
            logger.info(f"Identity data shape: {identity.shape}")

            logger.info(f"Merging transaction and identity tables on '{merge_key}'")
            merged = transactions.merge(identity, on=merge_key, how="left")
        else:
            logger.warning(
                f"Identity file not found at {identity_path}. "
                f"Proceeding with transaction-only features."
            )
            merged = transactions

        if target_column not in merged.columns:
            raise KeyError(
                f"Expected target column '{target_column}' not found in dataset. "
                f"Available columns (first 20): {list(merged.columns)[:20]}"
            )

        # Standardize the target column name across all future dataset loaders.
        merged = merged.rename(columns={target_column: "target"})

        existing_drop_columns = [c for c in drop_columns if c in merged.columns]
        if existing_drop_columns:
            logger.info(f"Dropping identifier columns: {existing_drop_columns}")
            merged = merged.drop(columns=existing_drop_columns)

        logger.info(f"Final merged dataset shape: {merged.shape}")
        logger.info(
            f"Class balance -> fraud: {merged['target'].sum()} "
            f"({merged['target'].mean() * 100:.3f}%), "
            f"legitimate: {(merged['target'] == 0).sum()}"
        )

        return merged


class PaySimLoader(BaseDatasetLoader):
    """
    Loader for the PaySim synthetic dataset.
    """

    def load(self) -> pd.DataFrame:
        transaction_path = self.raw_data_dir / self.dataset_config["transaction_file"]
        target_column = self.dataset_config["target_column"]
        drop_columns = self.dataset_config.get("drop_columns", [])

        if not transaction_path.exists():
            raise FileNotFoundError(
                f"PaySim file not found at {transaction_path}. "
                f"Download it and place it under data/raw/."
            )

        logger.info(f"Loading PaySim data from {transaction_path}")
        df = pd.read_csv(transaction_path)
        logger.info(f"PaySim data shape: {df.shape}")

        if target_column not in df.columns:
            raise KeyError(f"Expected target column '{target_column}' not found.")

        df = df.rename(columns={target_column: "target"})

        existing_drop_columns = [c for c in drop_columns if c in df.columns]
        if existing_drop_columns:
            logger.info(f"Dropping identifier columns: {existing_drop_columns}")
            df = df.drop(columns=existing_drop_columns)

        return df


class ULBLoader(BaseDatasetLoader):
    """
    Loader for the ULB Credit Card Fraud dataset.
    """

    def load(self) -> pd.DataFrame:
        transaction_path = self.raw_data_dir / self.dataset_config["transaction_file"]
        target_column = self.dataset_config["target_column"]
        drop_columns = self.dataset_config.get("drop_columns", [])

        if not transaction_path.exists():
            raise FileNotFoundError(
                f"ULB file not found at {transaction_path}. "
                f"Download it and place it under data/raw/."
            )

        logger.info(f"Loading ULB data from {transaction_path}")
        df = pd.read_csv(transaction_path)
        logger.info(f"ULB data shape: {df.shape}")

        if target_column not in df.columns:
            raise KeyError(f"Expected target column '{target_column}' not found.")

        df = df.rename(columns={target_column: "target"})

        existing_drop_columns = [c for c in drop_columns if c in df.columns]
        if existing_drop_columns:
            logger.info(f"Dropping identifier columns: {existing_drop_columns}")
            df = df.drop(columns=existing_drop_columns)

        return df


# Registry mapping config-declared loader names to their classes, used by
# the pipeline to instantiate the correct loader dynamically based on
# config.yaml's `active_dataset` setting.
LOADER_REGISTRY = {
    "IEEECISLoader": IEEECISLoader,
    "PaySimLoader": PaySimLoader,
    "ULBLoader": ULBLoader,
}


def get_loader(raw_data_dir: Path, dataset_config: Dict[str, Any]) -> BaseDatasetLoader:
    """Factory function that instantiates the correct loader from config."""
    loader_name = dataset_config["loader"]
    if loader_name not in LOADER_REGISTRY:
        raise ValueError(
            f"Unknown loader '{loader_name}'. "
            f"Available loaders: {list(LOADER_REGISTRY.keys())}"
        )
    loader_class = LOADER_REGISTRY[loader_name]
    logger.info(f"Instantiating loader: {loader_name}")
    return loader_class(raw_data_dir=raw_data_dir, dataset_config=dataset_config)
