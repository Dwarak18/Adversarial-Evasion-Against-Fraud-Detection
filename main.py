"""
main.py

Entry point for the fraud detection training pipeline.

Usage
-----
    python main.py
    python main.py --config configs/config.yaml
    python main.py --dataset ieee_cis

This script intentionally contains no business logic of its own -- it
only parses arguments, loads configuration, optionally overrides the
active dataset, and delegates to training/train_xgboost.py. Keeping
main.py thin means the same pipeline can later be triggered from a
notebook, a scheduled job, or an ensemble-training script without
duplicating orchestration logic.
"""

import argparse
import sys
from pathlib import Path

from training.train_xgboost import run_training_pipeline
from utils.config_loader import DEFAULT_CONFIG_PATH, load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the XGBoost fraud detection module."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["ieee_cis", "paysim", "ulb"],
        help="Override the active_dataset value from config.yaml.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        config = load_config(Path(args.config))

        if args.dataset:
            logger.info(
                f"Overriding active_dataset: "
                f"'{config['active_dataset']}' -> '{args.dataset}'"
            )
            config["active_dataset"] = args.dataset

        metrics = run_training_pipeline(config)

        logger.info("Pipeline finished successfully.")
        logger.info(
            f"Final metrics summary: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )
        return 0

    except FileNotFoundError as exc:
        logger.error(f"Required file not found: {exc}")
        return 1
    except KeyError as exc:
        logger.error(f"Configuration error: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level safety net, logged with context
        logger.exception(f"Pipeline failed with an unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
