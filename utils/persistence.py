"""
utils/persistence.py

Handles persistence of preprocessing artifacts.

Each fitted preprocessing object (numeric imputer, categorical imputer,
ordinal encoder, scaler) is saved as its OWN joblib file rather than one
bundled pickle. This is deliberate:

    - Individual artifacts can be inspected, versioned, or swapped
      independently (e.g., re-fitting only the scaler for a new dataset).
    - A future stacking ensemble module can selectively reuse specific
      artifacts (e.g., the same encoder for a Random Forest model)
      without deserializing the entire preprocessing bundle.

A manifest.json ties every artifact filename together with the column
manifests needed to reproduce the transformation exactly.
"""

import json
from pathlib import Path
from typing import Any, Dict

import joblib

from preprocessing.preprocessor import PreprocessingArtifacts
from utils.logger import get_logger

logger = get_logger(__name__)


def save_preprocessing_artifacts(
    artifacts: PreprocessingArtifacts, output_dir: Path, dataset_name: str
) -> Dict[str, str]:
    """
    Save every preprocessing object as a separate joblib file plus a
    JSON manifest describing column roles and file locations.

    Returns
    -------
    dict mapping artifact name -> saved file path (as string)
    """
    output_dir = Path(output_dir) / "preprocessing" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    file_map = {}

    def _save(obj: Any, name: str) -> None:
        if obj is None:
            logger.info(f"Skipping save for '{name}' (not used in this pipeline run)")
            return
        path = output_dir / f"{name}.joblib"
        joblib.dump(obj, path)
        file_map[name] = str(path)
        logger.info(f"Saved preprocessing artifact '{name}' to {path}")

    _save(artifacts.numeric_imputer, "numeric_imputer")
    _save(artifacts.categorical_imputer, "categorical_imputer")
    _save(artifacts.ordinal_encoder, "ordinal_encoder")
    _save(artifacts.scaler, "scaler")

    manifest = {
        "dataset_name": dataset_name,
        "artifact_files": file_map,
        "numeric_columns": artifacts.numeric_columns,
        "categorical_columns": artifacts.categorical_columns,
        "dropped_columns": artifacts.dropped_columns,
        "final_feature_order": artifacts.final_feature_order,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    logger.info(f"Preprocessing manifest saved to {manifest_path}")
    return file_map


def load_preprocessing_manifest(output_dir: Path, dataset_name: str) -> Dict[str, Any]:
    """Load the manifest.json describing a previously saved preprocessing run."""
    manifest_path = Path(output_dir) / "preprocessing" / dataset_name / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No preprocessing manifest found at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
