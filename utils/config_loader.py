"""
utils/config_loader.py

Loads `configs/config.yaml` into a Python dictionary and resolves all
relative paths against the project root. Centralizing this logic means
every module (loader, preprocessor, trainer, evaluator) reads
configuration identically, avoiding path-resolution bugs when the
pipeline is invoked from different working directories.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

from utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load the YAML configuration file and resolve filesystem paths.

    Parameters
    ----------
    config_path : Path
        Location of the config.yaml file. Defaults to configs/config.yaml
        at the project root.

    Returns
    -------
    dict
        Parsed configuration with an added "_resolved_paths" key
        containing absolute Path objects for every entry under `paths`.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve every path in the `paths` block to an absolute Path object,
    # anchored at the project root, regardless of the current working
    # directory the script is launched from.
    resolved_paths = {
        key: (PROJECT_ROOT / value).resolve()
        for key, value in config.get("paths", {}).items()
    }
    config["_resolved_paths"] = resolved_paths
    config["_project_root"] = PROJECT_ROOT

    logger.debug(f"Resolved paths: {resolved_paths}")
    return config


def get_active_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the configuration block for whichever dataset is currently
    active, as selected by `active_dataset` in config.yaml.
    """
    active_key = config["active_dataset"]
    try:
        dataset_config = config["datasets"][active_key]
    except KeyError as exc:
        raise KeyError(
            f"active_dataset '{active_key}' has no matching entry under "
            f"'datasets' in config.yaml"
        ) from exc

    dataset_config = dict(dataset_config)  # shallow copy to avoid mutation
    dataset_config["_dataset_name"] = active_key
    return dataset_config
