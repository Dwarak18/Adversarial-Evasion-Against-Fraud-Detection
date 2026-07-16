"""
utils/logger.py

Centralized logging configuration for the entire project.

Every module obtains its logger via `get_logger(__name__)` so that log
records are traceable to their source module. Logs are written both to
the console (for live feedback during training) and to a timestamped
file under `logs/`, giving a persistent audit trail of every pipeline run
-- important for a security-oriented project where reproducibility and
traceability of experiments matters.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

# A single timestamp is generated at import time so that every logger
# created during one execution of the pipeline writes to the SAME log
# file, rather than a new file per module.
_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = _LOG_DIR / f"run_{_RUN_TIMESTAMP}.log"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def _initialize_root_logging() -> None:
    """Configure the root logger exactly once per process."""
    global _initialized
    if _initialized:
        return

    os.makedirs(_LOG_DIR, exist_ok=True)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-scoped logger.

    Parameters
    ----------
    name : str
        Typically passed as `__name__` from the calling module.

    Returns
    -------
    logging.Logger
    """
    _initialize_root_logging()
    return logging.getLogger(name)
