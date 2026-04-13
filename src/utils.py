"""
Utility helpers: logging, saving tables/figures, directory setup.
"""

import logging
import shutil
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CLI
import matplotlib.pyplot as plt
import pandas as pd

from src.config import TABLES_DIR, FIGURES_DIR, MODELS_DIR, LOGS_DIR, OUTPUT_DIR


# ─── Directory setup ────────────────────────────────────────────────
def ensure_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in [TABLES_DIR, FIGURES_DIR, MODELS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def clean_output_dirs() -> None:
    """Remove all files from output directories so each run starts fresh."""
    for d in [TABLES_DIR, FIGURES_DIR, MODELS_DIR, LOGS_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


# ─── Logging ────────────────────────────────────────────────────────
def get_logger(name: str = "ml_project") -> logging.Logger:
    """Return a logger that writes to console and to a log file."""
    ensure_dirs()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(LOGS_DIR / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─── Saving helpers ─────────────────────────────────────────────────
def save_table(df: pd.DataFrame, name: str, index: bool = True) -> Path:
    """Save a DataFrame as CSV under outputs/tables/."""
    ensure_dirs()
    path = TABLES_DIR / name
    df.to_csv(path, index=index)
    return path


def save_figure(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    """Save a matplotlib figure under outputs/figures/."""
    ensure_dirs()
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_text(content: str, name: str, subdir: Optional[str] = "logs") -> Path:
    """Save a text/markdown file under outputs/<subdir>/."""
    ensure_dirs()
    base = OUTPUT_DIR / subdir if subdir else OUTPUT_DIR
    base.mkdir(parents=True, exist_ok=True)
    path = base / name
    path.write_text(content, encoding="utf-8")
    return path


# ─── Timer decorator ────────────────────────────────────────────────
def timer(func):
    """Decorator that logs execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        logger.info(f"▶ Starting {func.__name__}")
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        logger.info(f"✔ Finished {func.__name__} in {elapsed:.1f}s")
        return result
    return wrapper
