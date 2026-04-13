"""
Build classification targets from raw ratings.
Four formulations: binary, 3-class balanced, 3-class strict, 4-class.
"""

import numpy as np
import pandas as pd

from src.config import BINARY_THRESHOLD, STRICT_3CLASS_BINS, STRICT_3CLASS_LABELS
from src.utils import get_logger, save_table, save_figure, save_text, timer
from src.visualize import plot_class_distribution


def build_binary_target(y: pd.Series) -> pd.Series:
    """
    Binary classification: low (<threshold) vs high (>=threshold).
    Returns integer labels: 0=low, 1=high.
    """
    return (y >= BINARY_THRESHOLD).astype(int).rename("binary_target")


def build_3class_balanced_target(y: pd.Series) -> pd.Series:
    """
    3-class classification using quantile-based binning for balanced classes.
    Returns labels: 0, 1, 2.
    """
    labels, bins = pd.qcut(y, q=3, labels=False, retbins=True, duplicates="drop")
    return pd.Series(labels, index=y.index, name="3class_balanced_target")


def build_3class_strict_target(y: pd.Series) -> pd.Series:
    """
    3-class classification with fixed boundaries:
    low: 1-4, medium: 5-6, high: 7-10.
    Returns labels: 0=low, 1=medium, 2=high.
    """
    labels = pd.cut(
        y, bins=STRICT_3CLASS_BINS, labels=STRICT_3CLASS_LABELS, include_lowest=True
    )
    return labels.map({"low": 0, "medium": 1, "high": 2}).astype(int).rename("3class_strict_target")


def build_4class_target(y: pd.Series) -> pd.Series:
    """
    4-class classification using quantile-based binning.
    Returns labels: 0, 1, 2, 3.
    """
    labels, bins = pd.qcut(y, q=4, labels=False, retbins=True, duplicates="drop")
    return pd.Series(labels, index=y.index, name="4class_target")


@timer
def build_all_targets(y: pd.Series) -> dict:
    """
    Build all classification targets and save class mappings + distribution plots.

    Returns
    -------
    dict
        Keys are task names, values are pd.Series of integer labels.
    """
    logger = get_logger()

    targets = {}

    # Binary
    targets["binary"] = build_binary_target(y)
    _binary_map = {0: f"low (< {BINARY_THRESHOLD})", 1: f"high (>= {BINARY_THRESHOLD})"}

    # 3-class balanced (quantile)
    targets["3class_balanced"] = build_3class_balanced_target(y)

    # 3-class strict
    targets["3class_strict"] = build_3class_strict_target(y)

    # 4-class
    targets["4class"] = build_4class_target(y)

    # ── Save class distribution tables and plots ────────────────────
    mapping_lines = ["# Classification Target Mappings\n"]

    for task_name, labels in targets.items():
        dist = labels.value_counts().sort_index().rename("count").to_frame()
        dist["pct"] = (dist["count"] / dist["count"].sum() * 100).round(2)

        fig = plot_class_distribution(labels, title=f"Class Distribution — {task_name}")
        save_figure(fig, f"class_distribution_{task_name}.png")

        mapping_lines.append(f"## {task_name}")
        mapping_lines.append(f"- Number of classes: {labels.nunique()}")
        mapping_lines.append("- Value counts:")
        for val in sorted(labels.unique()):
            cnt = (labels == val).sum()
            mapping_lines.append(f"  - Class {val}: {cnt} samples ({cnt/len(labels)*100:.1f}%)")

        # Provide human-readable label details
        if task_name == "binary":
            mapping_lines.append(f"- Rule: low = rating < {BINARY_THRESHOLD}, "
                                 f"high = rating >= {BINARY_THRESHOLD}")
        elif task_name == "3class_strict":
            mapping_lines.append("- Rule: low = 1-4, medium = 5-6, high = 7-10")
        elif task_name == "3class_balanced":
            mapping_lines.append("- Rule: quantile-based 3 equal-frequency bins")
            # Compute and record the actual bin edges
            _, bins = pd.qcut(y, q=3, retbins=True, duplicates="drop")
            mapping_lines.append(f"- Bin edges: {[round(b, 1) for b in bins]}")
        elif task_name == "4class":
            mapping_lines.append("- Rule: quantile-based 4 equal-frequency bins")
            _, bins = pd.qcut(y, q=4, retbins=True, duplicates="drop")
            mapping_lines.append(f"- Bin edges: {[round(b, 1) for b in bins]}")
        mapping_lines.append("")

    save_text("\n".join(mapping_lines), "class_mapping.md")
    logger.info("Saved class_mapping.md and class distribution plots")

    return targets
