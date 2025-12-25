from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class SupervisedConfig:
    target: str = "FinalGrade"
    drop_cols: Tuple[str, ...] = ("ExamScore", "FinalGrade")  # Version B: do NOT use ExamScore or label as feature
    test_size: float = 0.2
    random_state: int = 42


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到資料檔：{csv_path}\n"
            f"請把 merged_dataset.csv 放到 data/merged_dataset.csv，或用 --data 指定路徑。"
        )
    df = pd.read_csv(csv_path)
    return df


def split_Xy(df: pd.DataFrame, target: str, drop_cols: Iterable[str]) -> Tuple[pd.DataFrame, pd.Series]:
    drop_cols = list(drop_cols)
    # Ensure target exists
    if target not in df.columns:
        raise ValueError(f"target 欄位不存在：{target}")

    # X: remove target + specified leak-prone cols
    X = df.drop(columns=[c for c in [target, *drop_cols] if c in df.columns], errors="ignore")
    y = df[target].astype(int)
    return X, y


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str) -> None:
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
