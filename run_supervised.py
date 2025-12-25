from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from utils import load_dataset, split_Xy, save_fig, plot_confusion_matrix, SupervisedConfig


def run_supervised(csv_path: Path, out_dir: Path, random_state: int = 42) -> dict:
    """
    Version B 監督式學習：
    - 目標：FinalGrade (0/1/2/3)
    - 特徵：不含 ExamScore、FinalGrade
    - 模型：Logistic Regression (multinomial) + Random Forest
    - 指標：Accuracy、Macro-F1、ROC-AUC (OvR)
    - 圖：Confusion Matrix、ROC curves、RF feature importance
    """
    df = load_dataset(csv_path)

    cfg = SupervisedConfig(random_state=random_state)
    X, y = split_Xy(df, target=cfg.target, drop_cols=cfg.drop_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )

    class_names = [str(c) for c in sorted(np.unique(y))]

    # 1) Logistic Regression (with scaling)
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                multi_class="multinomial",
                max_iter=2000,
                n_jobs=None,
                random_state=cfg.random_state,
            )),
        ]
    )
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)

    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr, average="macro")
    auc_lr = roc_auc_score(y_test, y_proba_lr, multi_class="ovr", average="macro")

    # ROC curves (OvR) - plot all classes on the same axes (to match PPT)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    classes = sorted(np.unique(y))
    for i, cls in enumerate(classes):
        RocCurveDisplay.from_predictions(
            (y_test == cls).astype(int),
            y_proba_lr[:, i],
            name=f"class {cls}",
            ax=ax,
        )
    ax.set_title("Logistic Regression ROC (OvR)")
    save_fig(out_dir / "lr_roc_ovr.png")

    # 2) Random Forest
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=cfg.random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average="macro")
    auc_rf = roc_auc_score(y_test, y_proba_rf, multi_class="ovr", average="macro")

    # Confusion matrix (use RF as the main)
    cm = confusion_matrix(y_test, y_pred_rf, labels=sorted(np.unique(y)))
    plot_confusion_matrix(cm, class_names, title=f"Version B Random Forest Confusion Matrix (acc={acc_rf:.3f})")
    save_fig(out_dir / "rf_confusion_matrix.png")

    # Feature importance (Top 12)
    importances = rf.feature_importances_
    feat = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(12)

    plt.figure(figsize=(8, 5))
    plt.barh(feat.index[::-1], feat.values[::-1])
    plt.title("Version B (No ExamScore) - Top Feature Importances")
    plt.xlabel("Importance")
    save_fig(out_dir / "rf_feature_importance_top12.png")

    metrics = {
        "logistic_regression": {"accuracy": float(acc_lr), "macro_f1": float(f1_lr), "roc_auc_ovr_macro": float(auc_lr)},
        "random_forest": {"accuracy": float(acc_rf), "macro_f1": float(f1_rf), "roc_auc_ovr_macro": float(auc_rf)},
        "classification_report_rf": classification_report(y_test, y_pred_rf, digits=3),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features_used": list(X.columns),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/merged_dataset.csv", help="Path to merged_dataset.csv")
    parser.add_argument("--out", type=str, default="outputs/figures", help="Output folder for figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_supervised(csv_path, out_dir, random_state=args.seed)

    # Print a concise summary
    print("=== Supervised (Version B) Summary ===")
    print(f"RF Accuracy={metrics['random_forest']['accuracy']:.3f} | Macro-F1={metrics['random_forest']['macro_f1']:.3f} | AUC={metrics['random_forest']['roc_auc_ovr_macro']:.3f}")
    print(f"LR Accuracy={metrics['logistic_regression']['accuracy']:.3f} | Macro-F1={metrics['logistic_regression']['macro_f1']:.3f} | AUC={metrics['logistic_regression']['roc_auc_ovr_macro']:.3f}")
    print("\nSaved figures to:", out_dir.resolve())
    print("\nRF classification report:\n", metrics["classification_report_rf"])


if __name__ == "__main__":
    main()
