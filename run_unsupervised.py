from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from utils import load_dataset, save_fig, ensure_dir


# Unsupervised features: behavior + environment + psychology + demographics (optional)
UNSUP_FEATURES = [
    "StudyHours", "Attendance", "Resources", "Extracurricular", "Motivation", "Internet",
    "Gender", "Age", "LearningStyle", "OnlineCourses", "Discussions", "AssignmentCompletion",
    "EduTech", "StressLevel",
]
# Exclude: ExamScore, FinalGrade


def elbow_and_silhouette(X_scaled: np.ndarray, k_min: int, k_max: int, seed: int) -> tuple[list[int], list[float], list[float]]:
    ks = list(range(k_min, k_max + 1))
    inertias, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        if k >= 2:
            sils.append(float(silhouette_score(X_scaled, labels)))
        else:
            sils.append(float("nan"))
    return ks, inertias, sils


def zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / (df.std(ddof=0) + 1e-9)


def run_unsupervised(csv_path: Path, out_dir: Path, seed: int = 42, k_chosen: int = 3, sample_n: int = 800) -> dict:
    """
    非監督式學習（K-Means）：
    - 分群特徵不含 ExamScore / FinalGrade（避免邏輯循環）
    - k 選擇：Elbow + Silhouette（抽樣）
    - 視覺化：PCA 2D + 群集輪廓 heatmap
    - post-hoc：ExamScore boxplot、FinalGrade 組成比例（堆疊條）
    """
    df = load_dataset(csv_path)

    # Keep only available columns
    features = [c for c in UNSUP_FEATURES if c in df.columns]
    X = df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # k selection on a sample for speed
    if len(df) > sample_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=sample_n, replace=False)
        X_s = X_scaled[idx]
    else:
        X_s = X_scaled

    ks, inertias, sils = elbow_and_silhouette(X_s, k_min=2, k_max=6, seed=seed)

    # Plot elbow
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(ks, inertias, marker="o")
    plt.title("Elbow Method (sample)")
    plt.xlabel("k")
    plt.ylabel("Inertia (WCSS)")
    plt.axvline(k_chosen, linestyle="--")
    save_fig(out_dir / "kmeans_elbow.png")

    # Plot silhouette
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(ks, sils, marker="o")
    plt.title("Silhouette Score (sample)")
    plt.xlabel("k")
    plt.ylabel("Avg Silhouette")
    plt.axvline(k_chosen, linestyle="--")
    save_fig(out_dir / "kmeans_silhouette.png")

    # Fit final KMeans on full data
    km = KMeans(n_clusters=k_chosen, random_state=seed, n_init=30)
    labels = km.fit_predict(X_scaled)
    df = df.copy()
    df["Cluster"] = labels

    # PCA visualization
    pca = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)
    df["PCA1"] = X_pca[:, 0]
    df["PCA2"] = X_pca[:, 1]

    plt.figure(figsize=(7.5, 5.5))
    for c in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == c]
        plt.scatter(sub["PCA1"], sub["PCA2"], s=10, alpha=0.6, label=f"Cluster {c}")
    plt.title("PCA Visualization of K-Means Clusters")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.legend()
    save_fig(out_dir / "kmeans_pca2d.png")

    # Cluster profile heatmap (z-score of means)
    means = df.groupby("Cluster")[features].mean()
    z = zscore_df(means)

    plt.figure(figsize=(10, 4.5))
    plt.imshow(z.values, aspect="auto")
    plt.colorbar(label="z-score (cluster mean)")
    plt.yticks(range(len(z.index)), [f"C{c}" for c in z.index])
    plt.xticks(range(len(z.columns)), z.columns, rotation=35, ha="right")
    plt.title("Cluster Profile Heatmap (z-score of means)")
    save_fig(out_dir / "cluster_profile_heatmap.png")

    # post-hoc validation: ExamScore boxplot if available
    if "ExamScore" in df.columns:
        plt.figure(figsize=(7, 5))
        data = [df[df["Cluster"] == c]["ExamScore"].values for c in sorted(df["Cluster"].unique())]
        plt.boxplot(data, labels=[f"C{c}" for c in sorted(df["Cluster"].unique())])
        plt.title("Post-hoc: ExamScore by Cluster (not used in clustering)")
        plt.xlabel("Cluster")
        plt.ylabel("ExamScore")
        save_fig(out_dir / "posthoc_examscore_boxplot.png")

    # post-hoc validation: FinalGrade composition if available
    if "FinalGrade" in df.columns:
        comp = pd.crosstab(df["Cluster"], df["FinalGrade"], normalize="index")
        comp = comp.sort_index()
        plt.figure(figsize=(7.5, 5))
        bottom = np.zeros(len(comp))
        for col in comp.columns:
            vals = comp[col].values
            plt.bar(comp.index.astype(str), vals, bottom=bottom, label=str(col))
            bottom += vals
        plt.title("Post-hoc: FinalGrade Composition by Cluster")
        plt.xlabel("Cluster")
        plt.ylabel("Proportion")
        plt.legend(title="FinalGrade")
        save_fig(out_dir / "posthoc_finalgrade_composition.png")

    results = {
        "k_chosen": int(k_chosen),
        "features_used": features,
        "sample_n_for_k_selection": int(min(sample_n, len(df))),
        "inertias": dict(zip(ks, inertias)),
        "silhouette_scores": dict(zip(ks, sils)),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/merged_dataset.csv")
    parser.add_argument("--out", type=str, default="outputs/figures")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--sample_n", type=int, default=800)
    args = parser.parse_args()

    csv_path = Path(args.data)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    res = run_unsupervised(csv_path, out_dir, seed=args.seed, k_chosen=args.k, sample_n=args.sample_n)
    print("=== Unsupervised Summary ===")
    print("k =", res["k_chosen"])
    print("PCA explained variance ratio:", res["pca_explained_variance_ratio"])
    print("Saved figures to:", out_dir.resolve())


if __name__ == "__main__":
    main()
