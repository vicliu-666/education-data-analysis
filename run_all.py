from __future__ import annotations

import argparse
from pathlib import Path
import json

from run_supervised import run_supervised
from run_unsupervised import run_unsupervised


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/merged_dataset.csv")
    parser.add_argument("--out", type=str, default="outputs/figures")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--sample_n", type=int, default=800)
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sup = run_supervised(data_path, out_dir, random_state=args.seed)
    unsup = run_unsupervised(data_path, out_dir, seed=args.seed, k_chosen=args.k, sample_n=args.sample_n)

    summary = {"supervised": sup, "unsupervised": unsup}
    (out_dir.parent / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n✅ 完成：監督式 + 非監督式 的圖表已輸出到", out_dir.resolve())
    print("✅ 指標摘要已輸出到", (out_dir.parent / "run_summary.json").resolve())


if __name__ == "__main__":
    main()
