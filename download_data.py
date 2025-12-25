"""
download_data.py

本作業資料來自 Zenodo（merged_dataset.csv）。
因為課堂環境可能沒有網路/權限，本腳本不強制下載。
請手動將 merged_dataset.csv 放到 data/merged_dataset.csv。
"""
from pathlib import Path

def main():
    p = Path("data/merged_dataset.csv")
    if p.exists():
        print("✅ 已存在：", p.resolve())
    else:
        print("❗ 請把 merged_dataset.csv 放到：", p.resolve())

if __name__ == "__main__":
    main()
