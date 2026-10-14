"""Download the Online Retail II dataset (UCI) and unzip it into data/.

Run from the project root:  python data/download_data.py
"""
import zipfile
from pathlib import Path

import requests

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DATA_DIR = Path(__file__).resolve().parent
ZIP_PATH = DATA_DIR / "online_retail_II.zip"
XLSX_PATH = DATA_DIR / "online_retail_II.xlsx"


def main() -> None:
    if XLSX_PATH.exists():
        print(f"Already downloaded: {XLSX_PATH.name}")
        return
    print("Downloading Online Retail II (about 45 MB)...")
    response = requests.get(URL, timeout=120)
    response.raise_for_status()
    ZIP_PATH.write_bytes(response.content)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)
    print(f"Saved {XLSX_PATH.name}")


if __name__ == "__main__":
    main()
