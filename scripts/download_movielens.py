import zipfile
import urllib.request
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
RAW_DIR = Path("data/raw")
ZIP_PATH = RAW_DIR / "ml-1m.zip"

RAW_DIR.mkdir(parents=True, exist_ok=True)

if not ZIP_PATH.exists():
    print("Downloading MovieLens 1M dataset...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
else:
    print("Zip file already exists, skipping download.")

print("Extracting...")
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(RAW_DIR)

print("Done. Dataset available at data/raw/ml-1m")
