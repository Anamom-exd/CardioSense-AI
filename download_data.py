"""
Downloads the raw Kaggle "Cardiovascular Disease dataset" (sulianova) into data/raw/.
Source: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
If the download needs a Kaggle login on your machine, download the zip manually from that page and
put cardio_train.csv into data/raw/ under the file name below. Run:  python download_data.py
"""
import hashlib, io, os, sys, urllib.request, zipfile
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
URL = "https://www.kaggle.com/api/v1/datasets/download/sulianova/cardiovascular-disease-dataset"
DEST = Path("data/raw/cardiovascular_disease_kaggle_sulianova_cardio_train.csv")
SHA256 = "21a705d23381b0dfd6a6416da701b490744f1fc3b47e9ff3db3968c420ffa10c"  # file used for the reported results

if DEST.exists() and hashlib.sha256(DEST.read_bytes()).hexdigest() == SHA256:
    sys.exit(f"{DEST} already present and matches the expected checksum.")
try:
    data = urllib.request.urlopen(URL, timeout=60).read()
    raw = zipfile.ZipFile(io.BytesIO(data)).read("cardio_train.csv")
except Exception as e:  # network error, login required, or unexpected archive
    sys.exit(f"Download failed ({e}). Download the zip manually from the Kaggle page in this file's header.")
DEST.parent.mkdir(parents=True, exist_ok=True)
DEST.write_bytes(raw)
ok = hashlib.sha256(raw).hexdigest() == SHA256
print(f"Saved {DEST} ({len(raw):,} bytes). Checksum {'matches' if ok else 'DIFFERS from'} the file used for the reported results.")
