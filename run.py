import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd, title):
    print(f"\n=== {title} ===", flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\nStep failed: {title}. Read the error above.")


if "--no-install" not in sys.argv:
    run([PY, "-m", "pip", "install", "-r", "requirements.txt"], "Installing packages")

if "--no-train" not in sys.argv:
    run([PY, "preprocess.py"], "Step 1: cleaning and splitting data")
    run([PY, "train_models.py"], "Step 2: training and comparing models (the SVM takes ~2-3 minutes)")
    run([PY, "clustering.py"], "Step 2b: exploratory K-Means clustering")

try:
    import streamlit  
except ImportError:
    sys.exit("\nStreamlit is not installed. Run: python run.py (without --no-install)")

run([PY, "-m", "streamlit", "run", "app/app.py"], "Step 3: starting the web app (Ctrl+C to stop)")