#!/usr/bin/env python3

import os
import sys
import subprocess

REPO_URL = os.environ.get("REPO_URL", "https://github.com/aurumco/janus.git")

# Preferred working root on Kaggle
ROOT = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = "/kaggle/working/janus"

def ensure_code_available() -> str:
    """Return path to repo root with HRM/ inside. Clone if missing."""
    # If this script is already inside a repo root with HRM/, use it
    local_hrm = os.path.join(ROOT, "HRM")
    if os.path.isdir(local_hrm):
        return ROOT
    # Else, clone into /kaggle/working/janus
    os.makedirs("/kaggle/working", exist_ok=True)
    if not os.path.isdir(WORK_ROOT):
        print("[setup] Cloning repository... (ensure Internet is enabled)")
        try:
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, WORK_ROOT], check=True)
        except Exception as e:
            print("[error] Git clone failed. In Kaggle, enable Internet (Settings > Internet) and retry.")
            raise
    return WORK_ROOT

repo_root = ensure_code_available()
HRM_DIR = os.path.join(repo_root, "HRM")
REQ_PATH = os.path.join(HRM_DIR, "requirements.txt")

# Kaggle paths
DATA_PARQUET = "/kaggle/input/janus-m15-dataset/janus_m15_dataset.parquet"
OUTPUT_DIR = "/kaggle/working"

# Environment for clean/minimal run
os.environ.setdefault("DATA_PATH", DATA_PARQUET)
os.environ.setdefault("OUTPUT_DIR", OUTPUT_DIR)
# Silence CUDA probing (use CPU by default to avoid capability warnings)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# Silence and offline W&B (no login prompt, no network)
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_MODE", "offline")

# Minimal dep install (best effort)
if os.path.exists(REQ_PATH):
    try:
        print("[setup] Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", REQ_PATH, "--quiet", "--no-input"], check=True)
    except Exception as e:
        print(f"[warn] Requirements install returned an error: {e}")

# Validate dataset path
if not os.path.exists(DATA_PARQUET):
    print("[error] Dataset parquet not found:", DATA_PARQUET)
    print("[hint] Add the dataset to your Kaggle Notebook: /kaggle/input/janus-m15-dataset")
    # List inputs to help debugging
    for d, _, fns in os.walk("/kaggle/input"):
        for fn in fns:
            print(os.path.join(d, fn))
    raise SystemExit(1)

# Launch training
print("[run] Starting HRM training on Kaggle...")
try:
    sys.path.insert(0, HRM_DIR)
    from pretrain import main as train_main
    train_main()
except Exception as e:
    print("[error] Training failed:", e)
    raise
print("[done] Training finished. Artifacts saved under:")
print(" - /kaggle/working/checkpoints/janus_v4/")
print(" - /kaggle/working/results/<timestamp>/")
