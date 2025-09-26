#!/usr/bin/env python3

import os
import sys
import subprocess

REPO_URL = os.environ.get("REPO_URL", "https://github.com/aurumco/janus.git")
REPO_BRANCH = os.environ.get("REPO_BRANCH", "main")

# Preferred working root on Kaggle
def _get_root() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Running inside a Kaggle notebook cell (no __file__)
        return os.getcwd()

ROOT = _get_root()
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
        print("[setup] Cloning repository...")
        try:
            subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, WORK_ROOT], check=True)
        except Exception as e:
            print("[error] Git clone failed. In Kaggle, enable Internet (Settings > Internet) and retry.")
            raise
    else:
        # Update to latest commit
        try:
            print("[setup] Updating repository to latest", REPO_BRANCH)
            subprocess.run(["git", "-C", WORK_ROOT, "fetch", "--all", "--quiet"], check=True)
            subprocess.run(["git", "-C", WORK_ROOT, "reset", "--hard", f"origin/{REPO_BRANCH}"], check=True)
        except Exception as e:
            print(f"[warn] Repo update failed: {e}. Continuing with existing files.")
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
# Silence and offline W&B (no login prompt, no network)
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_MODE", "offline")

# Minimal dep install (best effort) — do NOT reinstall torch, keep Kaggle's CUDA build intact
if os.path.exists(REQ_PATH):
    try:
        print("[setup] Installing requirements...")
        filtered_req = "/kaggle/working/requirements.txt"
        with open(REQ_PATH, "r") as rf, open(filtered_req, "w") as wf:
            for line in rf:
                pkg = line.strip()
                if not pkg or pkg.startswith("#"):
                    continue
                if pkg.split("[")[0].split("==")[0].strip().lower() == "torch":
                    continue
                wf.write(line)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", filtered_req, "--quiet", "--no-input", "--no-deps"], check=True)
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

# Brief device info
try:
    import torch
    print(f"[info] CUDA available: {torch.cuda.is_available()} | devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
except Exception:
    pass

# Launch training
print("[run] Starting HRM training on Kaggle...")
try:
    os.chdir(HRM_DIR)
    sys.path.insert(0, ".")
    from pretrain import main as train_main
    train_main()
except Exception as e:
    print("[error] Training failed:", e)
    raise
print("[done] Training finished. Artifacts saved under:")
print(" - /kaggle/working/checkpoints/janus_v4/")
print(" - /kaggle/working/results/<timestamp>/")
