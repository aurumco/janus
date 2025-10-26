#!/usr/bin/env python3
"""Kaggle training script for Mamba Bitcoin price regressor.

This script uses the Strategy Pattern for flexible package installation
across different environments, ensuring clean separation of concerns.
"""

import os
import sys
import subprocess
from pathlib import Path

REPO_URL = os.environ.get("REPO_URL", "https://github.com/aurumco/janus.git")
REPO_BRANCH = os.environ.get("REPO_BRANCH", "main")

DATA_PARQUET = "/kaggle/input/janusds/janusdataset/janus_finetune_30min_dataset.parquet"
OUTPUT_DIR = "/kaggle/working"


def get_root() -> str:
    """Get the root directory of the script."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def ensure_repository() -> str:
    """Clone or update repository if needed.
    
    Always pulls latest code from GitHub on each run to ensure
    code changes are reflected, while preserving installed packages.

    Returns:
        Path to repository root.
    """
    root = get_root()
    work_root = "/kaggle/working/janus"

    if os.path.isdir(os.path.join(root, "src")):
        return root

    os.makedirs("/kaggle/working", exist_ok=True)

    if not os.path.isdir(work_root):
        print("[setup] Cloning repository...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, work_root],
                check=True
            )
            print(f"[setup] Repository cloned successfully")
        except Exception as e:
            print(f"[error] Git clone failed: {e}")
            print("[hint] Enable Internet in Kaggle (Settings > Internet)")
            raise
    else:
        # Always update to latest code on each run
        try:
            print(f"[setup] Pulling latest code from {REPO_BRANCH}...")
            subprocess.run(["git", "-C", work_root, "fetch", "--all"], check=True)
            subprocess.run(["git", "-C", work_root, "reset", "--hard", f"origin/{REPO_BRANCH}"], check=True)
            subprocess.run(["git", "-C", work_root, "clean", "-fd"], check=True)
            print("[setup] ✓ Code updated to latest version")
        except Exception as e:
            print(f"[warn] Repository update failed: {e}. Using existing files.")

    return work_root


def install_requirements(repo_root: str) -> None:
    """Install required packages using Strategy Pattern.

    This function uses the InstallationContext with KaggleInstallationStrategy
    to handle package installation in a clean, maintainable way.

    Args:
        repo_root: Path to repository root.
    """
    req_path = Path(repo_root) / "requirements.txt"

    if not req_path.exists():
        print("[warn] requirements.txt not found, skipping installation")
        return

    # Add src to path to import our installation strategies
    sys.path.insert(0, str(Path(repo_root) / "src"))
    
    try:
        from setup.install_strategies import (
            KaggleInstallationStrategy,
            InstallationContext,
        )
        
        # Create installation context with Kaggle strategy
        strategy = KaggleInstallationStrategy(verbose=True)
        context = InstallationContext(strategy)
        
        # Execute complete installation process
        context.install_all(requirements_path=req_path)
        
    except ImportError as e:
        print(f"[error] Failed to import installation strategies: {e}")
        print("[info] Falling back to basic installation...")
        _fallback_install(req_path)
    except Exception as e:
        print(f"[warn] Installation error: {e}")


def _fallback_install(req_path: Path) -> None:
    """Fallback installation method if Strategy Pattern import fails.
    
    Args:
        req_path: Path to requirements.txt file.
    """
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_path), "--quiet"],
            check=True
        )
        print("[setup] Requirements installed successfully (fallback method)")
    except subprocess.CalledProcessError as e:
        print(f"[warn] Fallback installation failed: {e}")


def validate_dataset() -> None:
    """Validate that dataset exists."""
    if not os.path.exists(DATA_PARQUET):
        print(f"[error] Dataset not found: {DATA_PARQUET}")
        print("[hint] Add dataset to Kaggle: /kaggle/input/janus-m15-dataset")
        print("\nAvailable input files:")
        for root, _, files in os.walk("/kaggle/input"):
            for file in files:
                print(f"  {os.path.join(root, file)}")
        raise SystemExit(1)


def print_device_info() -> None:
    """Print CUDA device information."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        print(f"[info] CUDA available: {cuda_available} | Devices: {device_count}")
        if cuda_available:
            for i in range(device_count):
                print(f"[info] Device {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("[warn] PyTorch not available yet")


def main() -> None:
    """Main execution function."""
    print("="*70)
    print("MAMBA BITCOIN PRICE REGRESSOR - KAGGLE TRAINING")
    print("="*70)

    repo_root = ensure_repository()
    print(f"[info] Repository root: {repo_root}")

    install_requirements(repo_root)
    validate_dataset()
    print_device_info()

    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    print("\n" + "="*70)
    print("[run] Starting training...")
    print("="*70 + "\n")

    # Run training in a fresh process to ensure newly installed packages are imported cleanly.
    # This prevents stale modules (e.g., old scikit-learn) from lingering in sys.modules.
    cmd = [
        sys.executable,
        "-u",
        "train.py",
    ]
    # Ensure the subprocess sees the same activated environment (e.g., mamba/conda LD paths)
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        lib_path = os.path.join(conda_prefix, "lib")
        old_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_path}:{old_ld}" if old_ld else lib_path
    env.setdefault("FORCE_CUDA", "1")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env.setdefault("ABSL_LOG", "0")
    env.setdefault("GLOG_minloglevel", "3")
    try:
        result = subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[error] Training subprocess failed with return code {e.returncode}")
        raise

    print("\n" + "="*70)
    print("[done] Training completed successfully!")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}/results/")
    print(f"Checkpoints saved to: {OUTPUT_DIR}/checkpoints/")
    print("="*70)


if __name__ == "__main__":
    main()
