#!/usr/bin/env python
"""
Quick Start Runner Script

This script provides a simple interface to run the complete ML pipeline
locally without needing to understand all configuration options.

Usage:
    python run.py --mode train
    python run.py --mode preprocess
    python run.py --mode inference --checkpoint outputs/checkpoints/best_model.pt
    python run.py --mode test
"""

import argparse
import sys
import subprocess
from pathlib import Path


def check_data(auto_download: bool = False):
    """Check if data directory exists and has expected structure."""
    data_dir = Path("data/raw")
    
    if not data_dir.exists():
        print("=" * 60)
        print("DATA DIRECTORY NOT FOUND")
        print("=" * 60)
        print()
        
        if auto_download:
            print("Attempting automatic download from Kaggle...")
            return download_data()
        
        print("Run with --mode download to auto-download, or manually:")
        print("https://www.kaggle.com/datasets/ninadmehendale/multimodal-iris-fingerprint-biometric-data")
        print()
        print("Then extract it to: data/raw/")
        print()
        print("Expected structure:")
        print("  data/raw/iris/*.bmp")
        print("  data/raw/fingerprint/*.BMP")
        print()
        return False
    
    iris_dir = data_dir / "iris"
    fp_dir = data_dir / "fingerprint"
    
    if not iris_dir.exists() or not fp_dir.exists():
        print("Warning: Expected 'iris' and 'fingerprint' subdirectories in data/raw/")
        return False
    
    iris_count = len(list(iris_dir.glob("*.bmp")))
    fp_count = len(list(fp_dir.glob("*.BMP"))) + len(list(fp_dir.glob("*.bmp")))
    
    print(f"Found {iris_count} iris images and {fp_count} fingerprint images")
    return iris_count > 0 and fp_count > 0


def download_data(force: bool = False, output: str = "data/raw"):
    """Download data from Kaggle."""
    print("Downloading data from Kaggle...")
    
    cmd = [
        sys.executable, "-m", "src.data_download",
        "--output", output
    ]
    
    if force:
        cmd.append("--force")
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_download(args):
    """Download dataset from Kaggle."""
    cmd = [
        sys.executable, "-m", "src.data_download",
        "--output", args.data_output or "data/raw"
    ]
    
    if args.force:
        cmd.append("--force")
    
    if args.kaggle_username:
        cmd.extend(["--username", args.kaggle_username])
    
    if args.kaggle_key:
        cmd.extend(["--key", args.kaggle_key])
    
    return subprocess.run(cmd)


def run_preprocess(args):
    """Run data preprocessing."""
    print("Running preprocessing...")
    
    cmd = [
        sys.executable, "-m", "src.preprocessing",
        "--config", args.config
    ]
    
    if args.benchmark:
        cmd.append("--benchmark")
        cmd.extend(["--data-dir", "data/raw"])
    
    return subprocess.run(cmd)


def run_train(args):
    """Run model training."""
    print("Running training...")
    
    cmd = [
        sys.executable, "-m", "src.train",
        "--config", args.config
    ]
    
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    return subprocess.run(cmd)


def run_inference(args):
    """Run model inference."""
    print("Running inference...")
    
    checkpoint = args.checkpoint or "outputs/checkpoints/best_model.pt"
    
    cmd = [
        sys.executable, "-m", "src.inference",
        "--config", args.config,
        "--checkpoint", checkpoint
    ]
    
    if args.output:
        cmd.extend(["--output", args.output])
    
    return subprocess.run(cmd)


def run_tests(args):
    """Run test suite."""
    print("Running tests...")
    
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=html"])
    
    return subprocess.run(cmd)


def run_lint(args):
    """Run code quality checks."""
    print("Running lint checks...")
    
    commands = [
        [sys.executable, "-m", "black", "--check", "src/", "tests/"],
        [sys.executable, "-m", "isort", "--check-only", "src/", "tests/"],
        [sys.executable, "-m", "flake8", "src/", "tests/", "--max-line-length=100"],
    ]
    
    for cmd in commands:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            return result
    
    return result


def run_evaluate(args):
    """Run model evaluation."""
    print("Running model evaluation...")
    
    checkpoint = args.checkpoint or "outputs/checkpoints/best_model.pt"
    
    cmd = [
        sys.executable, "-m", "src.evaluate",
        "--config", args.config,
        "--checkpoint", checkpoint,
        "--split", "test"
    ]
    
    if args.output:
        cmd.extend(["--output", args.output])
    
    return subprocess.run(cmd)


def run_retrain(args):
    """Run champion/challenger retraining."""
    print("Running champion/challenger retraining...")
    
    cmd = [
        sys.executable, "-m", "src.retrain",
        "--config", args.config,
        "--threshold", str(args.threshold or 0.01)
    ]
    
    if args.force:
        cmd.append("--force-promote")
    
    if args.output:
        cmd.extend(["--output", args.output])
    
    return subprocess.run(cmd)


def run_monitor(args):
    """Run drift monitoring."""
    print("Running drift monitoring...")
    
    cmd = [
        sys.executable, "-m", "src.monitoring",
        "--config", args.config,
        "--action", "monitor"
    ]
    
    if args.output:
        cmd.extend(["--output", args.output])
    
    return subprocess.run(cmd)


def run_baseline(args):
    """Create monitoring baseline."""
    print("Creating monitoring baseline...")
    
    cmd = [
        sys.executable, "-m", "src.monitoring",
        "--config", args.config,
        "--action", "baseline",
        "--output", args.output or "outputs/monitoring/baseline"
    ]
    
    return subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Biometric MLOps Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --mode train
  python run.py --mode preprocess --benchmark
  python run.py --mode inference --checkpoint outputs/checkpoints/best_model.pt
  python run.py --mode evaluate --checkpoint outputs/checkpoints/best_model.pt
  python run.py --mode retrain
  python run.py --mode monitor
  python run.py --mode test --coverage
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["download", "train", "preprocess", "inference", "evaluate", 
                 "retrain", "monitor", "baseline", "test", "lint", "all"],
        required=True,
        help="Pipeline mode to run (all = download + preprocess + train)"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--checkpoint",
        help="Path to model checkpoint (for inference)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output path for predictions"
    )
    
    parser.add_argument(
        "--resume",
        help="Checkpoint path to resume training from"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run preprocessing benchmark"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage report"
    )
    
    parser.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip data directory check"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download data"
    )
    
    parser.add_argument(
        "--data-output",
        default="data/raw",
        help="Output directory for data download"
    )
    
    parser.add_argument(
        "--kaggle-username",
        help="Kaggle username (or set KAGGLE_USERNAME env var)"
    )
    
    parser.add_argument(
        "--kaggle-key",
        help="Kaggle API key (or set KAGGLE_KEY env var)"
    )
    
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Auto-download data if missing"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Promotion threshold for retraining (default: 1%%)"
    )
    
    args = parser.parse_args()
    
    # Handle 'all' mode - run complete pipeline
    if args.mode == "all":
        print("="*60)
        print("RUNNING COMPLETE PIPELINE: download -> preprocess -> train")
        print("="*60)
        
        # Step 1: Download
        if not check_data(auto_download=True):
            result = run_download(args)
            if result.returncode != 0:
                print("Download failed!")
                sys.exit(1)
        
        # Step 2: Preprocess
        result = run_preprocess(args)
        if result.returncode != 0:
            print("Preprocessing failed!")
            sys.exit(result.returncode)
        
        # Step 3: Train
        result = run_train(args)
        sys.exit(result.returncode)
    
    # Check data if needed
    if args.mode in ["train", "preprocess", "inference"] and not args.skip_data_check:
        if not check_data(auto_download=args.auto_download):
            print("\nUse --auto-download to download automatically,")
            print("or --skip-data-check to skip this check.")
            sys.exit(1)
    
    # Run selected mode
    if args.mode == "download":
        result = run_download(args)
    elif args.mode == "train":
        result = run_train(args)
    elif args.mode == "preprocess":
        result = run_preprocess(args)
    elif args.mode == "inference":
        result = run_inference(args)
    elif args.mode == "evaluate":
        result = run_evaluate(args)
    elif args.mode == "retrain":
        result = run_retrain(args)
    elif args.mode == "monitor":
        result = run_monitor(args)
    elif args.mode == "baseline":
        result = run_baseline(args)
    elif args.mode == "test":
        result = run_tests(args)
    elif args.mode == "lint":
        result = run_lint(args)
    else:
        parser.print_help()
        sys.exit(1)
    
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
