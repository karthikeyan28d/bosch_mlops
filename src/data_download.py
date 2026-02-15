"""
Automated data loading from Kaggle.

This module provides functionality to download and extract the 
multimodal biometric dataset from Kaggle automatically.

Requirements:
    - Kaggle API credentials (kaggle.json in ~/.kaggle/)
    - Or environment variables: KAGGLE_USERNAME, KAGGLE_KEY
"""

import os
import sys
import logging
import zipfile
import shutil
import argparse
from pathlib import Path
from typing import Optional, Union

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

logger = logging.getLogger("biometric_mlops")


# Dataset configuration
KAGGLE_DATASET = "ninadmehendale/multimodal-iris-fingerprint-biometric-data"
DATASET_NAME = "multimodal-iris-fingerprint-biometric-data"


def check_kaggle_credentials() -> bool:
    """
    Check if Kaggle credentials are available.
    
    Credentials can be provided via:
    1. ~/.kaggle/kaggle.json file
    2. Environment variables: KAGGLE_USERNAME, KAGGLE_KEY
    
    Returns:
        True if credentials are available
    """
    # Check environment variables
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    
    # Check kaggle.json file
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    
    # Windows alternative location
    kaggle_json_win = Path(os.environ.get("USERPROFILE", "")) / ".kaggle" / "kaggle.json"
    if kaggle_json_win.exists():
        return True
    
    return False


def setup_kaggle_credentials(username: Optional[str] = None, key: Optional[str] = None) -> bool:
    """
    Setup Kaggle credentials from provided values or environment.
    
    Args:
        username: Kaggle username (optional if using env vars)
        key: Kaggle API key (optional if using env vars)
        
    Returns:
        True if credentials are set up successfully
    """
    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        return True
    
    return check_kaggle_credentials()


def download_dataset(
    output_dir: Union[str, Path] = "data/raw",
    dataset: str = KAGGLE_DATASET,
    force: bool = False,
    unzip: bool = True
) -> bool:
    """
    Download dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset
        dataset: Kaggle dataset identifier (username/dataset-name)
        force: Force re-download even if data exists
        unzip: Whether to unzip the downloaded file
        
    Returns:
        True if download was successful
    """
    if not KAGGLE_AVAILABLE:
        logger.error(
            "Kaggle package not installed. Install with: pip install kaggle"
        )
        print("\n" + "="*60)
        print("KAGGLE PACKAGE NOT FOUND")
        print("="*60)
        print("\nInstall the Kaggle package:")
        print("  pip install kaggle")
        print()
        return False
    
    output_dir = Path(output_dir)
    
    # Check if data already exists
    iris_dir = output_dir / "iris"
    fp_dir = output_dir / "fingerprint"
    
    if not force and iris_dir.exists() and fp_dir.exists():
        iris_count = len(list(iris_dir.glob("*.bmp")))
        fp_count = len(list(fp_dir.glob("*.BMP"))) + len(list(fp_dir.glob("*.bmp")))
        
        if iris_count > 0 and fp_count > 0:
            logger.info(f"Data already exists: {iris_count} iris, {fp_count} fingerprint images")
            print(f"Data already exists: {iris_count} iris, {fp_count} fingerprint images")
            print("Use --force to re-download")
            return True
    
    # Check credentials
    if not check_kaggle_credentials():
        logger.error("Kaggle credentials not found")
        print("\n" + "="*60)
        print("KAGGLE CREDENTIALS NOT FOUND")
        print("="*60)
        print("\nTo use Kaggle API, you need to:")
        print()
        print("Option 1: Create kaggle.json file")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token'")
        print("  3. Save kaggle.json to ~/.kaggle/kaggle.json")
        print()
        print("Option 2: Set environment variables")
        print("  export KAGGLE_USERNAME=your_username")
        print("  export KAGGLE_KEY=your_api_key")
        print()
        print("Or in PowerShell:")
        print('  $env:KAGGLE_USERNAME="your_username"')
        print('  $env:KAGGLE_KEY="your_api_key"')
        print()
        return False
    
    # Initialize Kaggle API
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        logger.error(f"Failed to authenticate with Kaggle: {e}")
        print(f"Authentication failed: {e}")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    print(f"Downloading dataset: {dataset}")
    print(f"Output directory: {output_dir}")
    
    try:
        api.dataset_download_files(
            dataset,
            path=str(output_dir),
            unzip=unzip,
            quiet=False
        )
        logger.info(f"Dataset downloaded to {output_dir}")
        print(f"\nDownload complete!")
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        print(f"Download failed: {e}")
        return False
    
    # Verify and reorganize if needed
    return verify_and_organize_data(output_dir)


def verify_and_organize_data(data_dir: Union[str, Path]) -> bool:
    """
    Verify downloaded data and reorganize if needed.
    
    Some Kaggle datasets come with different directory structures.
    This function ensures the expected structure.
    
    Args:
        data_dir: Data directory to verify
        
    Returns:
        True if data is valid and organized correctly
    """
    data_dir = Path(data_dir)
    
    iris_dir = data_dir / "iris"
    fp_dir = data_dir / "fingerprint"
    
    # Check if expected structure exists
    if iris_dir.exists() and fp_dir.exists():
        iris_count = len(list(iris_dir.glob("*.bmp")))
        fp_count = len(list(fp_dir.glob("*.BMP"))) + len(list(fp_dir.glob("*.bmp")))
        
        if iris_count > 0 and fp_count > 0:
            print(f"\nData verified:")
            print(f"  Iris images: {iris_count}")
            print(f"  Fingerprint images: {fp_count}")
            return True
    
    # Look for alternative structures (nested directories)
    # The Kaggle dataset might extract to a subdirectory
    for subdir in data_dir.iterdir():
        if subdir.is_dir() and subdir.name not in ["iris", "fingerprint"]:
            nested_iris = subdir / "iris"
            nested_fp = subdir / "fingerprint"
            
            if nested_iris.exists() and nested_fp.exists():
                # Move to correct location
                print(f"Reorganizing data from {subdir}")
                
                if iris_dir.exists():
                    shutil.rmtree(iris_dir)
                if fp_dir.exists():
                    shutil.rmtree(fp_dir)
                
                shutil.move(str(nested_iris), str(iris_dir))
                shutil.move(str(nested_fp), str(fp_dir))
                
                # Clean up empty directory
                if subdir.exists() and not any(subdir.iterdir()):
                    subdir.rmdir()
                
                return verify_and_organize_data(data_dir)
    
    # Check for zip files that need extraction
    for zip_file in data_dir.glob("*.zip"):
        print(f"Extracting {zip_file}")
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(data_dir)
        zip_file.unlink()  # Remove zip after extraction
        return verify_and_organize_data(data_dir)
    
    # Data not found in expected format
    print("\nWarning: Could not find expected data structure")
    print("Expected: iris/*.bmp and fingerprint/*.BMP")
    print("\nActual contents:")
    for item in data_dir.iterdir():
        print(f"  {item}")
    
    return False


def download_from_config(config_path: Union[str, Path]) -> bool:
    """
    Download data using settings from config file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        True if download was successful
    """
    try:
        from .utils import Config
    except ImportError:
        from utils import Config
    
    config = Config.from_yaml(config_path)
    output_dir = config.data.raw_dir
    
    return download_dataset(output_dir=output_dir)


def main():
    """Main entry point for data loading."""
    parser = argparse.ArgumentParser(
        description="Download biometric dataset from Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download dataset
  python -m src.data_download
  
  # Force re-download
  python -m src.data_download --force
  
  # Custom output directory
  python -m src.data_download --output data/raw
  
  # Use credentials from arguments
  python -m src.data_download --username YOUR_USERNAME --key YOUR_KEY

Setup Kaggle credentials:
  1. Go to https://www.kaggle.com/settings
  2. Click 'Create New Token'
  3. Save kaggle.json to ~/.kaggle/kaggle.json
  
  Or set environment variables:
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_api_key
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        default="data/raw",
        help="Output directory for dataset (default: data/raw)"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to config file to get output directory"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if data exists"
    )
    
    parser.add_argument(
        "--username",
        help="Kaggle username (alternative to env var)"
    )
    
    parser.add_argument(
        "--key",
        help="Kaggle API key (alternative to env var)"
    )
    
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing data without downloading"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Setup credentials if provided
    if args.username and args.key:
        setup_kaggle_credentials(args.username, args.key)
    
    # Get output directory
    if args.config:
        try:
            from .utils import Config
        except ImportError:
            from utils import Config
        config = Config.from_yaml(args.config)
        output_dir = config.data.raw_dir
    else:
        output_dir = args.output
    
    # Verify only mode
    if args.verify_only:
        success = verify_and_organize_data(output_dir)
        sys.exit(0 if success else 1)
    
    # Download
    success = download_dataset(
        output_dir=output_dir,
        force=args.force
    )
    
    if success:
        print("\n" + "="*60)
        print("DATA READY")
        print("="*60)
        print(f"\nData directory: {output_dir}")
        print("\nYou can now run:")
        print("  python run.py --mode train")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
