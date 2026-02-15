"""
Setup configuration for biometric_mlops package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read version
version = "1.0.0"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = []

# Read long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="biometric_mlops",
    version=version,
    author="Biometric MLOps Team",
    description="Multimodal biometric recognition MLOps pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/biometric-mlops",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "ray": [
            "ray>=2.0.0",
        ],
        "databricks": [
            "databricks-sdk>=0.20.0",
            "mlflow>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "biometric-train=src.train:main",
            "biometric-inference=src.inference:main",
            "biometric-preprocess=src.preprocessing:preprocess_from_config",
            "biometric-download=src.data_download:main",
            "biometric-evaluate=src.evaluate:main",
            "biometric-retrain=src.retrain:main",
            "biometric-monitor=src.monitoring:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
