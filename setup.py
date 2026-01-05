"""Setup script for TTM-based Human Activity Recognition."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("pytest")
        ]

# Development requirements
dev_requirements = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.4.0",
]

setup(
    name="ttm-har",
    version="0.1.0",
    author="TTM HAR Team",
    description="Human Activity Recognition using Tiny Time Mixers foundation model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ttm-har",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "docs"]),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "ttm-har-train=scripts.train:main",
            "ttm-har-evaluate=scripts.evaluate:main",
            "ttm-har-preprocess=scripts.preprocess_dataset:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "configs/**/*.yaml"],
    },
    zip_safe=False,
)
