"""
Sales Forecasting with Advanced Machine Learning
Setup configuration for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements.txt
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path, 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
with open(readme_path, 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="sales-forecasting-ml",
    version="2.0.0",
    author="AUPP Student",
    description="Advanced sales forecasting using ensemble machine learning (Prophet + LSTM + XGBoost)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Skylarrrolala/ML---Final-Project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0',
            'black>=22.0',
            'flake8>=4.0',
            'mypy>=0.9',
        ],
    },
    entry_points={
        'console_scripts': [
            'sales-forecast=evaluation.quickstart:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.json', '*.csv', '*.pkl'],
    },
)
