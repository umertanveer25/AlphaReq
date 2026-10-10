from setuptools import setup, find_packages

setup(
    name="alphareq",
    version="1.0.0",
    author="Umer Tanveer",
    author_email="umer.research@openscience.org",
    description="Automated Conflict Resolution and Autonomous Software Architecture Synthesis from Requirements",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/umertanveer25/AlphaReq",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Compilers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pytest>=7.3.0",
        "python-docx>=0.8.11"
    ],
    entry_points={
        "console_scripts": [
            "alphareq=run_pipeline:main",
        ],
    },
)
