from setuptools import setup, find_packages
import os

# Define dependencies directly in setup.py
requirements = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "peft>=0.4.0",
    "bitsandbytes>=0.40.0",
    "nltk>=3.8.0",
    "numpy>=1.22.0",
    "huggingface_hub>=0.16.0",
    "safetensors>=0.3.0",
    "tqdm>=4.65.0"
]

# Read the long description from README.md
long_description = ""
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="hdm2",
    version="0.6.0",
    author="Bibek Paudel",
    author_email="bibek@aimon.ai",
    description="A tool for detecting and quantifying hallucinations in LLM responses through context and common knowledge verification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aimonlabs/hallucination-detection-model",
    project_urls={
        "Bug Tracker": "https://github.com/aimonlabs/hallucination-detection-model/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["accelerate"],
    },
)
