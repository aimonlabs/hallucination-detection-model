from setuptools import setup, find_packages
import subprocess
import sys
import os

def check_gpu_available():
    try:
        # Check for NVIDIA GPU on Linux/Mac
        nvidia_output = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        )
        return nvidia_output.returncode == 0
    except FileNotFoundError:
        return False

# Base dependencies required for both CPU and GPU
base_dependencies = [
    "huggingface_hub==0.29.2",
    "nltk==3.9.1",
    "numpy==2.2.4",
    "peft==0.14.0",
    "safetensors==0.5.3",
    "transformers==4.49.0",
]

# GPU-specific dependencies
gpu_dependencies = [
    "torch==2.6.0",
    "bitsandbytes==0.45.5",
]

# Determine which dependencies to use
if check_gpu_available():
    print("GPU detected. Installing with GPU support.")
    install_requires = base_dependencies + gpu_dependencies
else:
    print("No GPU detected. Installing CPU-only version.")
    # Set environment variable for pip to use the CPU torch index
    os.environ['PIP_EXTRA_INDEX_URL'] = 'https://download.pytorch.org/whl/cpu'
    install_requires = base_dependencies + ["torch==2.6.0+cpu"]

setup(
    name="hallucination-detection-model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.8",
    description="Hallucination detection model",
    author="",
    author_email="",
) 