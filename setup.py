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

# Base dependencies required regardless of hardware
base_dependencies = [
    "huggingface_hub==0.29.2",
    "nltk==3.9.1",
    "numpy==2.2.4",
    "peft==0.14.0",
    "safetensors==0.5.3",
    "transformers==4.49.0",
]

# Parse command line args to see if --cpu is specified
force_cpu = "--cpu" in sys.argv
if "--cpu" in sys.argv:
    sys.argv.remove("--cpu")

if force_cpu or not check_gpu_available():
    print("Installing CPU-only version (no CUDA dependencies).")
    
    # For CPU-only setup, we don't include the torch requirement in setup.py
    # Instead we'll install it separately with the correct URL
    
    # Create a script that will run during setup
    with open("install_cpu_torch.py", "w") as f:
        f.write("""
import subprocess
import sys

# Install CPU version of PyTorch first
subprocess.check_call([
    sys.executable, "-m", "pip", "install", 
    "torch==2.6.0+cpu", 
    "--extra-index-url", "https://download.pytorch.org/whl/cpu"
])
""")
    
    # Execute the script
    subprocess.check_call([sys.executable, "install_cpu_torch.py"])
    
    # Remove bitsandbytes from the dependencies as it's GPU-specific
    install_requires = base_dependencies
else:
    print("GPU detected. Installing with GPU support.")
    install_requires = base_dependencies + [
        "torch==2.6.0",
        "bitsandbytes==0.45.5",
    ]

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