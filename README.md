# hallucination-detection-model

This repository contains the inference code for AIMon's Hallucination Detection Model-2 (HDM-2).

## Installation

The installation process will automatically detect if you have a GPU available and install the appropriate dependencies:

```bash
# Install directly from source
pip install .

# Force CPU-only installation (no CUDA dependencies)
pip install . --cpu

# Or install in development mode
pip install -e .
```

### What happens during installation:

1. The setup script checks for GPU availability using `nvidia-smi`
2. If a GPU is detected:
   - PyTorch with CUDA support is installed
   - GPU-specific libraries like bitsandbytes are installed
3. If no GPU is detected (or `--cpu` flag is used):
   - CPU-only version of PyTorch is installed (without CUDA dependencies)
   - GPU-specific dependencies are skipped

### Manual Installation

If the automatic detection doesn't work correctly, you can manually install:

#### For GPU:
```bash
pip install -r requirements.txt
```

#### For CPU-only:
```bash
pip install torch==2.6.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# Exclude bitsandbytes as it's GPU-specific
pip uninstall -y bitsandbytes
```
