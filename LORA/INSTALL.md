# Installation Guide for LoRA and QLoRA Implementation

This guide will help you set up the environment needed to run the LoRA and QLoRA implementations.

## Prerequisites

- Python 3.8 or newer
- CUDA compatible GPU (for training) - Optional but recommended
- Git

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/LoRA-QLoRA-Implementation.git
cd LoRA-QLoRA-Implementation
```

## Step 2: Create a Virtual Environment

### Using venv (recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Using conda

```bash
# Create a conda environment
conda create -n lora python=3.8
conda activate lora
```

## Step 3: Install Dependencies

```bash
# Install the required packages
pip install -r requirements.txt
```

### Installing bitsandbytes for QLoRA

bitsandbytes is required for 4-bit quantization in QLoRA. Installation can sometimes be tricky:

#### For Windows:

```bash
# Install from pre-built wheels
pip install bitsandbytes
```

#### For Linux:

```bash
# Install with CUDA support (adjust CUDA version as needed)
pip install bitsandbytes-cuda117  # For CUDA 11.7
# OR
pip install bitsandbytes-cuda118  # For CUDA 11.8
```

If you encounter issues, follow the detailed installation guide at the [bitsandbytes repository](https://github.com/TimDettmers/bitsandbytes).

## Step 4: Verify Installation

To verify that everything is installed correctly, run the following command:

```bash
python -c "import torch; import transformers; import bitsandbytes; print('Installation successful!')"
```

## Step 5: Getting Started

Now you can run the example training scripts:

### For LoRA:

```bash
python train_vit_image_classifier.py
```

### For QLoRA:

```bash
python train_qlora_text_classifier.py
```

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related errors:

1. Verify your CUDA installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

2. Check your CUDA version:
```bash
nvcc --version
```

3. Ensure your installed PyTorch version matches your CUDA version by following the instructions at [PyTorch's official website](https://pytorch.org/get-started/locally/).

### bitsandbytes Issues

For issues with bitsandbytes, try:

1. Uninstall and reinstall with specific CUDA version:
```bash
pip uninstall bitsandbytes
pip install bitsandbytes-cuda117  # Replace with your CUDA version
```

2. Check that the library can be loaded correctly:
```bash
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
```

### Other Issues

- Check out the GitHub issues page for known problems and solutions
- Make sure all dependencies are correctly installed
- Ensure your GPU has enough memory for the models you're trying to work with 