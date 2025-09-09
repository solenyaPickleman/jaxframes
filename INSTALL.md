# JaxFrames Installation Guide

## Prerequisites

### System Requirements
- Python 3.10 or later (minimum 3.10, tested with 3.10-3.12)
- Operating System: Linux, macOS, or Windows (with WSL2)
- Optional: CUDA-capable GPU or Google Cloud TPU for accelerated computing

### Hardware Recommendations
- **CPU-only**: Any modern processor (for development/testing)
- **GPU**: NVIDIA GPU with CUDA 11.8+ for acceleration
- **TPU**: Google Cloud TPU v2/v3/v4 for maximum performance

## Installation Methods

### Method 1: Install from Source (Recommended for Development)

#### Using UV (Recommended)

[UV](https://docs.astral.sh/uv/) is a fast Python package manager that we use for development:

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/jaxframes.git
cd jaxframes

# Install in development mode with all dependencies
uv venv
uv pip install -e .

# Or install with development dependencies
uv pip install -e ".[dev]"
```

#### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/jaxframes.git
cd jaxframes

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Method 2: Install as a Package

#### Direct from GitHub

```bash
# Latest development version
pip install git+https://github.com/yourusername/jaxframes.git

# Specific version/tag
pip install git+https://github.com/yourusername/jaxframes.git@v0.2.0

# Specific branch
pip install git+https://github.com/yourusername/jaxframes.git@main
```

#### From PyPI (Future)

Once published to PyPI:

```bash
# Basic installation
pip install jaxframes

# With optional dependencies
pip install jaxframes[tpu]  # TPU support
pip install jaxframes[dev]  # Development tools
pip install jaxframes[all]  # Everything
```

## Platform-Specific Installation

### CPU-Only Installation

For development or testing without GPU/TPU:

```bash
# Install JAX CPU-only version
pip install --upgrade "jax[cpu]"

# Then install JaxFrames
pip install -e .
```

### GPU Installation (NVIDIA CUDA)

For NVIDIA GPU acceleration:

```bash
# Install JAX with CUDA support (CUDA 11.8)
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CUDA 12.x
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Then install JaxFrames
pip install -e .
```

### TPU Installation (Google Cloud)

For Google Cloud TPU:

```bash
# On a TPU VM
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Then install JaxFrames
pip install -e .
```

### Apple Silicon (M1/M2/M3)

For Apple Silicon Macs with Metal acceleration:

```bash
# Install JAX with Metal support
pip install --upgrade jax-metal

# Then install JaxFrames
pip install -e .
```

## Dependencies

### Core Dependencies
- `jax >= 0.4.25` - Core JAX library
- `jaxlib >= 0.4.25` - JAX XLA backend
- `numpy >= 1.24.0` - Numerical arrays
- `pandas >= 2.0.0` - DataFrame compatibility reference

### Optional Dependencies
- `pyarrow >= 14.0.0` - Parquet file support
- `fsspec >= 2023.1.0` - Cloud storage access
- `tqdm >= 4.65.0` - Progress bars

### Development Dependencies
- `pytest >= 8.0.0` - Testing framework
- `pytest-benchmark >= 4.0.0` - Performance testing
- `hypothesis >= 6.0.0` - Property-based testing
- `ruff >= 0.1.0` - Linting
- `mypy >= 1.0.0` - Type checking

## Verification

After installation, verify everything is working:

```python
# Test basic import
import jaxframes as jf
print(f"JaxFrames version: {jf.__version__}")

# Test JAX backend
import jax
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")

# Test basic functionality
import numpy as np

# Create a simple DataFrame
df = jf.JaxFrame({
    'a': np.arange(10),
    'b': np.random.randn(10)
})

# Test operations
result = (df * 2 + 1).sum()
print(f"Basic operations: ✓")

# Test distributed features (if multiple devices available)
if len(jax.devices()) > 1:
    mesh = jf.create_device_mesh()
    dist_df = jf.DistributedJaxFrame(
        {'x': np.arange(1000)},
        sharding=jf.row_sharded(mesh)
    )
    print(f"Distributed features: ✓")
else:
    print(f"Distributed features: Single device only")

print("\nInstallation successful! 🎉")
```

## Troubleshooting

### Common Issues

#### 1. JAX Installation Issues

If JAX fails to install or detect devices:

```bash
# Uninstall all JAX packages
pip uninstall jax jaxlib -y

# Reinstall appropriate version
# For CPU:
pip install --upgrade jax

# For GPU (choose CUDA version):
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

#### 2. CUDA Version Mismatch

Check your CUDA version:
```bash
nvidia-smi  # Check CUDA version
nvcc --version  # Check CUDA compiler version
```

Install matching JAX version:
- CUDA 11.8: `jax[cuda11_pip]`
- CUDA 12.x: `jax[cuda12_pip]`

#### 3. Import Errors

If you get import errors:

```bash
# Ensure you're in the correct environment
which python
pip list | grep jax

# Reinstall in development mode
pip install -e . --force-reinstall
```

#### 4. Memory Issues

For large datasets:

```python
# Set memory allocation options
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'

import jaxframes as jf
```

#### 5. Multi-Device Issues

If distributed features aren't working:

```python
import jax

# Check available devices
print(f"Devices: {jax.devices()}")
print(f"Device count: {len(jax.devices())}")

# For GPU: ensure all GPUs are visible
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # Use available GPU IDs
```

## Environment Variables

Useful environment variables for JAX/JaxFrames:

```bash
# Memory management
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # Don't preallocate GPU memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75  # Use 75% of GPU memory

# Device visibility
export CUDA_VISIBLE_DEVICES=0,1  # Use specific GPUs
export JAX_PLATFORMS=cpu  # Force CPU-only mode

# Debugging
export JAX_DEBUG_NANS=True  # Check for NaN values
export JAX_DISABLE_JIT=True  # Disable JIT for debugging
export JAX_LOG_COMPILES=True  # Log JIT compilations

# Performance
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda  # CUDA path
export JAX_ENABLE_X64=True  # Enable 64-bit precision
```

## Docker Installation

### Using the Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install JaxFrames
RUN uv venv && \
    uv pip install -e .

# Set entrypoint
CMD ["python"]
```

Build and run:

```bash
# Build image
docker build -t jaxframes .

# Run container
docker run -it jaxframes python -c "import jaxframes; print(jaxframes.__version__)"

# With GPU support
docker run --gpus all -it jaxframes
```

### Using Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  jaxframes:
    build: .
    volumes:
      - .:/app
      - ~/.cache:/root/.cache  # Cache JAX compilations
    environment:
      - XLA_PYTHON_CLIENT_PREALLOCATE=false
      - JAX_ENABLE_X64=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Run with:
```bash
docker-compose up
```

## Development Setup

For contributing to JaxFrames:

```bash
# Clone with submodules (if any)
git clone --recursive https://github.com/yourusername/jaxframes.git
cd jaxframes

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create development environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with all dev dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks (if configured)
pre-commit install

# Run tests
uv run pytest

# Run benchmarks
uv run pytest tests/benchmarks/

# Run linting
uv run ruff check .

# Run type checking
uv run mypy src/jaxframes
```

## Upgrading

To upgrade JaxFrames:

```bash
# If installed from GitHub
pip install --upgrade git+https://github.com/yourusername/jaxframes.git

# If installed in development mode
cd jaxframes
git pull
pip install -e . --upgrade

# With UV
uv pip install -e . --upgrade
```

## Uninstallation

To remove JaxFrames:

```bash
# Uninstall package
pip uninstall jaxframes

# Remove virtual environment (if created)
rm -rf venv/  # or .venv/

# With UV
uv pip uninstall jaxframes
```

## Getting Help

If you encounter issues:

1. Check the [documentation](docs/)
2. Search [existing issues](https://github.com/yourusername/jaxframes/issues)
3. Create a [new issue](https://github.com/yourusername/jaxframes/issues/new) with:
   - Python version (`python --version`)
   - JAX version (`python -c "import jax; print(jax.__version__)"`)
   - JaxFrames version (`python -c "import jaxframes; print(jaxframes.__version__)"`)
   - Device info (`python -c "import jax; print(jax.devices())"`)
   - Full error traceback
   - Minimal reproducible example

## License

JaxFrames is distributed under the [LICENSE](LICENSE) terms.