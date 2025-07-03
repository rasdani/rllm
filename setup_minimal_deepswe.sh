#!/bin/bash
# Setup script for minimal DeepSWE reproduction with Qwen3-8B

set -e

echo "=== DeepSWE Minimal Setup Script ==="
echo "This script will set up a minimal DeepSWE training environment with Qwen3-8B"
echo

# Check system requirements
echo "Checking system requirements..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"
if [[ ! "$python_version" == "$required_version"* ]]; then
    echo "Error: Python $required_version is required, but found $python_version"
    echo "Please install Python 3.10 first"
    exit 1
fi
echo "✓ Python $python_version"

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU support may not be available."
    echo "Continue without GPU? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "✓ Found $gpu_count GPU(s)"
    if [[ $gpu_count -lt 4 ]]; then
        echo "Warning: Minimum 4 GPUs recommended, found $gpu_count"
        echo "Training will be slower and may require further config adjustments"
    fi
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker not found. Please install Docker first."
    exit 1
fi
echo "✓ Docker installed"

# Create virtual environment
echo
echo "Creating virtual environment..."
if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo
echo "Installing dependencies..."
pip install --upgrade pip

# Install verl with vllm support
echo "Installing verl..."
pip install -e ./verl
pip install -e "./verl[vllm]"

# Install rllm
echo "Installing rllm..."
pip install -e .

# Clone and install R2E-Gym if not present
echo
echo "Setting up R2E-Gym..."
if [[ ! -d "../R2E-Gym" ]]; then
    echo "Cloning R2E-Gym..."
    cd ..
    git clone https://github.com/R2E-Gym/R2E-Gym.git
    cd R2E-Gym
    pip install -e .
    cd ../$(basename "$OLDPWD")
else
    echo "✓ R2E-Gym already exists"
    cd ../R2E-Gym
    pip install -e .
    cd ../$(basename "$OLDPWD")
fi

# Install additional required packages
echo
echo "Installing additional packages..."
pip install datasets pandas tqdm pyarrow

# Create necessary directories
echo
echo "Creating necessary directories..."
mkdir -p data/swe
mkdir -p checkpoints
mkdir -p logs
mkdir -p swe-images
echo "✓ Directories created"

# Prepare minimal datasets
echo
echo "Preparing minimal datasets..."
if [[ -f "data/swe/R2E_Gym_Subset_minimal.parquet" ]] && [[ -f "data/swe/SWE_Bench_Verified_minimal.parquet" ]]; then
    echo "✓ Datasets already exist"
else
    python3 prepare_minimal_swe_data.py
fi

# Setup Docker environment
echo
echo "Setting up Docker environment..."
echo "Starting Docker Compose services..."
docker-compose -f docker-compose-minimal.yml up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Verify services
echo
echo "Verifying services..."
docker-compose -f docker-compose-minimal.yml ps

# Make training script executable
chmod +x train_deepswe_8b_minimal.sh

echo
echo "=== Setup Complete! ==="
echo
echo "Next steps:"
echo "1. Ensure you have Qwen3-8B model weights accessible"
echo "2. Configure wandb for logging: wandb login"
echo "3. Start training: ./train_deepswe_8b_minimal.sh"
echo
echo "To monitor training:"
echo "- Ray dashboard: http://localhost:8265"
echo "- MinIO console: http://localhost:9001 (user: minioadmin, pass: minioadmin)"
echo
echo "To stop services: docker-compose -f docker-compose-minimal.yml down"