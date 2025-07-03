# DeepSWE Minimal Reproduction - Quick Start Checklist

## Pre-requisites Checklist

- [ ] **Hardware**
  - [ ] 4+ GPUs (A100 40GB or A6000 48GB recommended)
  - [ ] 128GB+ system RAM
  - [ ] 2TB+ SSD storage
  - [ ] Linux OS (Ubuntu 20.04+ recommended)

- [ ] **Software**
  - [ ] Python 3.10
  - [ ] CUDA 11.8+
  - [ ] Docker 20.10+
  - [ ] nvidia-docker2
  - [ ] Git

## Quick Start Steps

### 1. Clone and Setup (5 minutes)
```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/agentica-project/rllm.git
cd rllm

# Run automated setup
chmod +x setup_minimal_deepswe.sh
./setup_minimal_deepswe.sh
```

### 2. Verify Installation (2 minutes)
```bash
# Activate environment
source .venv/bin/activate

# Verify Python packages
python -c "import rllm, verl, r2egym; print('All packages imported successfully')"

# Check Docker services
docker-compose -f docker-compose-minimal.yml ps
```

### 3. Prepare Data (10 minutes)
```bash
# This is already done by setup script, but can re-run if needed
python prepare_minimal_swe_data.py
```

### 4. Start Training (Launch and forget)
```bash
# Make sure services are running
docker-compose -f docker-compose-minimal.yml up -d

# Configure wandb (optional but recommended)
wandb login

# Start training
./train_deepswe_8b_minimal.sh
```

## Key Modifications from Full DeepSWE

| Component | Full DeepSWE | Minimal Version | Impact |
|-----------|--------------|-----------------|---------|
| Model | Qwen3-32B | Qwen3-8B | Faster training, lower memory |
| GPUs | 64+ | 4 | Reduced parallelism |
| Tensor Parallel | 8 | 2 | Fits on fewer GPUs |
| Batch Size | 8 | 4 | Slower convergence |
| Docker Containers | 512 | 8-16 | Less parallel evaluation |
| Infrastructure | K8s cluster | Docker Compose | Simpler setup |
| Dataset | Full | 100 train, 20 val | Faster iteration |
| Training Time | 2 weeks | 1 week | Reduced epochs |

## Monitoring Progress

### Training Metrics
```bash
# Watch training logs
tail -f logs/training.log

# Monitor GPU usage
nvidia-smi -l 1

# Check Ray dashboard
open http://localhost:8265
```

### Expected Milestones
- Hour 1: Environment setup and data loading
- Hour 2-4: First epoch completion
- Day 1: Initial validation results
- Day 3: Observable improvement in metrics
- Week 1: Convergence on minimal dataset

## Common Commands Reference

### Stop Training
```bash
# Graceful shutdown
pkill -f train_agent_ppo

# Stop all services
docker-compose -f docker-compose-minimal.yml down
```

### Resume Training
```bash
# Training automatically resumes from last checkpoint
./train_deepswe_8b_minimal.sh
```

### Clean Up
```bash
# Remove Docker containers and volumes
docker-compose -f docker-compose-minimal.yml down -v

# Clean checkpoint directory (careful!)
rm -rf checkpoints/*

# Clean logs
rm -rf logs/*
```

## Resource Usage Expectations

### GPU Memory (per GPU)
- Model weights: ~16GB
- Activations: ~10GB
- Gradients: ~8GB
- Buffer: ~6GB
- **Total: ~40GB per GPU**

### System Memory
- Ray workers: ~32GB
- Docker containers: ~32GB
- Model loading: ~32GB
- Buffer: ~32GB
- **Total: ~128GB recommended**

### Disk Usage
- Model checkpoints: ~50GB per checkpoint
- Docker images: ~500GB
- Datasets: ~10GB
- Logs: ~10GB
- **Total: ~2TB recommended**

## Success Criteria

You'll know the setup is working when:
1. ✓ No OOM errors in first hour
2. ✓ Training loss decreasing
3. ✓ Validation running without errors
4. ✓ Docker containers stable
5. ✓ Checkpoints being saved

## Next Steps After Basic Training

1. **Scale Data**: Gradually increase dataset size
2. **Tune Hyperparameters**: Adjust learning rate, batch size
3. **Add More GPUs**: Scale to 8 GPUs if available
4. **Implement Improvements**: Try different RL algorithms
5. **Full Evaluation**: Run on complete SWE-Bench-Verified

## Need Help?

- Check `TROUBLESHOOTING.md` for common issues
- Review `deepswe_minimal_reproduction_guide.md` for detailed explanations
- Join Discord community: https://discord.gg/BDH46HT9en