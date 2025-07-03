# DeepSWE Minimal Reproduction Summary

## Executive Summary

After thoroughly investigating the DeepSWE codebase, I've identified the key challenges and created a minimal viable reproduction strategy for training Qwen3-8B on a subset of the data. The main obstacles are infrastructure requirements (K8s cluster with 200+ CPUs per node), GPU resources (64+ GPUs for full training), and container orchestration (512 parallel Docker containers).

## Key Findings

### 1. **Architecture Analysis**
- DeepSWE uses rLLM framework with PPO algorithm
- R2E-Gym provides high-quality SWE-Bench environments
- Training requires complex orchestration of Docker containers for code execution
- Original model: Qwen3-32B with 8-way tensor parallelism

### 2. **Major Obstacles Identified**

#### Infrastructure Constraints
- **Full Setup**: K8s cluster with nodes having 200 CPUs and 6TB+ disk space
- **Container Management**: 512 parallel Docker containers for environment execution
- **README Warning**: "Running K8 cluster locally with `kind` will not suffice for full training"

#### Resource Requirements
- **GPUs**: 64+ GPUs with tensor parallelism size of 8
- **Memory**: High GPU memory utilization (60% in original config)
- **Storage**: 6TB+ for Docker image caching

### 3. **Minimal Viable Approach**

I've created a complete minimal reproduction setup that includes:

1. **Adapted Configuration** (`train_deepswe_8b_minimal.sh`)
   - Qwen3-8B instead of Qwen3-32B
   - 4 GPUs with tensor parallelism of 2
   - Reduced batch sizes and sequence lengths
   - Docker backend instead of full K8s

2. **Data Preparation** (`prepare_minimal_swe_data.py`)
   - 100 training examples (vs full dataset)
   - 20 validation examples
   - Parquet format for efficient loading

3. **Infrastructure Setup** (`docker-compose-minimal.yml`)
   - Docker-in-Docker for container management
   - Ray for distributed training
   - VLLM server for efficient inference
   - MinIO for artifact storage

4. **Automated Setup** (`setup_minimal_deepswe.sh`)
   - One-click environment setup
   - Dependency installation
   - Service initialization

## Files Created

1. **`deepswe_minimal_reproduction_guide.md`** - Comprehensive technical guide
2. **`prepare_minimal_swe_data.py`** - Data preparation script
3. **`train_deepswe_8b_minimal.sh`** - Training launch script
4. **`docker-compose-minimal.yml`** - Local infrastructure setup
5. **`setup_minimal_deepswe.sh`** - Automated setup script
6. **`TROUBLESHOOTING.md`** - Common issues and solutions
7. **`QUICK_START_CHECKLIST.md`** - Quick reference guide

## Expected Resource Requirements

### Minimal Setup
- **GPUs**: 4x A100 40GB or 4x A6000 48GB
- **RAM**: 128GB system memory
- **Storage**: 2TB SSD
- **Time**: ~1 week for meaningful results

### Comparison to Full Training
| Aspect | Full DeepSWE | Minimal Version | Reduction |
|--------|--------------|-----------------|-----------|
| GPUs | 64 | 4 | 16x |
| Model Size | 32B | 8B | 4x |
| Containers | 512 | 16 | 32x |
| Training Time | 2 weeks | 1 week | 2x |
| Dataset | Full | 100/20 | ~100x |

## Key Insights

1. **Core Innovation**: The RL training loop on SWE tasks can be studied at smaller scales
2. **Scaling Strategy**: Start small and gradually increase resources
3. **Infrastructure**: Docker Compose provides sufficient orchestration for minimal setup
4. **Model Size**: 8B model is sufficient for algorithm validation

## Recommendations

1. **Start with Minimal Setup**: Use the provided scripts to validate the approach
2. **Monitor Resource Usage**: Watch for OOM errors and adjust accordingly
3. **Incremental Scaling**: Gradually increase dataset size and model complexity
4. **Profile Performance**: Use profiling tools to identify bottlenecks
5. **Join Community**: Leverage Discord and GitHub for support

## Next Steps

1. Run `./setup_minimal_deepswe.sh` to set up the environment
2. Execute `./train_deepswe_8b_minimal.sh` to start training
3. Monitor progress via Ray dashboard (http://localhost:8265)
4. Iterate on hyperparameters based on initial results
5. Scale up gradually as resources permit

## Conclusion

While full DeepSWE reproduction requires massive computational resources, this minimal approach enables:
- Algorithm validation with smaller models
- Research insights without full-scale infrastructure
- Gradual scaling as resources become available
- Practical learning about RL for code generation

The core innovation—using RL to train coding agents on real software engineering tasks—can be effectively studied and improved upon even with limited resources.