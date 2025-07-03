# DeepSWE Minimal Viable Reproduction Analysis

## Executive Summary

This report analyzes the feasibility of reproducing DeepSWE training with Qwen3-8B on a subset of data. Based on extensive codebase investigation, while minimal reproduction is possible, significant infrastructure challenges and resource constraints must be addressed.

## Codebase Structure Analysis

### Core Components

1. **Training Infrastructure**: `examples/swe/`
   - `train_deepswe_32b.sh` - Main training script (currently configured for 32B model)
   - `train_deepswe_agent.py` - Simplified training entry point
   - `prepare_swe_data.py` - Data preparation and registration
   - `run_deepswe.py` - Inference/evaluation script

2. **Framework Architecture**: `rllm/`
   - `environments/swe/` - SWE environment implementation with Docker/K8s integration
   - `trainer/` - Training infrastructure based on modified VERL
   - `agents/` - Agent implementations (SWEAgent)
   - `data/` - Data handling and registry

3. **Environment Implementation**: `rllm/environments/swe/swe.py`
   - Wraps R2E-Gym for SWE-Bench environments
   - Supports both Docker and Kubernetes backends
   - Requires parallel Docker container execution (512 containers for full training)

## Resource Requirements Analysis

### Full DeepSWE Training (Current Implementation)
- **Model**: Qwen3-32B (~60GB VRAM for inference, ~240GB for training)
- **GPUs**: Minimum 64 GPUs (2 nodes × 8 GPUs per node × 4+ nodes)
- **Containers**: 512 Docker containers running in parallel
- **Infrastructure**: Kubernetes cluster with:
  - 200+ CPUs per node
  - 6TB+ disk space per node for Docker images
  - High-bandwidth networking (InfiniBand recommended)

### Qwen3-8B Minimal Reproduction Estimates
Based on model scaling and web search findings:

**Qwen3-8B Model Requirements**:
- **Inference**: ~16GB VRAM (FP16), ~8GB (INT4 quantized)
- **Training**: ~64GB VRAM (FP16 with Adam optimizer)
- **Recommended**: 2-4 × RTX 4090 (24GB each) or 1 × A100 80GB

**Scaled Down Training Setup**:
- **GPUs**: 4-8 GPUs minimum (vs. 64+ for full training)
- **Containers**: 32-64 Docker containers (vs. 512)
- **Memory**: 128-256GB system RAM
- **Storage**: 2-4TB NVMe SSD for Docker images and datasets

## Critical Infrastructure Challenges

### 1. Kubernetes vs. Docker Backend Limitations

**Current Kind Limitation** (from README):
> "To run Kubernetes locally, we suggest installing [`kind`](https://kind.sigs.k8s.io/) and launching it with `kind create cluster`. However, please do note that this is not sufficient to launch a full training run."

**Root Causes**:
- Kind creates single-node clusters with limited resources
- Cannot efficiently manage hundreds of parallel Docker containers
- Lacks distributed storage and networking capabilities
- Limited CPU/memory allocation compared to cloud K8s clusters

### 2. Docker Image Management Complexity

From `cache_images_k8.py` analysis:
- Training requires unique Docker images for each SWE-Bench task
- Images must be pre-cached across all K8s nodes using DaemonSets
- Process involves pulling 1000+ unique Docker images
- Each image can be several GB in size

### 3. Environment Orchestration Requirements

From `swe.py` investigation:
- Each training sample requires isolated Docker container
- Containers must support:
  - File editing tools
  - Bash execution
  - Search capabilities
  - Git operations
- Timeout management (90s per step, 300s for reward computation)
- Resource cleanup and image management

## Dataset and Training Configuration

### Available Datasets
```python
SWE_DATASETS = [
    "R2E-Gym/R2E-Gym-Subset",     # Primary training dataset
    "R2E-Gym/R2E-Gym-Lite",      # Smaller subset
    "R2E-Gym/R2E-Gym-V1",        # Full dataset
    "R2E-Gym/SWE-Bench-Lite",    # Evaluation subset
    "R2E-Gym/SWE-Bench-Verified", # Gold standard evaluation
]
```

### Training Configuration Adaptation Required

From `train_deepswe_32b.sh` - Key parameters to modify:

```bash
# Current 32B configuration
actor_rollout_ref.model.path=Qwen/Qwen3-32B
actor_rollout_ref.rollout.tensor_model_parallel_size=8
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
trainer.n_gpus_per_node=8
trainer.nnodes=2

# Proposed 8B configuration
actor_rollout_ref.model.path=Qwen/Qwen3-8B  # Target model
actor_rollout_ref.rollout.tensor_model_parallel_size=2  # Reduced parallelism
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2  # Increased batch size
trainer.n_gpus_per_node=4  # Reduced GPU count
trainer.nnodes=1  # Single node
```

## Identified Obstacles and Mitigation Strategies

### 1. **Infrastructure Complexity** (HIGH SEVERITY)
**Problem**: Full K8s cluster requirement with 512 parallel containers
**Mitigation**: 
- Use Docker Compose or simpler orchestration for 32-64 containers
- Implement sequential processing with smaller batches
- Consider cloud-based solutions (AWS EKS, GCP GKE)

### 2. **Memory and Storage Requirements** (MEDIUM-HIGH SEVERITY)
**Problem**: 6TB+ storage needed for Docker images
**Mitigation**:
- Use R2E-Gym-Lite dataset (smaller subset)
- Implement on-demand image pulling instead of pre-caching
- Use image compression and layer sharing

### 3. **Training Stability** (MEDIUM SEVERITY)
**Problem**: PPO training notorious for instability with smaller batch sizes
**Mitigation**:
- Start with supervised fine-tuning (SFT) before RL
- Use smaller learning rates (1e-7 instead of 1e-6)
- Implement gradient accumulation to simulate larger batches

### 4. **Evaluation Infrastructure** (MEDIUM SEVERITY)
**Problem**: Need parallel evaluation across SWE-Bench tasks
**Mitigation**:
- Focus on SWE-Bench-Lite (23 tasks vs. 500)
- Implement sequential evaluation
- Use simpler success metrics initially

## Recommended Minimal Reproduction Strategy

### Phase 1: Infrastructure Setup
1. **Hardware**: 4 × RTX 4090 or 2 × A100 80GB
2. **Software**: 
   - Docker with increased resource limits
   - R2E-Gym installation
   - Modified rLLM with reduced parallelism
3. **Dataset**: Start with R2E-Gym-Lite (smallest subset)

### Phase 2: Model and Training Adaptation
1. **Model**: Qwen3-8B with INT4 quantization during inference
2. **Training**: 
   - Reduce container parallelism to 32-64
   - Implement gradient accumulation
   - Use smaller context windows initially (16K instead of 32K)

### Phase 3: Validation Strategy
1. **Start Simple**: Single SWE-Bench task validation
2. **Scale Gradually**: 5-10 tasks → subset → full lite dataset
3. **Metrics**: Focus on task completion rate before optimizing for full SWE-Bench performance

## Estimated Resource Requirements for Minimal Reproduction

### Hardware Minimum
- **GPUs**: 4 × RTX 4090 (96GB total VRAM)
- **RAM**: 256GB DDR5
- **Storage**: 4TB NVMe SSD
- **Network**: High-bandwidth internet for Docker image pulls

### Software Dependencies
- **Docker**: With privileged access and increased limits
- **Python 3.10**: With uv package manager
- **CUDA 12.2+**: For GPU acceleration
- **R2E-Gym**: Latest version with environment support

### Expected Training Time
- **Full reproduction**: 2-4 weeks (vs. original months)
- **Proof of concept**: 3-7 days
- **Single task validation**: Hours to days

## Risk Assessment

### HIGH RISK
- Container orchestration complexity may require significant engineering time
- PPO training instability with reduced resources
- Docker image storage and management overhead

### MEDIUM RISK  
- Model convergence with smaller batch sizes
- Environment timeout issues in resource-constrained setup
- Evaluation pipeline complexity

### LOW RISK
- Basic model loading and inference
- Data preprocessing and loading
- Single-task environment testing

## Conclusion and Recommendations

**Feasibility**: Minimal reproduction is possible but requires significant infrastructure adaptation and engineering effort.

**Primary Recommendation**: 
1. Start with Docker-only backend (avoid Kubernetes initially)
2. Use R2E-Gym-Lite dataset
3. Implement sequential container processing
4. Focus on proof-of-concept with 5-10 SWE-Bench tasks

**Alternative Approach**: Consider cloud-based reproduction using managed Kubernetes services (EKS/GKE) which would eliminate infrastructure complexity but increase costs.

**Timeline Estimate**: 2-4 weeks for minimal reproduction, 1-2 months for robust training pipeline.

The key insight is that while the codebase supports minimal reproduction, the primary bottleneck is environment orchestration rather than model size, making this a systems engineering challenge as much as a machine learning one.