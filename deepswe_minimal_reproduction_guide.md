# DeepSWE Minimal Viable Reproduction Guide with Qwen3-8B

## Overview

This guide details how to reproduce DeepSWE training with Qwen3-8B on a minimal scale, addressing resource constraints and technical obstacles.

## Key Findings from Codebase Analysis

### 1. Architecture Overview
- **Framework**: rLLM (reinforcement learning for language models)
- **Environment**: R2E-Gym for high-quality SWE-Bench environments
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Infrastructure**: Kubernetes cluster for Docker container orchestration
- **Model**: Originally Qwen3-32B, we'll adapt to Qwen3-8B

### 2. Major Obstacles & Resource Constraints

#### Infrastructure Requirements
- **Full Training**: K8 cluster with nodes having 200 CPUs and 6TB+ disk space
- **Containers**: Supports 512 Docker containers in parallel
- **WARNING**: README explicitly states `kind` (local K8) is insufficient for full training

#### GPU Requirements
- **Original**: 64+ GPUs with tensor parallelism size of 8
- **Memory**: High GPU memory utilization (0.6 in config)
- **VLLM**: Uses async rollout for efficiency

#### Data Processing
- Multiple datasets: R2E-Gym-Subset, SWE-Bench-Verified, etc.
- Data must be converted to parquet format
- Requires high disk I/O for Docker image caching

## Minimal Viable Reproduction Strategy

### 1. Infrastructure Adaptations

#### For Limited Resources
```bash
# Use Docker instead of full K8 for minimal setup
# Reduce parallel environments from 512 to 8-16
# Use local storage with SSD for faster I/O
```

#### Modified Environment Configuration
```python
# In rllm/environments/swe/swe.py
class SWEEnv(BaseEnv):
    def __init__(self, ...):
        # Change backend from 'kubernetes' to 'docker' for local testing
        self.backend = 'docker'  # instead of 'kubernetes'
```

### 2. Model Adaptations for Qwen3-8B

#### Configuration Changes
```yaml
# Modified train_deepswe_8b.sh
actor_rollout_ref.model.path=Qwen/Qwen3-8B  # Changed from Qwen3-32B
actor_rollout_ref.rollout.tensor_model_parallel_size=2  # Reduced from 8
actor_rollout_ref.rollout.gpu_memory_utilization=0.8  # Increased for smaller model
actor_rollout_ref.actor.ulysses_sequence_parallel_size=2  # Reduced from 8
trainer.n_gpus_per_node=4  # Reduced from 8
trainer.nnodes=1  # Reduced from 2
```

### 3. Data Subset Strategy

#### Prepare Minimal Dataset
```python
# prepare_minimal_swe_data.py
from datasets import load_dataset
from rllm.data.dataset import DatasetRegistry

def prepare_minimal_swe_data():
    # Use only subset for testing
    dataset = load_dataset("R2E-Gym/R2E-Gym-Subset", split="train[:100]")
    val_dataset = load_dataset("R2E-Gym/SWE-Bench-Verified", split="test[:20]")
    
    # Register datasets
    DatasetRegistry.register_dataset("minimal_swe_train", dataset, "train")
    DatasetRegistry.register_dataset("minimal_swe_val", val_dataset, "test")
```

### 4. Training Script Modifications

```bash
#!/bin/bash
# train_deepswe_8b_minimal.sh

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=./data/minimal_swe_train.parquet \
    data.val_files=./data/minimal_swe_val.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=10 \
    env.name=swe \
    env.env_args.backend=docker \
    agent.name=sweagent \
    agent.max_steps=30 \
    agent.trajectory_timeout=1800
```

### 5. Critical Bottlenecks & Solutions

#### Memory Constraints
- **Problem**: 32B model requires 8x tensor parallelism
- **Solution**: 8B model can run with 2x tensor parallelism on 4 GPUs

#### Docker Container Management
- **Problem**: 512 parallel containers require massive resources
- **Solution**: Reduce to 8-16 containers, increase per-container processing

#### Data Loading
- **Problem**: Full datasets are huge
- **Solution**: Use subset sampling and streaming datasets

#### Environment Setup Time
- **Problem**: Each Docker environment takes time to initialize
- **Solution**: Pre-cache base images, use persistent volumes

### 6. Step-by-Step Minimal Reproduction

```bash
# 1. Setup environment
conda create -n deepswe python=3.10
conda activate deepswe
pip install -e ./verl
pip install -e ./verl[vllm]
pip install -e .
pip install -e ../R2E-Gym

# 2. Prepare minimal data
python prepare_minimal_swe_data.py

# 3. Start minimal K8/Docker setup
# For testing, use docker-compose instead of full K8
docker-compose up -d

# 4. Run training
bash train_deepswe_8b_minimal.sh
```

### 7. Expected Resource Requirements

#### Minimal Setup
- **GPUs**: 4x A100 40GB or 4x A6000 48GB
- **RAM**: 128GB system memory
- **Storage**: 2TB SSD (for Docker images)
- **Time**: ~1 week for meaningful results

#### Comparison to Full Training
- **Original**: 64 GPUs, 2 weeks, 512 containers
- **Minimal**: 4 GPUs, 1 week, 16 containers
- **Trade-off**: Lower performance but viable for research

### 8. Monitoring & Validation

```python
# Monitor training progress
from rllm.utils import compute_pass_at_k

# Validate on small test set
results = engine.execute_tasks(minimal_test_tasks)
pass_at_1 = compute_pass_at_k(results, k=1)
print(f"Pass@1 on minimal test: {pass_at_1}")
```

### 9. Alternative Approaches

#### CPU-Only Development
- Use smaller models (1B-3B) for algorithm development
- Test on synthetic SWE tasks
- Scale up once algorithm is validated

#### Gradient Accumulation
- Simulate larger batch sizes with gradient accumulation
- Trade compute time for memory

#### LoRA Fine-tuning
- Enable LoRA in config: `lora_rank: 32`
- Reduces memory requirements significantly

## Conclusion

While full DeepSWE reproduction requires massive resources, this minimal approach allows:
1. Algorithm validation with smaller models
2. Testing on subset of data
3. Gradual scaling as resources permit
4. Research insights without full-scale infrastructure

Key insight: The core innovation is the RL training loop on SWE tasks, which can be studied at smaller scales before scaling up.