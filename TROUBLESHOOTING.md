# DeepSWE Minimal Reproduction - Troubleshooting Guide

## Common Issues and Solutions

### 1. Out of Memory (OOM) Errors

**Symptoms:**
- CUDA out of memory errors
- Process killed due to insufficient memory

**Solutions:**
```bash
# Reduce batch size
data.train_batch_size=2  # instead of 4
actor_rollout_ref.actor.ppo_mini_batch_size=2  # instead of 4

# Enable gradient checkpointing and offloading
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

# Reduce sequence lengths
data.max_prompt_length=1024  # instead of 2048
data.max_response_length=4096  # instead of 8192

# Reduce tensor parallelism if using fewer GPUs
actor_rollout_ref.rollout.tensor_model_parallel_size=1  # for 2 GPUs
```

### 2. Docker Container Failures

**Symptoms:**
- "Cannot connect to Docker daemon"
- Container startup failures

**Solutions:**
```bash
# Ensure Docker daemon is running
sudo systemctl start docker

# Check Docker permissions
sudo usermod -aG docker $USER
newgrp docker

# For Docker-in-Docker issues
docker run --rm --privileged docker:dind docker version

# Increase Docker resources
# Edit ~/.docker/daemon.json:
{
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "data-root": "/path/to/larger/disk"
}
```

### 3. Kubernetes/Container Orchestration Issues

**Symptoms:**
- "kind is not sufficient for full training"
- Container scheduling failures

**Solutions:**
```bash
# Use Docker backend instead of Kubernetes for minimal setup
env.env_args.backend=docker  # in training script

# Reduce parallel containers
agent.async_engine=False  # disable async to reduce container count
trainer.n_gpus_per_node=2  # reduce parallelism

# Pre-pull Docker images
docker pull python:3.10
docker pull ubuntu:22.04
```

### 4. Model Loading Issues

**Symptoms:**
- "Model not found"
- Tokenizer errors

**Solutions:**
```bash
# Download model manually
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save locally
model.save_pretrained("./models/Qwen3-8B")
tokenizer.save_pretrained("./models/Qwen3-8B")

# Update config to use local path
actor_rollout_ref.model.path=./models/Qwen3-8B
```

### 5. VLLM Server Issues

**Symptoms:**
- "Connection refused to localhost:30000"
- VLLM initialization failures

**Solutions:**
```bash
# Start VLLM manually with reduced resources
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen3-8B \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --port 30000

# Check VLLM logs
docker-compose -f docker-compose-minimal.yml logs vllm-server

# Use smaller context window
export MAX_CONTEXT_LEN=8192  # instead of 16384
```

### 6. Data Loading Errors

**Symptoms:**
- "Dataset not found"
- Parquet file errors

**Solutions:**
```python
# Verify dataset files exist
import os
print(os.path.exists("./data/swe/R2E_Gym_Subset_minimal.parquet"))

# Re-run data preparation
python prepare_minimal_swe_data.py

# Use absolute paths in config
data.train_files=/absolute/path/to/data/swe/R2E_Gym_Subset_minimal.parquet
```

### 7. Training Instabilities

**Symptoms:**
- Loss explosion
- Gradient overflow

**Solutions:**
```bash
# Reduce learning rate
actor_rollout_ref.actor.optim.lr=1e-6  # instead of 5e-6

# Enable gradient clipping
actor_rollout_ref.actor.grad_clip=0.5  # instead of 1.0

# Use warmup
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1

# Monitor gradient norms
actor_rollout_ref.actor.grad_norm_threshold=10.0
```

### 8. Environment Timeout Issues

**Symptoms:**
- "Environment step timeout"
- Docker container unresponsive

**Solutions:**
```bash
# Increase timeouts
env.env_args.step_timeout=120  # instead of 60
env.env_args.reward_timeout=300  # instead of 180
agent.trajectory_timeout=3600  # instead of 1800

# Reduce environment complexity
agent.max_steps=20  # instead of 30
```

### 9. Ray Cluster Issues

**Symptoms:**
- "Ray cluster not initialized"
- Worker registration failures

**Solutions:**
```bash
# Initialize Ray manually
ray start --head --port=10001

# Check Ray status
ray status

# Increase timeout
trainer.ray_wait_register_center_timeout=600  # instead of 300

# Use local Ray
ray.init(num_cpus=16, num_gpus=4)
```

### 10. Insufficient Disk Space

**Symptoms:**
- "No space left on device"
- Docker image build failures

**Solutions:**
```bash
# Clean up Docker
docker system prune -a -f
docker volume prune -f

# Use external storage for checkpoints
trainer.default_local_dir=/path/to/larger/disk/checkpoints

# Reduce checkpoint frequency
trainer.save_freq=10  # instead of 5

# Limit Docker image cache
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    docker/docker-gc
```

## Performance Optimization Tips

### 1. Enable Mixed Precision Training
```bash
# Add to training script
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
```

### 2. Use Gradient Accumulation
```bash
# Simulate larger batch with accumulation
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
# Then accumulate over multiple steps
```

### 3. Profile Performance
```python
# Add profiling
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # training step
    pass
```

## Getting Help

If you encounter issues not covered here:

1. Check the logs:
   ```bash
   # Training logs
   tail -f logs/training.log
   
   # Docker logs
   docker-compose -f docker-compose-minimal.yml logs -f
   
   # Ray logs
   ray logs
   ```

2. Enable debug mode:
   ```bash
   export RLLM_DEBUG=1
   export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
   ```

3. Join the community:
   - Discord: https://discord.gg/BDH46HT9en
   - GitHub Issues: https://github.com/agentica-project/rllm/issues