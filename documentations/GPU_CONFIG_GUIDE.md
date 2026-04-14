# GPU Device Configuration Guide

## Overview

You can now specify which GPU devices to use directly in your config files. The training pipeline will automatically set `CUDA_VISIBLE_DEVICES` based on your configuration.

---

## Configuration Options

### gpuconfig Section in YAML

```yaml
gpu_config:
  # GPU device IDs to use (comma-separated string)
  device_ids: "0"           # Examples: "0", "0,1", "0,1,2,3"
  
  # Multi-GPU training flag
  use_distributed: false    # Set to true for DDP/multi-GPU training
  
  # Device mapping for model loading
  device_map: "auto"        # "auto" or device specification
  
  # Mixed precision optimization
  use_tf32: true            # Enable TF32 for faster computation (less precise)
```

---

## Common GPU Configurations

### Single GPU (GPU 0)
```yaml
gpu_config:
  device_ids: "0"
  use_distributed: false
  device_map: "auto"
  use_tf32: true
```

### Single GPU (GPU 1)
```yaml
gpu_config:
  device_ids: "1"
  use_distributed: false
  device_map: "auto"
  use_tf32: true
```

### Multiple GPUs (0 and 1)
```yaml
gpu_config:
  device_ids: "0,1"
  use_distributed: true     # Enable for multi-GPU training
  device_map: "auto"
  use_tf32: true
```

### All Available GPUs
```yaml
gpu_config:
  device_ids: "0,1,2,3"
  use_distributed: true
  device_map: "auto"
  use_tf32: true
```

### GPU with Specific Device Map
```yaml
gpu_config:
  device_ids: "0"
  use_distributed: false
  device_map: "cuda:0"      # Explicit device specification
  use_tf32: true
```

---

## Usage Examples

### Run with GPU 0 (Default)
```bash
bash scripts/train_multitask_dataset.sh ./config.yaml
# Uses: CUDA_VISIBLE_DEVICES=0
```

### Run with GPU 1
Edit `config.yaml`:
```yaml
gpu_config:
  device_ids: "1"
```

Then run:
```bash
bash scripts/train_multitask_dataset.sh
```

### Run with Multiple GPUs
Edit `config.yaml` or use `config_gpu_multi.yaml`:
```yaml
gpu_config:
  device_ids: "0,1"
  use_distributed: true
  device_map: "auto"
```

Then run:
```bash
bash scripts/train_multitask_dataset.sh ./config_gpu_multi.yaml
```

---

## How It Works

### Workflow

```
User config specifies device_ids: "0,1,2"
              ↓
Bash script extracts device_ids using Python YAML parser
              ↓
Sets CUDA_VISIBLE_DEVICES=0,1,2 in environment
              ↓
Python training script loads, re-reads config
              ↓
Python also sets CUDA_VISIBLE_DEVICES (redundant but safe)
              ↓
Model loads with device_map from config
              ↓
Training uses specified GPUs
```

### Two-Level Setting

1. **Bash Script Level**: Extracts device_ids from YAML and sets environment variable
2. **Python Level**: Re-reads GPU config and applies settings

This two-level approach ensures GPU devices are properly allocated before any GPU access.

---

## Checking GPU Availability

Before configuring, check your available GPUs:

```bash
# List all GPUs
nvidia-smi

# Check GPU indices and memory
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

Output example:
```
index, name, memory.total
0, NVIDIA A6000, 47520 MiB
1, NVIDIA A6000, 47520 MiB
2, NVIDIA A6000, 47520 MiB
```

---

## TF32 Flag Explained

### `use_tf32: true` (Default - Recommended for LLMs)
- **Speed**: ~2-3x faster matrix operations
- **Precision**: Uses TensorFloat-32 (less precise than FP32)
- **Use Case**: Training (acceptable trade-off)
- **GPU Support**: NVIDIA GPUs with compute capability ≥ 8.0 (A100, A6000, H100, etc.)

### `use_tf32: false` (Maximum Precision)
- **Speed**: Slower (full FP32)
- **Precision**: Full single-precision
- **Use Case**: Evaluation, inference (when precision is critical)
- **Impact**: ~30-50% slower training

For Llama-3.1-8B training on math tasks, **use_tf32: true** is recommended.

---

## Device Map Explained

### `device_map: "auto"` (Default)
- Automatically distributes model layers across available GPUs
- Balances memory usage
- **Best for**: Multi-GPU training, complex model architectures

### `device_map: "cuda:0"` (Single GPU)
- Explicitly use GPU 0
- **Best for**: Single GPU training when you want explicit control

### Other Options
- `"cpu"` - Load model on CPU (very slow)
- `"balanced"` - Balance layers across devices
- `"balanced_low_0"` - Minimize GPU 0 usage
- Custom specifications (advanced)

---

## Distributed Training

### Single GPU (Current Default)
```yaml
gpu_config:
  device_ids: "0"
  use_distributed: false
```

### Multi-GPU (Future Enhancement)
```yaml
gpu_config:
  device_ids: "0,1,2,3"
  use_distributed: true
```

**Note**: Current implementation supports single GPU. Multi-GPU distributed training can be enabled in future updates with DDP (Distributed Data Parallel).

---

## Pre-Made GPU Configs

### config_gpu_single.yaml
Single GPU (GPU 0) - standard configuration
```bash
bash scripts/train_multitask_dataset.sh ./config_gpu_single.yaml
```

### config_gpu_alternate.yaml
Use alternate GPU (GPU 1)
```bash
bash scripts/train_multitask_dataset.sh ./config_gpu_alternate.yaml
```

---

## Troubleshooting GPU Issues

### Issue: "CUDA out of memory"
**Solution**: Try alternate GPU or reduce batch size
```yaml
gpu_config:
  device_ids: "1"           # Switch to GPU 1
```

### Issue: "No GPU available"
**Solution**: Check nvidia-smi output and verify device_ids are valid
```bash
nvidia-smi  # Should list your GPUs
```

### Issue: GPU not detected
**Solution**: Check CUDA_VISIBLE_DEVICES is set correctly
```bash
echo $CUDA_VISIBLE_DEVICES  # Should show your device IDs
```

### Issue: Training using wrong GPU
**Solution**: Verify config.yaml has correct device_ids, then restart training

---

## Configuration Validation

Verify GPU settings will work:

```bash
# 1. Check available GPUs
nvidia-smi

# 2. Verify config device_ids match available GPUs
grep "device_ids" config.yaml

# 3. Run training (GPU setup logged at start)
bash scripts/train_multitask_dataset.sh ./config.yaml
```

Expected output:
```
Config file: ./config.yaml
GPU Devices: 0
✓ Set CUDA_VISIBLE_DEVICES=0
...
✓ Set CUDA_VISIBLE_DEVICES=0
✓ Enabled TF32 for faster computation
```

---

## Performance Tips

### GPU Selection
- Use GPU with most available memory
- Check `nvidia-smi` for memory usage
- Switch GPUs if one is busy

### TF32 Configuration
- Default (true) for Llama-3.1-8B training ✓
- Only change if precision is critical
- Provides ~2-3x speedup with minimal accuracy loss

### Batch Size Adjustment
If "CUDA out of memory" occurs:
1. Try alternate GPU (less busy)
2. Reduce `per_device_train_batch_size`
3. Reduce `gradient_accumulation_steps`
4. Enable gradient checkpointing (already enabled)

---

## Complete GPU Configuration Example

```yaml
################################## GPU CONFIGURATION ##################################
gpu_config:
  # Use GPU 0 and GPU 1 for training
  device_ids: "0,1"
  
  # Enable distributed training with multiple GPUs
  use_distributed: true
  
  # Automatically balance model across GPUs
  device_map: "auto"
  
  # Enable TF32 for 2-3x speedup (safe for Llama training)
  use_tf32: true

training_config:
  # Adjust batch size based on GPU memory
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 4
  
  # Use gradient accumulation on top of GPU setup
  gradient_accumulation_steps: 8
```

---

## Next Steps

1. **Check GPU availability**:
   ```bash
   nvidia-smi
   ```

2. **Update config.yaml with desired GPU devices**:
   ```yaml
   gpu_config:
     device_ids: "0"  # or your preferred GPUs
   ```

3. **Run training**:
   ```bash
   bash scripts/train_multitask_dataset.sh
   ```

4. **Monitor GPU usage** (in separate terminal):
   ```bash
   watch nvidia-smi
   ```

---

**GPU configuration is now centralized in your config files!** 🚀
