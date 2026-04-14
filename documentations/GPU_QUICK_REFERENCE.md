# GPU Configuration Quick Reference

## Accessing Different GPUs

### Use GPU 0 (Default)
```yaml
gpu_config:
  device_ids: "0"
```
```bash
bash scripts/train_multitask_dataset.sh  # Uses GPU 0 by default
```

### Use GPU 1
```yaml
gpu_config:
  device_ids: "1"
```
```bash
bash scripts/train_multitask_dataset.sh ./config.yaml
```

### Use GPU 2, 3, or 4
```yaml
gpu_config:
  device_ids: "2"  # or "3" or "4"
```

### Use Multiple GPUs (GPUs 0 and 1)
```yaml
gpu_config:
  device_ids: "0,1"
  use_distributed: true
```

---

## Check Available GPUs

```bash
# List all GPUs with details
nvidia-smi

# Quick summary
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv

# Monitor GPU usage while training
watch nvidia-smi

# Check specific GPU
nvidia-smi -i 0  # Details for GPU 0
nvidia-smi -i 1  # Details for GPU 1
```

---

## Update config.yaml for Different GPU

### Before Training
1. Open `config.yaml`
2. Find the `gpu_config` section
3. Change `device_ids: "0"` to desired GPU (e.g., `"1"`, `"2"`)
4. Save file
5. Run training: `bash scripts/train_multitask_dataset.sh`

OR use pre-made configs:
```bash
# For GPU 0
bash scripts/train_multitask_dataset.sh ./config_gpu_0.yaml

# For GPU 1  
bash scripts/train_multitask_dataset.sh ./config_gpu_1.yaml

# For custom setup
bash scripts/train_multitask_dataset.sh ./config_adamw.yaml  # Edit gpu_config first
```

---

## Troubleshooting GPU Selection

### Problem: "CUDA out of memory" on GPU 0

**Solution** → Use GPU 1:
```bash
# Edit config.yaml
gpu_config:
  device_ids: "1"

# Run training
bash scripts/train_multitask_dataset.sh
```

### Problem: GPU usage shows Python not using GPU

**Verify** GPU is being used:
```bash
# Should output your device_id(s)
echo $CUDA_VISIBLE_DEVICES

# Check if CUDA is available in Python
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.current_device()}')"
```

### Problem: Wrong GPU being used

**Check** current setting:
```bash
# Verify device_ids in config
grep "device_ids" config.yaml

# Verify environment variable is set correctly
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Best practice: explicitly set before running
CUDA_VISIBLE_DEVICES=1 bash scripts/train_multitask_dataset.sh ./config.yaml
```

---

## Advanced: Override via Command Line

You can also override GPU selection at runtime without editing config:

```bash
# Force GPU 0
CUDA_VISIBLE_DEVICES=0 bash scripts/train_multitask_dataset.sh ./config.yaml

# Force GPU 1
CUDA_VISIBLE_DEVICES=1 bash scripts/train_multitask_dataset.sh ./config.yaml

# Force GPUs 0,1 (multi-GPU)
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_multitask_dataset.sh ./config.yaml
```

---

## What Gets Set Where

### 1. Bash Script (`train_multitask_dataset.sh`)
- ✓ Reads `device_ids` from config.yaml
- ✓ Sets `CUDA_VISIBLE_DEVICES` environment variable
- ✓ Logs which GPUs are being used

### 2. Python Training Script (`train_multitask.py`)
- ✓ Reads GPU config again (redundant but safe)
- ✓ Re-sets `CUDA_VISIBLE_DEVICES` 
- ✓ Enables TF32 if `use_tf32: true`
- ✓ Sets `device_map` for model loading
- ✓ Logs GPU configuration applied

### 3. Model Loading
- ✓ Uses `device_map` from config
- ✓ Loads model on specified device(s)
- ✓ Logs device info

---

## Performance Considerations

### GPU Utilization
- **Goal**: Keep GPU utilization >90% during training
- **Check**: `watch nvidia-smi` and look for GPU% column
- **If <90%**: Increase batch size or try different GPU

### GPU Memory
- **Llama-3.1-8B with LoRA + BS=2 + GA=8**: ~35-40 GB VRAM
- **If OOM**: Try reducing batch size or gradient accumulation
- **Alternative**: Use different GPU with more available memory

### Multi-GPU Training (Future)
- Currently supports single GPU per run
- To train on multiple GPUs: run multiple processes
- Each process: `CUDA_VISIBLE_DEVICES=0,1,... bash scripts/...`

---

## Complete Configuration Structure

Your config.yaml now has 9 sections:

```yaml
1. model_config              (Llama model and dtype)
2. tokenizer_config         (Tokenizer setup)
3. lora_config              (LoRA hyperparameters)
4. dataset_config           (Dataset paths and settings)
5. training_config          (Batch size, LR, scheduler, etc.)
6. optimizer_config         (AdamW or Muon settings)
7. tokenizer_args           (Tokenization options)
8. multitask_config         (Multi-task settings)
9. wandb_config             (Weights & Biases tracking)
10. system_config           (Workers, memory pins)
11. gpu_config              (← NEW: GPU device selection)
```

---

## One-Page Cheat Sheet

```bash
# Check GPUs
nvidia-smi

# Train on GPU 0
bash scripts/train_multitask_dataset.sh

# Train on GPU 1
# Edit config.yaml: device_ids: "1"
# Then run:
bash scripts/train_multitask_dataset.sh

# Train on GPU 2,3,4 (custom multi-GPU)
CUDA_VISIBLE_DEVICES=2,3,4 bash scripts/train_multitask_dataset.sh

# Monitor during training
watch -n 1 nvidia-smi

# Check if training is using GPU
ps aux | grep python | grep train_multitask
```

---

**GPU configuration is now fully integrated!** 🚀

Switch GPUs by editing `gpu_config.device_ids` in your config file or use environment variable override.
