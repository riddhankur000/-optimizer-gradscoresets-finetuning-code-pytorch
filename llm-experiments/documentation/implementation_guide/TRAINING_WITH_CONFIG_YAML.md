# Sequential Training with config.yaml

## 📋 Overview

This training setup uses **config.yaml** for all configuration, supporting:
- ✅ **Both AdamW and Muon optimizers** (selectable in config)
- ✅ **Single GPU or distributed multi-GPU training** (selectable in config)
- ✅ **Automatic WandB tracking** with 20+ metrics
- ✅ **Eval loss tracking** for overfitting detection
- ✅ **Sequential multi-task training** (model loaded once, persists through tasks)

---

## 🚀 Quick Start

### 1. Basic Usage (Default Config)
```bash
# Make script executable
chmod +x train.sh

# Run with default config.yaml settings
bash train.sh
```

### 2. Override Optimizer
```bash
# Use Muon optimizer instead of AdamW
bash train.sh --optimizer muon

# Use AdamW optimizer explicitly
bash train.sh --optimizer adamw
```

### 3. Override GPU Profile  
```bash
# Use GPU 0
bash train.sh --gpu gpu_0

# Use GPU 1
bash train.sh --gpu gpu_1

# Use multi-GPU (distributed training)
bash train.sh --gpu gpu_multi

# Use CPU only
bash train.sh --gpu cpu
```

### 4. Custom Config File
```bash
# Use custom configuration
bash train.sh --config ./my_custom_config.yaml

# Combine with overrides
bash train.sh --config ./my_config.yaml --optimizer muon --gpu gpu_multi
```

---

## ⚙️ Configuration (config.yaml)

### Active Profiles Section

Select your optimizer and GPU setup:

```yaml
active_profiles:
  optimizer: "adamw"    # Options: "adamw", "muon"
  gpu: "gpu_multi"      # Options: "gpu_0", "gpu_1", "gpu_multi", "cpu"
```

### Optimizer Profiles

#### AdamW Profile
```yaml
optimizer_profiles:
  adamw:
    optimizer_type: "adamw"
    adam_beta1: 0.9       # Momentum beta
    adam_beta2: 0.999     # RMS-prop beta
    adam_epsilon: 1e-8    # Numerical stability
```

#### Muon Profile
```yaml
muon:
  optimizer_type: "muon"
  muon_lr: 0.002                           # Learning rate
  muon_weight_decay: 0.1                   # Weight decay
  muon_momentum: 0.95                      # Momentum factor
  muon_nesterov: true                      # Nesterov momentum
  muon_ns_coefficients: [3.4445, -4.775, 2.0315]  # Newton-Schulz
  muon_eps: 1e-7                           # Epsilon
  muon_ns_steps: 5                         # NS iterations
  muon_adjust_lr_fn: "match_rms_adamw"    # LR adjustment
```

### GPU Profiles

#### Single GPU Profiles
```yaml
gpu_profiles:
  gpu_0:
    device_ids: "0"
    use_distributed: false
    device_map: "auto"
    use_tf32: true
    
  gpu_1:
    device_ids: "1"
    use_distributed: false
    device_map: "auto"
    use_tf32: true
```

#### Multi-GPU Distributed
```yaml
gpu_multi:
  device_ids: "0,1"
  use_distributed: true
  device_map: "auto"
  use_tf32: true
```

#### CPU Only
```yaml
cpu:
  device_ids: null
  use_distributed: false
  device_map: "cpu"
  use_tf32: false
```

---

## 📁 Key Files

### New Files Created

| File | Purpose |
|------|---------|
| `colm/train/config_parser.py` | YAML config loader and parser |
| `colm/train/train_sequential_from_config.py` | Main training script (uses config.yaml) |
| `train.sh` | Bash wrapper for easy training with overrides |
| **This README** | Documentation |

### Existing Files (Still Work)
- `colm/train/train_sequential_riemannian.py` - Still available (legacy)
- `configs/sequential_riemannian_config.json` - Still available (legacy)
- All other CoLM files unchanged

---

## 🎯 Training Examples

### Example 1: AdamW with Single GPU

```bash
# Edit config.yaml
active_profiles:
  optimizer: "adamw"
  gpu: "gpu_0"

# Run training
bash train.sh
```

**What will happen:**
- Uses GPU 0
- AdamW optimizer with beta1=0.9, beta2=0.999
- Sequential multi-task training
- Single WandB run for all tasks
- 20+ metrics tracked automatically

### Example 2: Muon with Multi-GPU

```bash
# Edit config.yaml
active_profiles:
  optimizer: "muon"
  gpu: "gpu_multi"

# Run training
bash train.sh
```

**What will happen:**
- Distributed training on GPUs 0 and 1
- Muon optimizer with momentum=0.95, Newton-Schulz updates
- Higher throughput with multi-GPU
- All metrics aggregated across GPUs

### Example 3: Quick Override (No Config Editing)

```bash
# Train with Muon on multi-GPU without editing config.yaml
bash train.sh --optimizer muon --gpu gpu_multi
```

---

## 🔄 How It Works

### Model Persistence Pattern

```
Step 1: Load model ONCE (before task loop)
        model = load_model()
        model = add_lora(model)

Step 2: Task loop (model persists!)
        for task_id in num_tasks:
            trainer = setup_trainer(model=model)  ← SAME model object
            train(trainer)                        ← Updates weights in-place
            eval(trainer)                         ← Tracks eval_loss
            # Model persists to next iteration!
```

### Optimizer Selection

```python
# If config says: optimizer: "adamw"
  → Use HF Trainer's built-in AdamW optimizer
  → Settings from optimizer_profiles.adamw

# If config says: optimizer: "muon"
  → Try to import Muon library
  → Create Muon optimizer with settings from optimizer_profiles.muon
  → Falls back to AdamW if Muon not available
```

### GPU Setup

```python
# If config says: gpu_profile: "gpu_0"
  → Use single GPU 0
  → No distributed training
  
# If config says: gpu_profile: "gpu_multi"
  → Use GPUs 0 and 1
  → Enable PyTorch DDP (Distributed Data Parallel)
  → Scale batch size appropriately
```

---

## 📊 Metrics Tracked in WandB

### Real-time Metrics (Per Step)
- `loss` - Training loss
- `eval_loss` - Validation loss
- `learning_rate` - Current learning rate
- `task_id` - Current task number
- `train_perplexity` - exp(loss)
- `eval_perplexity` - exp(eval_loss)
- `overfit_ratio` - eval_loss / train_loss

### Performance Metrics
- `grad_norm` - L2 norm of gradients
- `grad_norm_avg` - Average gradient norm
- GPU memory usage per GPU
- GPU utilization percentage
- CPU usage percentage
- System memory availability

### Summary Metrics
- Per-task training loss
- Per-task validation loss
- Per-task perplexity
- Total tasks completed
- Optimizer used
- GPU profile used
- Run name

---

## 🐛 Troubleshooting

### Issue 1: "Optimizer not available" error

**Problem**: Error saying Muon optimizer not available

**Solution**: 
```bash
# Muon needs to be installed
pip install muon

# Or install with extras
pip install -e . [muon]
```

**Fallback**: The system automatically falls back to AdamW if Muon is not available.

---

### Issue 2: Out of Memory (OOM) when using multi-GPU

**Problem**: CUDA out of memory error

**Solutions**:
1. Reduce batch size in config.yaml:
```yaml
per_device_train_batch_size: 4  # from 8
```

2. Increase gradient accumulation:
```yaml
gradient_accumulation_steps: 8  # from 4
```

3. Enable gradient checkpointing (already enabled by default):
```yaml
gradient_checkpointing: true
```

---

### Issue 3: Multi-GPU not working

**Problem**: Distributed training not starting

**Solutions**:
1. Verify GPUs are visible:
```bash
nvidia-smi  # Should show 2+ GPUs
```

2. Check GPU profile in config:
```yaml
gpu: "gpu_multi"  # Must be set to gpu_multi
```

3. Run with PyTorch distributed launcher:
```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    colm/train/train_sequential_from_config.py \
    ./config.yaml
```

---

### Issue 4: WandB not logging

**Problem**: Metrics not appearing in WandB dashboard

**Solutions**:
1. Install wandb:
```bash
pip install wandb
```

2. Login to WandB:
```bash
wandb login
```

3. Check config.yaml:
```yaml
wandb_config:
  enabled: true
  project: "colm-sequential-training"
```

---

## 🔧 Advanced Configuration

### Changing Learning Rate

```yaml
training_config:
  learning_rate: 5e-5  # Default: 2e-4
```

**Note**: For Muon, there's a separate `muon_lr` in the optimizer profile.

### Adjusting Evaluation Frequency

```yaml
training_config:
  eval_steps: 50  # Evaluate every 50 steps (default: 16)
```

More frequent evaluation = better overfitting detection but slower training.

### Batch Size Tuning

```yaml
training_config:
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 2
  # Effective batch = 16 * 2 * num_gpus = 32 (for 1 GPU) or 64 (for 2 GPUs)
```

### Model Selection

```yaml
model_config:
  model_id: "meta-llama/Llama-3.1-8B"  # Change model here
  torch_dtype: "bfloat16"
```

Supports any HuggingFace model ID.

---

## 📈 Expected Output

### Console Output

```
╔════════════════════════════════════════════════════════════╗
║    Sequential Multi-Task Training with config.yaml        ║
║          Support: AdamW & Muon, Single/Multi GPU          ║
╚════════════════════════════════════════════════════════════╝

✓ Config file found: ./config.yaml
✓ Python found: Python 3.10.12

Parsing configuration...
✓ Active Optimizer: muon
✓ Active GPU Profile: gpu_multi
✓ Dataset path: ./dataset

✓ Found 2 GPUs for distributed training

═════════════════════════════════════════════════════════════
Training Configuration:
═════════════════════════════════════════════════════════════

  Config File:      ./config.yaml
  Optimizer:        muon
  GPU Profile:      gpu_multi
  Dataset:          ./dataset

═════════════════════════════════════════════════════════════

Starting training...

================================================================================
CONFIGURATION SUMMARY
================================================================================

Active Profiles:
  Optimizer: muon
  GPU Profile: gpu_multi

Model Configuration:
  Model: meta-llama/Llama-3.1-8B
  Data Type: bfloat16

Training Configuration:
  Batch Size: 8
  Learning Rate: 0.002
  Max Steps: 4096

Optimizer Configuration (muon):
  LR: 0.002
  Weight Decay: 0.1
  Momentum: 0.95
  Nesterov: True

GPU Configuration (gpu_multi):
  Device IDs: 0,1
  Distributed: True
  Num GPUs: 2

================================================================================

STEP 1: Loading tokenizer...
✓ Tokenizer loaded

STEP 2: Loading base model ONCE...
✓ Model loaded: LlamaForCausalLM
  Size: 8.03B parameters

STEP 3: Applying LoRA...
trainable params: 8,388,608 || all params: 8,391,885,824 || trainable%: 0.10

STEP 4: Initializing WandB...
✅ WandB initialized: llama-3.1-8b_muon_lora_r16_2e-3

STEP 5: Loading dataset...
✓ Dataset loaded: 50000 examples

================================================================================
STARTING SEQUENTIAL TRAINING: 1 tasks
Optimizer: muon
GPU Profile: gpu_multi
================================================================================

================================================================================
TASK 0: Task_0
Model state: Already trained on 0 previous task(s)
Optimizer: muon
================================================================================

Dataset split: 45000 training, 5000 validation
Training on Task_0...
... (training progress)

✓ Task 0 training completed
  Train loss: 1.8234
✓ Task 0 evaluation completed
  Eval loss: 1.9456
✓ Checkpoint saved: ./out/llama-3.1-8b-math-multitask-lora/task_0_checkpoint

================================================================================
SEQUENTIAL TRAINING COMPLETED
================================================================================

✓ Results saved to: ./out/llama-3.1-8b-math-multitask-lora

═════════════════════════════════════════════════════════════
✓ Training completed successfully!
═════════════════════════════════════════════════════════════
```

### WandB Dashboard

Single run showing:
- Training loss (continuous curve across all tasks)
- Eval loss (showing overfitting detection)
- Overfit ratio (automatically calculated)
- GPU memory usage (per GPU)
- Gradient norms
- Per-task metrics

---

## 🎓 Training Strategies

### Strategy 1: AdamW for Stable Training

```yaml
active_profiles:
  optimizer: "adamw"
  gpu: "gpu_0"

training_config:
  learning_rate: 2e-4
  num_train_epochs: 3
  per_device_train_batch_size: 8
```

**Best for**: Starting experiments, stable baseline

### Strategy 2: Muon for Efficient Training

```yaml
active_profiles:
  optimizer: "muon"
  gpu: "gpu_multi"

optimizer_profiles:
  muon:
    muon_lr: 0.002        # Higher LR for Muon
    muon_weight_decay: 0.1
    muon_momentum: 0.95
```

**Best for**: Production runs, efficiency over stability

### Strategy 3: Multi-GPU with Large Batches

```yaml
active_profiles:
  optimizer: "adamw"
  gpu: "gpu_multi"

training_config:
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 4
  # Effective batch: 16 * 4 * 2 = 128
```

**Best for**: Maximum throughput on available resources

---

## 📚 File Relationships

```
config.yaml
    ├─ Loaded by: config_parser.py
    │   ├─ Parses YAML structure
    │   ├─ Extracts active profiles
    │   ├─ Returns structured configs
    │   └─ Sets up distributed training
    │
    ├─ Used by: train_sequential_from_config.py
    │   ├─ Loads config via ConfigLoader
    │   ├─ Sets up model/tokenizer/LoRA
    │   ├─ Creates optimizer (adamw or muon)
    │   ├─ Sets up distributed training
    │   ├─ Runs task loop
    │   └─ Logs to WandB
    │
    └─ Wrapped by: train.sh
        ├─ Parses command-line args
        ├─ Overrides config values
        ├─ Checks GPU availability
        ├─ Calls train_sequential_from_config.py
        └─ Reports results
```

---

## ✅ Checklist Before Training

- [ ] `config.yaml` exists and is valid YAML
- [ ] Dataset path is correct: `dataset_config.dataset_path`
- [ ] Model can be downloaded from HuggingFace
- [ ] GPU is available (run `nvidia-smi`)
- [ ] For multi-GPU: 2+ GPUs available
- [ ] WandB installed: `pip install wandb`
- [ ] WandB logged in: `wandb login`
- [ ] Output directory is writable
- [ ] Sufficient disk space for checkpoints

---

## 🚀 Next Steps

1. **Customize config.yaml** for your setup
2. **Test with small dataset** first
3. **Monitor WandB dashboard** during training
4. **Compare optimizers** (adamw vs muon results)
5. **Experiment with GPU profiles** (single vs multi)
6. **Analyze metrics** for best approach

---

## 📞 Quick Reference

```bash
# List all available commands
bash train.sh --help

# Train with default settings
bash train.sh

# Quick experiments without editing config
bash train.sh --optimizer muon --gpu gpu_multi
bash train.sh --optimizer adamw --gpu gpu_0

# Use custom config
bash train.sh --config ./my_config.yaml

# Train and save logs
bash train.sh 2>&1 | tee training.log
```

---

## 🎯 Key Features Summary

| Feature | Status | Details |
|---------|--------|---------|
| Config-based setup | ✅ | All settings in config.yaml |
| AdamW optimizer | ✅ | Default, stable |
| Muon optimizer | ✅ | Alternative, efficient |
| Single GPU | ✅ | gpu_0 or gpu_1 |
| Multi-GPU | ✅ | gpu_multi with DDP |
| Sequential tasks | ✅ | Model persists, cumulative learning |
| Eval loss tracking | ✅ | During training for overfitting |
| WandB logging | ✅ | 20+ metrics tracked |
| Automatic fallback | ✅ | Muon→AdamW if not available |
| Easy overrides | ✅ | Command-line args override config |

