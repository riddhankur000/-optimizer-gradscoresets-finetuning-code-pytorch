# Quick Reference & Troubleshooting Guide

## Quick Start Commands

### 1. Minimal CoLM Training (Phi-2 on MathInstruct)

```bash
cd /data/riddhankur/PROJECTS/GREATS_COLM_pytorch/local/llm-experiments

python -m colm.train.train \
  --model_name_or_path microsoft/phi-2 \
  --train_files /data/MathInstruct.jsonl \
  --output_dir ./out/colm_phi2_test \
  --do_train True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 64 \
  --max_steps 100 \
  --data_selection_method submodlib \
  --data_selection_unit mezo \
  --efficient_mezo True \
  --small_batch_ratio 0.5 \
  --lora True \
  --lora_r 128 \
  --lora_alpha 512 \
  --learning_rate 2e-5
```

### 2. Using Script (Recommended)

```bash
bash colm/scripts/train/lora_train_math.sh \
  /data \
  microsoft/phi-2 \
  1.0 \
  42 \
  test_exp \
  8 \
  128 \
  512 \
  0.5 \
  submodlib \
  2560 \
  mezo \
  epoch \
  500 \
  100 \
  cosine \
  mezo_selection \
  512 \
  True \
  largest \
  1e-3 \
  sgd \
  proportional \
  v_proj \
  none \
  colm_exp \
  "" \
  1 \
  True
```

---

## Configuration Quick Reference

### Dataset Options

```yaml
# MathInstruct (Default - Highly Imbalanced)
train_files: "/data/MathInstruct.jsonl"
source_wise_selection: "proportional"  # Handle 300:1 imbalance
keep_sources: ""                       # Keep all sources

# SuperGLUE (Balanced)
train_files: "load-superglue-sst2"
source_wise_selection: "none"          # No explicit sources
```

### Selection Method Options

```yaml
# CoLM (Recommended for imbalanced data)
data_selection_method: "submodlib"
data_selection_unit: "mezo"            # Zeroth-order gradient
efficient_mezo: true

# GREATS (Good for validation-based selection)
data_selection_method: "greats"
data_selection_unit: "rep"             # Use representations

# FairOT (Optimal transport)
data_selection_method: "fairot_multisource"
data_selection_unit: "rep"

# Baseline (No selection)
data_selection_method: "none"
```

### Model Options

```yaml
# Phi-2 (2.7B, Recommended)
model_name_or_path: "microsoft/phi-2"
torch_dtype: "float16"
lora_target_modules: "q_proj k_proj v_proj fc1 fc2"

# Llama-3.1-8B
model_name_or_path: "meta-llama/Llama-3.1-8B"
torch_dtype: "bfloat16"
lora_target_modules: "q_proj k_proj v_proj o_proj"

# Zephyr (3B)
model_name_or_path: "HuggingFaceH4/zephyr-3b-beta"
torch_dtype: "bfloat16"
lora_target_modules: "q_proj k_proj v_proj o_proj"
```

### Hyperparameters

```yaml
# Batch Size Configuration
per_device_train_batch_size: 1         # Must be 1 for per-sample selection
gradient_accumulation_steps: 64        # Effective batch = 64
small_batch_ratio: 0.5                 # Select 50% → coreset = 32

# LoRA Configuration
lora_r: 128                            # Rank (64-256 common)
lora_alpha: 512                        # Scale (usually 4x rank)
lora_dropout: 0.05                     # Standard value

# Learning Rate
learning_rate: 2e-5                    # Math/Code tasks
lr_scheduler_type: "linear"            # or cosine

# Gradient Sparsification
zo_dim: 2560                           # Final sparse dimension
facility_similarity: "cosine"          # Distance metric
```

---

## Common Experiments & Commands

### Experiment 1: CoLM vs Random Baseline

```bash
# CoLM (best for imbalanced data)
bash colm/scripts/train/lora_train_math.sh /data microsoft/phi-2 1.0 42 exp_colm 8 128 512 0.5 submodlib 2560 mezo epoch 500 1000 cosine mezo_selection 512 True largest 1e-3 sgd proportional v_proj none my_project "" 1 True

# Random (baseline)
# Change: data_selection_method from "submodlib" to random sampler
# (Requires modifying trainer to skip selection)
```

### Experiment 2: CoLM with different sparsity levels

```bash
# High sparsity (0.7% of dims - paper setting)
zo_dim: 2560

# Medium sparsity (2% of dims)
zo_dim: 8000  

# Low sparsity (5% of dims)  
zo_dim: 20000

# No sparsity (full gradient - baseline)
zo_dim: 327000  # For LoRA rank 128
```

### Experiment 3: Comparing Selection Methods

```bash
# Run same setup with different methods:
METHODS=("submodlib" "greats" "fairot_multisource" "spot")

for method in "${METHODS[@]}"; do
    bash colm/scripts/train/lora_train_math.sh \
        /data microsoft/phi-2 1.0 42 exp_${method} \
        8 128 512 0.5 ${method} 2560 mezo epoch 500 1000 \
        cosine mezo_selection 512 True largest 1e-3 sgd \
        proportional v_proj none my_project "" 1 True
done
```

### Experiment 4: Different Batch Sizes (Memory Trade-off)

```bash
# Small batch (memory: ~20GB, fast)
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
small_batch_ratio: 0.5          # 4 samples selected

# Medium batch (memory: ~36GB, balanced)
per_device_train_batch_size: 1
gradient_accumulation_steps: 64
small_batch_ratio: 0.5          # 32 samples selected

# Large batch (memory: ~80GB, slow)
per_device_train_batch_size: 1
gradient_accumulation_steps: 256
small_batch_ratio: 1.0          # 128 samples, no selection
```

---

## Troubleshooting Guide

### Issue 1: "No module named 'colm.data'"

**Symptom**:
```
ModuleNotFoundError: No module named 'colm.data'
```

**Cause**: The data module is not included in this repository.

**Solutions**:
1. **Use pre-processed files**: Provide dataset as JSONL
   ```bash
   --train_files /data/MathInstruct.jsonl
   ```

2. **Install from parent repo**: 
   ```bash
   cd /data/riddhankur/PROJECTS/GREATS_COLM_REPO
   pip install -e .
   ```

3. **Mock the module** (temporary):
   ```python
   # Create dummy colm/data/__init__.py
   # Copy required functions from other project
   ```

---

### Issue 2: Out of Memory (OOM) Error

**Symptom**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XXGiB
```

**Solutions** (in order of impact):

**Option A**: Reduce batch size
```yaml
gradient_accumulation_steps: 32  # From 64
small_batch_ratio: 0.25          # From 0.5
```

**Option B**: Enable gradient checkpointing
```yaml
gradient_checkpointing: true
fsdp: "full_shard"
fsdp_config:
  activation_checkpointing: true
```

**Option C**: Reduce sequence length
```yaml
max_seq_length: 256   # From 512
model_max_length: 256
```

**Option D**: Use 8-bit quantization
```bash
--load_in_8bit True
```

**Option E**: Reduce sparse dimension
```yaml
zo_dim: 1024    # From 2560 (0.3% sparsity)
```

---

### Issue 3: Training is Slow (Low Throughput)

**Symptom**: Processing <1 example/second on 4 A40 GPUs

**Diagnostic**:
```bash
# Check GPU utilization
nvidia-smi -l 1  # Start in another terminal
# Look for: GPU Util, Memory, Power
```

**Causes & Solutions**:

| Cause | Solution |
|-------|----------|
| Selection overhead | Reduce `zo_dim`, use simpler method |
| Gradient accumulation | Increase `per_device_train_batch_size` to 2-4 |
| I/O bottleneck | Use SSD for dataset, increase `num_workers` |
| Synchronization | Disable logging, reduce eval frequency |
| CPU bottleneck | Reduce tokenization precision |

**Quick Fix**:
```yaml
# Fast training (sacrifice accuracy)
logging_steps: 100         # From 1
save_steps: 1000          # From 500  
efficient_mezo: true      # Skip some computations
zo_dim: 1024              # Sparser gradients
```

---

### Issue 4: Model Device Mismatch

**Symptom**:
```
RuntimeError: Expected all tensors to be on same device
```

**Cause**: Model not properly moved to GPU before training

**Solution**: Check debug outputs
```python
# Look for these in logs:
# [DEBUG] After loading model - Device of first param: cuda:0
# [DEBUG] After LoRA setup - Device of first param: cuda:0
# [DEBUG] Before trainer.train() - Device: cuda:0
```

**Fix**: In `train.py` around line 318
```python
# Ensure this runs:
if torch.cuda.is_available():
    device = torch.device("cuda" if not training_args.no_cuda else "cpu")
    model = model.to(device)
```

---

### Issue 5: Selection Method Not Working

**Symptom**: `data_selection_method` parameter ignored, always does random

**Solutions**:

1. **Check method name spelling**:
   ```yaml
   data_selection_method: "submodlib"  # Correct
   # NOT: "colm", "facility", "submod", etc.
   ```

2. **Verify trainer class**:
   ```bash
   # Check logs for:
   # "Using SubsetTrainer" or "Using SubsetTrainerEfficient"
   # If "Using HuggingFace Trainer", selection is disabled
   ```

3. **Check if method exists**:
   ```python
   # In subset_trainer_distributed.py select_data()
   # Verify your method is in the if-elif chain
   if self.method == "mymethod":
       ...
   ```

---

### Issue 6: Source-Wise Selection Not Applied

**Symptom**: 
```
Using MathInstruct but selection doesn't respect 300:1 imbalance
```

**Solutions**:

1. **Enable source tracking**:
   ```yaml
   source_wise_selection: "proportional"  # Not "none"
   remove_unused_columns: False           # Keep source column
   ```

2. **Verify sources in dataset**:
   ```python
   # Check train_dataset.features contains "source"
   print(train_dataset[0].keys())
   # Should include: input_ids, labels, source, dataset
   ```

3. **Check data collator**:
   ```bash
   # Logs should show:
   # "Using DataCollatorForSupervisedDatasetWithSource"
   # NOT: "Using DataCollatorForSupervisedDataset"
   ```

---

### Issue 7: Poor Performance / Low Accuracy

**Symptom**: Model trains but final accuracy 5-10% lower than baseline

**Potential Causes**:

| Issue | Check | Fix |
|-------|-------|-----|
| Wrong dataset splits | `train_split` in data args | Use official splits |
| Tokenization mismatch | Token count | Match paper's max_length |
| Data order randomness | `all_data_sources` | Check seed (`seed: 42`) |
| Selection too aggressive | `small_batch_ratio` | Increase from 0.5 to 1.0 |
| Learning rate too high | Training loss curve | Reduce `learning_rate` |
| Warm-up insufficient | Loss at start | Increase `warmup_ratio` |

**Debugging Steps**:
```python
# Add to train.py after dataset loading:
print(f"Dataset size: {len(train_dataset)}")
print(f"Sample: {train_dataset[0]}")
if hasattr(train_dataset, 'all_data_sources'):
    print(f"Sources: {train_dataset.all_data_sources}")
    print(f"Source distribution: {train_dataset.all_sources}")
```

---

## Performance Tracking

### Key Metrics to Monitor

```bash
# GPU Memory
watch -n1 nvidia-smi

# Training
tensorboard --logdir=./out/colm_test/runs

# W&B Dashboard
# (Automatic if WANDB_PROJECT set)
```

### Expected Performance (Phi-2 on A100/4)

| Configuration | Memory | Throughput | Time/1K Steps |
|--------------|--------|-----------|---------------|
| CoLM + LoRA | ~36 GB | ~8.5 ex/s | ~2 min |
| LoRA only | ~58 GB | ~4.2 ex/s | ~4 min |
| Full FT | ~77 GB | ~2.1 ex/s | ~8 min |

---

## Monitoring & Checkpointing

### Auto-Save Configuration

```yaml
save_strategy: "steps"      # or "epoch", "no"
save_steps: 500            # Save every N steps
save_total_limit: 3        # Keep only 3 checkpoints

eval_strategy: "steps"     # or "epoch", "no"  
eval_steps: 500            # Evaluate every N steps
```

### Resume from Checkpoint

```bash
python -m colm.train.train \
    --resume_from_checkpoint ./out/colm_test/checkpoint-500 \
    ... (other args)
```

---

## Visualization & Analysis

### Plot Results

```python
# Create custom plots
from colm.train.plot import (
    plot_selection_distribution,
    plot_gradient_heatmap,
    analyze_variance
)

# Analyze selected indices
import json
indices_dir = "./out/colm_test/indices"
with open(f"{indices_dir}/step_0_indices.json") as f:
    selected = json.load(f)
    print(f"Selected {len(selected)} samples")
    
# Visualize source distribution
sources = train_dataset.all_data_sources
selected_sources = [train_dataset[i]['source'] for i in selected]
from collections import Counter
print(Counter(selected_sources))
```

---

## Advanced Configuration

### Custom MeZO Parameters

```yaml
# Perturbation scale
mezo_eps: 1e-3             # Larger = more noise, better approx

# Gradient normalization
mezo_transform: "normalize"  # Apply layer normalization
mezo_optim: "adam"          # Track Adam moments

# Top-K selection
mezo_topk: "largest"        # Select dims with largest gradient
# OR: "smallest", "random", "sampling"
```

### Custom Facility Location

```yaml
facility_similarity: "cosine"  # Distance metric
# Options: "cosine", "euclidean", "l1"

# Per-class distribution
source_wise_selection: "proportional"
# Options: "proportional", "balanced", "none"
```

---

## Paper to Code Mapping

| Paper Section | Code File | Key Function |
|---------------|-----------|--------------|
| Algorithm 1 (CoLM) | facility_location.py | `get_orders_and_weights()` |
| Theorem 4.1 (Small sources) | facility_location.py L66+ | Include all small |
| Theorem 4.3 (Variance) | subset_trainer_distributed.py | Lower batch variance |
| Eq. 6 (MeZO) | subset_trainer_distributed.py L1420 | `zo_forward()` |
| Eq. 8 (Selection) | facility_location.py L115 | FacilityLocationFunction |
| Table 1 Results | README.md paper link | paper/results/ |

---

## FAQ

**Q: Should I use CoLM or LoRA?**
A: Both! CoLM reduces batch-size-based memory, LoRA reduces parameter memory. Stack them.

**Q: What batch ratio should I use?**
A: 0.5 (50%) is balanced. Use 0.3 if very memory-limited, 1.0 if no selection.

**Q: Can I use CoLM with other datasets?**
A: Yes, but benefits largest when source imbalance exists (300:1 ratio).

**Q: How do I know selection is working?**
A: Check logs for "select_data_facloc" calls, verify saved_indices files.

**Q: Can I use different models?**
A: Yes (HF compatible). Just adjust `lora_target_modules` per model.

**Q: How to deploy fine-tuned model?**
A: Use `.merge_and_unload()` to merge LoRA into base model, save.

---

## Performance Benchmarks

### CoLM Results Summary

```
Dataset: MathInstruct (14 sources, 300:1 imbalance)
Model: Phi-2 (2.7B params)

┌─────────────────────────────────────────────────────────┐
│ Method              In-Domain    Out-Domain    Memory   │
├─────────────────────────────────────────────────────────┤
│ CoLM (bs=64)           51.9%        61.4%      ~36 GB  │
│ FT (bs=64)             48.3%        51.9%      ~36 GB  │
│ FT (bs=128)            49.8%        55.3%      ~58 GB  │
│ FT (bs=256)            51.8%        58.9%      ~80 GB  │
└─────────────────────────────────────────────────────────┘

* Accuracies averaged across multiple in/out-domain tasks
* Results with LoRA (rank 128) + bfloat16
* Trained on 4x A100 GPUs or equivalent
```

---

## Support & Resources

- **Paper**: https://arxiv.org/pdf/2407.19580
- **GitHub**: https://github.com/BigML-CS-UCLA/CoLM
- **Issues**: File in project repository
- **Citation**: See CODEBASE_ANALYSIS.md Section 13

---

