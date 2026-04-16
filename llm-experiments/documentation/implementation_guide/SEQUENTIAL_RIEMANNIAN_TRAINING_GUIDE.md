# Sequential Riemannian-Style Training for CoLM: Implementation Guide

## 📋 Overview

This document explains the new sequential training implementation that mirrors the Riemannian multi-task training approach in CoLM. The key difference from standard CoLM training is:

**Standard CoLM**: Single unified training on all mixed data sources at once
**Sequential Riemannian**: Sequential multi-task training where model is loaded once and reused for all tasks

---

## 🔑 Key Architecture Changes

### 1. Model Loading Strategy

```python
# ❌ OLD CoLM Approach (unified):
train_entire_mixed_dataset()  # All 14 sources mixed in one training run

# ✅ NEW Approach (sequential):
model = load_model()  # LOAD ONCE before loop
for task in tasks:
    train(model, task_data)  # REUSE same model object
    model = model  # Weights accumulate!
```

### 2. Training Flow

```
┌─────────────────────────────────────────┐
│ 1. Load Tokenizer (once)                │
│ 2. Load Base Model (once, outside loop) │
│ 3. Apply LoRA (once, outside loop)      │
│ 4. Initialize WandB (single run)        │
│ 5. Load Dataset                         │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ↓                     ↓
   Task 0 Training      Task 1 Training
   ┌────────────────┐   ┌────────────────┐
   │ Split: 90/10   │   │ Split: 90/10   │
   │ Train/Val      │   │ Train/Val      │
   │ Train model    │   │ Train model    │
   │ Eval loss      │   │ (from Task 0!) │
   │ Save ckpt      │   │ Eval loss      │
   │ Log metrics    │   │ Save ckpt      │
   └────────────────┘   │ Log metrics    │
                        └────────────────┘
                                │
                                ↓
                        ✓ Single WandB run
                        ✓ Cumulative weights
                        ✓ Per-task metrics
```

### 3. WandB Integration

#### Single Run vs Multiple Runs

```
❌ WRONG (multiple runs):
Task 0 → WandB Run 1
Task 1 → WandB Run 2 (Fresh start)
Task 2 → WandB Run 3 (Fresh start)
Problem: Lose continuity, weights restart

✅ CORRECT (single run):
Task 0 → WandB Run (starts)
Task 1 → WandB Run (continues, weights from task 0)
Task 2 → WandB Run (continues, weights from task 0+1)
Benefit: See cumulative learning curve
```

#### Run Name Format

```python
run_name = f"{model}_{optimizer}_{method}_{learning_rate}"

Examples:
- "phi-2_adamw_lora_r128_1e-4"
- "llama-7b_muon_lora_r64_5e-5"
- "mistral_adamw_full_1e-3"
```

#### Metrics Tracked

**Per Step (via Trainer):**
- `loss` - Training loss
- `learning_rate` - Current LR
- `epoch` - Current epoch
- `global_step` - Step counter
- `task_id` - Which task being trained

**Via Monitoring Callback:**
- `train_perplexity` - exp(loss)
- `grad_norm` - L2 norm of all gradients
- `grad_norm_avg` - Average gradient norm
- `gpu_memory_used_gb` - GPU RAM used
- `gpu_memory_utilization_%` - GPU % utilization
- `cpu_percent` - CPU usage %
- `cpu_memory_percent` - System memory %

**Per Evaluation:**
- `eval_loss` - Validation loss
- `eval_perplexity` - exp(eval_loss)
- `eval_grad_norm` - Gradient norms at eval time
- `eval_gpu_memory_used_gb` - GPU mem at eval
- `overfit_ratio` - eval_loss / train_loss (detects overfitting!)

---

## 📁 Files Changed/Created

### New Files Created

1. **`colm/train/train_sequential_riemannian.py`** (850+ lines)
   - Main sequential training script
   - Implements load-once, train-many pattern
   - Enhanced monitoring with overfitting detection
   - WandB integration with single run
   - Validation split creation

2. **`configs/sequential_riemannian_config.json`** (Example config)
   - Pre-configured arguments for sequential training
   - Optimized batch sizes and learning rates
   - All model and data settings

### Files Unchanged (But Can Be Extended)

The following can be reused without changes:
- `colm/train/subset_trainer_distributed.py` - Trainer logic
- `colm/train/facility_location.py` - Selection algorithms
- `colm/train/optimizer_factory.py` - Optimizer creation
- Data loading utilities

---

## 🚀 Usage

### Option 1: Using JSON Config File

```bash
cd /data/riddhankur/PROJECTS/GREATS_COLM_pytorch/local/llm-experiments

python colm/train/train_sequential_riemannian.py \
    configs/sequential_riemannian_config.json
```

### Option 2: Using Command-Line Arguments

```bash
python colm/train/train_sequential_riemannian.py \
    --model_name_or_path meta-llama/Llama-2-7b \
    --lora_rank 128 \
    --dataset_path /path/to/dataset \
    --num_tasks 3 \
    --val_split_ratio 0.1 \
    --output_dir ./outputs/sequential \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --eval_steps 100 \
    --logging_steps 50 \
    --report_to wandb tensorboard
```

### Option 3: Programmatic Usage

```python
from colm.train.train_sequential_riemannian import (
    run_sequential_training,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

model_args = ModelArguments(
    model_name_or_path="meta-llama/Llama-2-7b",
    lora_rank=128,
    use_lora=True,
)

data_args = DataArguments(
    dataset_path="/path/to/dataset",
    num_tasks=3,
    val_split_ratio=0.1,
)

training_args = TrainingArguments(
    output_dir="./outputs",
    learning_rate=1e-4,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    report_to=["wandb"],
)

metrics = run_sequential_training(
    model_args=model_args,
    data_args=data_args,
    training_args=training_args,
    num_tasks=3,
    val_split_ratio=0.1,
)
```

---

## 📊 Key Arguments

### Model Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `model_name_or_path` | - | Model HF ID or path |
| `model_max_length` | 512 | Max sequence length |
| `use_lora` | true | Enable LoRA |
| `lora_rank` | 128 | LoRA rank (r) |
| `lora_alpha` | 512 | LoRA alpha (α) |
| `lora_dropout` | 0.05 | LoRA dropout rate |

### Data Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `dataset_path` | - | Path to dataset |
| `val_split_ratio` | 0.1 | Validation split % |
| `num_tasks` | 1 | Number of tasks |

### Training Arguments (HF Standard)

| Argument | Default | Description |
|----------|---------|-------------|
| `num_train_epochs` | 3 | Epochs per task |
| `per_device_train_batch_size` | 8 | Batch size |
| `learning_rate` | 1e-4 | Learning rate |
| `eval_steps` | 100 | Eval frequency |
| `logging_steps` | 50 | Log frequency |
| `bf16` | true | Use bfloat16 |
| `report_to` | ["wandb"] | Tracking backends |

---

## 🎯 Expected Behavior

### Task Progression (Cumulative Learning)

```
Initial Model State:
├─ Base weights (from pretrained)
└─ LoRA weights (initialized)

↓ Task 0 Training
├─ Takes: base + LoRA
├─ Trains: on Task 0 data
├─ Results: base + LoRA + Δ_task0
└─ Saves: checkpoint_task_0

↓ Task 1 Training (SAME MODEL OBJECT)
├─ Takes: base + LoRA + Δ_task0  ← Cumulative!
├─ Trains: on Task 1 data
├─ Results: base + LoRA + Δ_task0 + Δ_task1
└─ Saves: checkpoint_task_1

↓ Task 2 Training (SAME MODEL OBJECT)
├─ Takes: base + LoRA + Δ_task0 + Δ_task1  ← All accumulated!
├─ Trains: on Task 2 data
├─ Results: base + LoRA + Δ_task0 + Δ_task1 + Δ_task2
└─ Saves: checkpoint_task_2

Final Model: Contains knowledge from all 3 tasks
```

### Metrics Evolution (Expected Pattern)

```
WandB Dashboard:
┌─────────────────────────────────────────────┐
│         Training Curve (Single Run)          │
├─────────────────────────────────────────────┤
│                                             │
│   Loss ↓       ┌─────────────────────┐     │
│          \     │ Task 0 Training     │     │
│           \    │ (fresh start)       │     │
│            \___│ gradual decrease    │     │
│                │                     │     │
│                ├─────────────────────┤     │
│                │ Task 1 Training     │     │
│                │ (warmer start)      │  ↑  │ May
│        (Task 1 starts here,╲        │  F  │ show
│         continuing from \   \       │  o  │ higher
│         task 0 weights)  \   \___   │  r  │ initial
│                           \       \│  g  │ loss
│                ├─────────────────────┤     │
│                │ Task 2 Training     │     │
│                │ (warmest start)     │     │
│                │                     │     │
│                └─────────────────────┘     │
│                                             │
│ Each task shows step increases in loss     │
│ (task switching) but continues training    │
│                                             │
└─────────────────────────────────────────────┘

Key: Loss shows SINGLE continuous curve
     (not separate curves per task)
```

### Overfitting Detection

```
WandB shows: overfit_ratio = eval_loss / train_loss

Good (no overfitting):     ratio ≈ 1.0-1.2
Mild overfitting:          ratio ≈ 1.2-1.5
Severe overfitting:        ratio > 1.5

Lower val_split_ratio (use smaller validation set)
impacts this ratio - adjust as needed
```

---

## ⚙️ Customization

### Adjusting Sequential Behavior

```python
# More tasks, less training each
num_tasks=5,
num_train_epochs=1,

# Fewer tasks, more training each
num_tasks=2,
num_train_epochs=5,
```

### Adjusting Validation Monitoring

```python
# More frequent evaluation (detect overfitting quicker)
eval_steps=50,

# Larger validation set (more robust eval, less training)
val_split_ratio=0.2,

# Smaller validation set (less overhead, less signal)
val_split_ratio=0.05,
```

### Different Optimizers

```python
# AdamW (default, stable)
--optim adamw_torch

# Muon (faster, may diverge)
--optim muon

# SGD with momentum
--optim sgd
```

---

## 📈 Monitoring in WandB

### Charts to Create

1. **Training Loss Over Time**
   - Track: `loss` metric
   - Shows: Cumulative learning across all tasks

2. **Eval Loss vs Train Loss**
   - Track: `loss` vs `eval_loss`
   - Shows: Overfitting ratio

3. **GPU Memory Usage**
   - Track: `gpu_memory_used_gb`
   - Shows: Peak memory per task

4. **Task-Specific Metrics**
   - Track: `task_0_eval_loss`, `task_1_eval_loss`, etc.
   - Shows: Performance per task at completion

5. **Overfitting Ratio**
   - Track: `overfit_ratio`
   - Shows: When model starts overfitting

---

## 🐛 Debugging

### Common Issues

1. **OOM (Out of Memory)**
   - Reduce: `per_device_train_batch_size`
   - Reduce: `model_max_length`
   - Increase: `gradient_accumulation_steps`

2. **Training Too Slow**
   - Ensure: `bf16=true` for faster computation
   - Increase: `per_device_train_batch_size` (if memory allows)
   - Reduce: `eval_steps` (less frequent evaluation)

3. **Loss Not Decreasing**
   - Increase: `learning_rate`
   - Check: `grad_norm` (should not be 0 or inf)
   - Verify: Data is being loaded correctly

4. **High Overfitting**
   - Increase: `val_split_ratio` (more regularization signal)
   - Enable: Early stopping
   - Add: Weight decay or dropout

### Checking Logs

```bash
# Monitor training in real-time
tail -f logs/training.log

# Check wandb connection
python -c "import wandb; print(wandb.__version__)"

# Verify dataset loading
python -c "from datasets import load_from_disk; print(load_from_disk('/path/to/dataset'))"
```

---

## 🔄 Comparison: CoLM vs Sequential Riemannian vs Original Riemannian

| Aspect | Original CoLM | Sequential CoLM | Original Riemannian |
|--------|---------------|-----------------|-------------------|
| **Data** | All 14 sources mixed | Sequential per-task | Sequential per-task |
| **Model Loading** | Load once | Load once | Load once |
| **Task Loop** | Single training | Task loop | Task loop |
| **Cumulative Learning** | N/A | YES | YES |
| **WandB Runs** | 1 | 1 | Multiple |
| **Eval Loss Tracking** | Optional | YES (built-in) | YES |
| **Overfitting Detection** | Manual | Automatic | Manual |
| **Optimizer** | Muon/AdamW | Any HF optimizer | Riemannian SGD |
| **Final Model** | All data learned | Task 0+1+2+... | Task-specific heads |

---

## ✅ Validation Checklist

- [ ] Dataset path exists and contains valid data
- [ ] Model can be loaded from HF hub or local path
- [ ] WandB project is created or auto-created
- [ ] GPU has enough memory for batch size
- [ ] `num_tasks` > 1 for sequential training
- [ ] `eval_steps` is reasonable (not too frequent)
- [ ] Output directory is writable
- [ ] `report_to=["wandb"]` is set

---

## 📚 References

- Riemannian Training: `REIMANIAN_FINETUNE/src/run_experimet.py`
- Original CoLM: `colm/train/train.py`
- Checkpoint Flow Analysis: `CHECKPOINT_AND_TRAINING_FLOW.md`
- Dataset Loading: `colm/data/get_training_dataset.py`

---

## 💡 Tips for Best Results

1. **Start Small**: Test with 2-3 tasks first
2. **Monitor Early**: Check WandB in first few steps
3. **Adjust Learning Rate**: Higher LR for first task, lower for later
4. **Use Validation**: Always use non-zero `val_split_ratio`
5. **Save Checkpoints**: Enable `save_steps` for recovery
6. **Track Metrics**: Log everything to WandB for analysis

