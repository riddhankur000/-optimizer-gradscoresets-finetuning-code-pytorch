# Quick Reference: Codebase Changes for Sequential Riemannian Training

## 📝 What Was Changed?

### Files Created (3 Total)

#### 1. **Main Training Script** ✨ NEW
```
📍 Location: colm/train/train_sequential_riemannian.py
📊 Size: 850+ lines
🎯 Purpose: Sequential multi-task training with Riemannian checkpoint flow
```

**Key Features:**
- Loads model ONCE before task loop
- Creates validation split per task (90/10)
- Tracks train AND eval loss during training
- Single WandB run for all tasks
- Enhanced metrics: gradient norms, GPU usage, overfitting ratio
- Per-task metrics logging

**Main Function:**
```python
run_sequential_training(
    model_args,
    data_args,
    training_args,
    num_tasks,
    val_split_ratio=0.1
)
```

---

#### 2. **Example Configuration** 📋 NEW
```
📍 Location: configs/sequential_riemannian_config.json
📊 Size: 50 lines
🎯 Purpose: Drop-in configuration for sequential training
```

**Includes:**
- Model setup (Llama-2-7b, LoRA rank 128, etc.)
- Data configuration (3 tasks, 10% validation split)
- Training hyperparameters (batch size, learning rate, eval frequency)
- WandB reporting setup

---

#### 3. **Comprehensive Guide** 📚 NEW
```
📍 Location: SEQUENTIAL_RIEMANNIAN_TRAINING_GUIDE.md
📊 Size: 600+ lines
🎯 Purpose: End-to-end user guide for sequential training
```

**Contents:**
- Architecture overview
- Usage instructions (3 methods)
- Argument reference table
- Expected behavior visualization
- Customization guide
- WandB monitoring instructions
- Debugging tips
- Comparison with other methods

---

#### 4. **Implementation Details** 🔍 NEW
```
📍 Location: CODEBASE_CHANGES_SEQUENTIAL_TRAINING.md
📊 Size: 500+ lines
🎯 Purpose: Technical deep-dive on implementation
```

**Contents:**
- Detailed change breakdown
- Code snippets showing key mechanisms
- Model persistence implementation
- WandB single-run setup
- Metrics tracking explanation
- Execution flow diagrams
- Integration points

---

### Files Modified (0 Total)

✅ **NO existing CoLM code was changed**
- `colm/train/train.py` - Unchanged
- `colm/train/train_multitask.py` - Unchanged
- `colm/train/subset_trainer_distributed.py` - Unchanged
- All data loading utilities - Unchanged
- All optimizer utilities - Unchanged

**Why?** The sequential training is implemented as a completely new script that reuses existing components without modification.

---

## 🚀 Quick Start

### Method 1: Using Config File (Easiest)
```bash
python colm/train/train_sequential_riemannian.py \
    configs/sequential_riemannian_config.json
```

### Method 2: Command-Line Arguments
```bash
python colm/train/train_sequential_riemannian.py \
    --model_name_or_path meta-llama/Llama-2-7b \
    --num_tasks 3 \
    --dataset_path /path/to/data \
    --output_dir ./outputs \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-4 \
    --report_to wandb
```

### Method 3: Programmatic
```python
from colm.train.train_sequential_riemannian import run_sequential_training

metrics = run_sequential_training(
    model_args=ModelArguments(...),
    data_args=DataArguments(...),
    training_args=TrainingArguments(...),
    num_tasks=3,
    val_split_ratio=0.1,
)
```

---

## 🎯 Core Implementation (What Changed Under The Hood)

### Model Persistence (Heart of the Implementation)
```python
# OUTSIDE the task loop (loaded ONCE)
model = AutoModelForCausalLM.from_pretrained(...)
model = get_peft_model(model, lora_config)

# INSIDE the task loop (SAME model object persists)
for task_id in range(num_tasks):
    trainer = SubsetTrainerEfficient(model=model, ...)  # ← SAME model!
    trainer.train()  # Updates model in-place
    # model persists to next iteration with accumulated weights
```

### Single WandB Run
```python
# Initialize ONCE before loop
wandb.init(project="...", name=run_name, ...)

# Inside loop, keep logging to SAME run
for task_id in range(num_tasks):
    wandb.log({f"task_{task_id}/loss": ...})  # ← SAME run!
```

### Validation Split Per Task
```python
# Inside loop, create NEW validation split each task
for task_id in range(num_tasks):
    train_subset, val_subset = create_validation_split(
        dataset, 
        val_split_ratio=0.1  # 90% train, 10% val
    )
    trainer = SubsetTrainerEfficient(
        train_dataset=train_subset,
        eval_dataset=val_subset,  # ← Enables eval loss tracking
        ...
    )
```

---

## 📊 Metrics Tracked (What WandB Sees)

### Per Training Step
- `loss` - Training loss
- `learning_rate` - Current learning rate
- `epoch` - Current epoch number
- `global_step` - Total steps across all tasks
- `task_id` - Which task being trained

### Via Monitoring Callback
- `train_perplexity` - exp(loss)
- `grad_norm` - L2 norm of all gradients
- `grad_norm_avg` - Average per-parameter gradient norm
- `gpu_memory_used_gb` - GPU RAM usage
- `gpu_memory_utilization_%` - GPU utilization percentage
- `cpu_percent` - CPU usage
- `cpu_memory_percent` - System RAM percentage

### Per Evaluation
- `eval_loss` - Validation loss
- `eval_perplexity` - exp(eval_loss)
- `overfit_ratio` - eval_loss / train_loss (detects overfitting!)
- `eval_grad_norm` - Gradient norms at eval time
- `eval_gpu_memory_used_gb` - GPU mem at eval

### Task Summaries
- `task_0_train_loss` - Task 0 final training loss
- `task_0_eval_loss` - Task 0 final validation loss
- `task_1_train_loss` - Task 1 final training loss
- `task_1_eval_loss` - Task 1 final validation loss
- (etc for all tasks)

---

## 🔄 Architecture Comparison

| Aspect | Original CoLM | NEW Sequential | Original Riemannian |
|--------|---------------|-----------------|-------------------|
| **Training** | Single unified | Sequential loop | Sequential loop |
| **Model Loading** | Load once | Load ONCE before loop | Load ONCE before loop |
| **Data** | All mixed at once | One dataset per iteration | One dataset per iteration |
| **WandB Runs** | 1 | 1 (single) | 1 per task (multiple) |
| **Cumulative Learning** | N/A | ✅ YES | ✅ YES |
| **Eval Loss During Train** | Optional | ✅ YES (built-in) | ✅ YES (built-in) |
| **Overfitting Detection** | Manual | ✅ Automatic | Manual |
| **New Code** | - | train_sequential_riemannian.py | - |

---

## ✅ Validation Checklist

Before running:
- [ ] Dataset exists at `dataset_path`
- [ ] Model can be downloaded (or local path valid)
- [ ] GPU has enough memory (~20GB for 7B model + LoRA)
- [ ] WandB account configured (or local tensorboard)
- [ ] `num_tasks` ≥ 2 (sequential training)
- [ ] `output_dir` is writable

After running (check in WandB):
- [ ] Single run showing multiple tasks
- [ ] Loss decreasing per task (not resetting)
- [ ] `overfit_ratio` < 1.5 (no excessive overfitting)
- [ ] GPU memory stable (not growing unbounded)
- [ ] Checkpoints saved for each task

---

## 📈 Expected WandB Output

### Training Curve (All Tasks in One Graph)
```
Loss ↓
  │     Task 0        Task 1         Task 2
  │   ╱─────╲      ╱──────╲      ╱────────╲
  │  ╱        ╲    ╱        ╲    ╱          ╲
  │ ╱          ╲──╱          ╲──╱            ╲──
  │
  └────────────────────────────────────────────→ Training Steps

✓ Single continuous curve (not three separate curves)
✓ Each task shows loss increase at start (new task data)
✓ Loss continues to decrease within each task
✓ Horizontal dips show eval periods
```

### Key Metrics
- **Run Name**: `phi-2_adamw_lora_r128_1e-4`
- **Total Steps**: 300 (100 steps/task × 3 tasks)
- **Duration**: ~2-3 hours (varies by model size)
- **Final Model**: Contains knowledge from Tasks 0+1+2

---

## 🐛 Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| OOM error | Batch too large | Reduce `per_device_train_batch_size` |
| Loss not decreasing | Learning rate too low | Increase `learning_rate` |
| WandB not logging | Not installed/configured | `pip install wandb` + `wandb login` |
| High overfitting | Too little validation data | Increase `val_split_ratio` to 0.2 |
| Training very slow | Eval too frequent | Increase `eval_steps` to 200+ |
| Checkpoint not saving | Path doesn't exist | Create `output_dir` first |

---

## 📚 Documentation Guide

**Quick overview?** → Start with **CHECKPOINT_AND_TRAINING_FLOW.md** (you already have this)

**Want to run?** → Follow **SEQUENTIAL_RIEMANNIAN_TRAINING_GUIDE.md**

**Technical deep-dive?** → Read **CODEBASE_CHANGES_SEQUENTIAL_TRAINING.md**

**Need implementation details?** → Check inline docs in **train_sequential_riemannian.py**

---

## 🎯 Summary

### What You Get
✅ Sequential Riemannian-style training for CoLM
✅ Cumulative learning across tasks (model persists)
✅ Automatic eval loss tracking for overfitting detection
✅ Single WandB run showing continuous training
✅ 15+ metrics tracked automatically
✅ Per-task performance summaries

### How Much Code Was Changed
📊 **Total Lines Changed**: ~0 (no modifications to existing code)
📊 **Total Lines Added**: 850+ (new training script)
📊 **Files Modified**: 0
📊 **Files Created**: 3 (1 Python + 1 config + 1 guide)
📊 **Breaking Changes**: None (fully compatible)

### Key Innovation
🔑 **Model loaded ONCE, reused for all tasks** = Cumulative learning
🔑 **Single WandB run** = Continuous monitoring
🔑 **Per-task validation** = Automatic overfitting detection

---

## 🚀 Next Steps

1. **Review**: Read SEQUENTIAL_RIEMANNIAN_TRAINING_GUIDE.md
2. **Customize**: Edit configs/sequential_riemannian_config.json for your setup
3. **Test**: Run with a small model on a small dataset first
4. **Monitor**: Watch WandB dashboard during training
5. **Analyze**: Compare results with original CoLM approach

---

## 📞 Quick Reference

**File Locations:**
```
Training Script:    colm/train/train_sequential_riemannian.py
Config Example:     configs/sequential_riemannian_config.json
User Guide:         SEQUENTIAL_RIEMANNIAN_TRAINING_GUIDE.md
Technical Details:  CODEBASE_CHANGES_SEQUENTIAL_TRAINING.md
Flow Explanation:   CHECKPOINT_AND_TRAINING_FLOW.md
```

**Quick Commands:**
```bash
# Run with config
python colm/train/train_sequential_riemannian.py configs/sequential_riemannian_config.json

# Run with CLI args
python colm/train/train_sequential_riemannian.py --num_tasks 3 --dataset_path /data ...

# Check WandB
wandb online  # Verify WandB is configured
```

