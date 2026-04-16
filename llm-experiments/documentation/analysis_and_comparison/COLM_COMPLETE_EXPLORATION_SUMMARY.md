# CoLM Training Codebase - COMPLETE EXPLORATION SUMMARY

Generated: April 15, 2026

---

## OVERVIEW

You now have **3 comprehensive documents** exploring the CoLM training codebase:

1. **COLM_TRAINING_CODEBASE_EXPLORATION.md** (Main reference)
   - Detailed code locations and line numbers
   - Function signatures and flow
   - Configuration options
   - Wandb integration details

2. **COLM_ARCHITECTURE_VISUAL.md** (Visual guide)
   - System architecture diagrams
   - Data selection loop flowcharts
   - Representation extraction pipeline
   - Loss tracking architecture

3. **COLM_SEQUENTIAL_IMPLEMENTATION_GUIDE.md** (Implementation guide)
   - Exact code injection points
   - Before/after examples
   - Command-line examples
   - Testing procedures

---

## QUICK FACTS

### Repository Location
```
/data/riddhankur/PROJECTS/GREATS_COLM_pytorch/local/llm-experiments
```

### Key Entry Points
- **Single-task**: `colm/train/train.py`
- **Multi-task**: `colm/train/train_multitask.py`
- **Core engine**: `colm/train/subset_trainer_distributed.py` (SubsetTrainer class)

### Training Types

| Mode | File | Use Case |
|------|------|----------|
| Standard HF | train.py + Trainer | No data selection |
| CoLM | train.py + SubsetTrainer | Batch-wise data selection |
| Multi-task | train_multitask.py + MultiTaskTrainer | Multiple tasks with config |

---

## TRAINING PIPELINE FLOW

```
Input → Parse Args → Load Model + Tokenizer + LoRA
  ↓
Load Dataset (SuperGLUE or Generic)
  ↓
Choose Trainer Class:
  - data_selection_method="none" → HF Trainer
  - data_selection_method="greats"|"fairot"|etc → SubsetTrainer
  ↓
Main Training Loop (_inner_training_loop):
  FOR epoch in epochs:
    FOR batch in dataloader:
      [Accumulate Phase] Collect B batches' representations
      [Selection Phase] Select K < B samples (rank 0 only, then broadcast)
      [Training Phase] Train on selected K samples
      [Logging Phase] Log metrics to wandb
      [Eval Phase] Evaluate periodically on eval set
  ↓
Save Model + Metrics
```

---

## DATA SELECTION MECHANISM (Core Innovation)

### 1. What Gets Selected?
**Representation units** (controlled by `data_selection_unit`):
- `"rep"` → Last token hidden state [B, hidden_dim]
- `"mezo"` → Zeroth-order gradient [B×params,]
- `"masked_grad"` → Last-N-layer gradients via backprop
- `"completion_length"` → Sequence length (scalar)
- `"length_loss_weighted"` → Length × loss product

### 2. How Are They Selected?
**Selection algorithms** (controlled by `data_selection_method`):
- `"greats"` → Gradient-based coreset selection (diversity + representativeness)
- `"fairot"` → Fair outlier truncation
- `"spot"` → SPOT greedy subset selection
- `"submodlib"` → Facility location-based (submodular)
- `"none"` → No selection (standard HF Trainer)

### 3. Key Parameters
```python
small_batch_ratio = K/B                    # E.g., 0.1 = select top 10%
gradient_accumulation_steps = B            # How many batches to accumulate
data_selection_method = "greats"           # Which algorithm
data_selection_unit = "rep"                # What to extract
```

**Formula**: `K_selected = B × small_batch_ratio`
- B = number accumulated batches
- K = selected samples trained on

---

## WHERE TO ADD SEQUENTIAL TASK TRAINING

### Three Key Injection Points:

**Point 1: Per-Task Eval Loss Tracking**
- File: `colm/train/train_multitask.py` (MonitoringCallback)
- What: Add per-task `eval_loss_task_{name}` metrics
- Why: Track how each task performs individually
- Line: ~170 (on_evaluate method)

**Point 2: Sequential Training Loop**
- File: `colm/train/train_multitask.py` (main function)
- What: Wrap trainer.train() to loop over tasks 1-by-1
- Why: Train pure sequential instead of mixed
- Line: ~400 (main function)

**Point 3: Task-Aware Selection (Optional)**
- File: `colm/train/subset_trainer_distributed.py` (select_data method)
- What: Balance K selections across tasks
- Why: Ensure each task gets represented during mixed training
- Line: ~1358 (select_data method)

### Add New Configuration Args
- File: `colm/train/training_arguments.py`
- Add: `sequential_training`, `task_aware_selection`, `per_task_eval`

---

## WANDB METRICS LOGGING

### Automatic Logging (by HF Trainer)
```
training/loss per step
training/learning_rate
training/epoch
```

### Additional Logging (train_multitask.py's MonitoringCallback)
```
train_perplexity = exp(loss)          # Training perplexity
eval_perplexity = exp(eval_loss)      # Eval perplexity
grad_norm                              # Gradient L2 norm
gpu_memory_used_gb                     # GPU memory
gpu_memory_utilization_%               # GPU util %
gpu_load_%                             # GPU compute load
cpu_percent                            # CPU usage
cpu_memory_percent                     # System memory %
cpu_memory_available_gb                # Free memory
```

### Your New Metrics (after implementation)
```
eval_loss_task_math                   # Per-task eval loss
eval_loss_task_logic
eval_loss_task_qa
eval_perplexity_task_math              # Per-task perplexity
eval_perplexity_task_logic
eval_perplexity_task_qa
```

---

## DATA FLOW DIAGRAM

```
┌──────────────────────────────────────────────────────────────┐
│ INPUT: MultiTask Dataset                                      │
│ (Examples with task field: 'task1', 'task2', 'task3')        │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────────┐
│ CONFIGURATION                                                  │
│ sequential_training=true/false                                │
│ data_selection_method="greats"                               │
│ small_batch_ratio=0.5  (K/B ratio)                          │
│ per_task_eval=true                                           │
└────────────────────────┬──────────────────────────────────────┘
                         │
         ┌───────────────┴────────────────┐
         │                                 │
         ▼                                 ▼
    ┌─SEQUENTIAL_MODE──┐        ┌──MIXED_MODE──┐
    │ Train each task  │        │ All tasks    │
    │ one at a time    │        │ mixed        │
    │ (Loop over tasks)│        │ (Single pass)│
    └───────┬──────────┘        └──────┬───────┘
            │                          │
            └──────────┬───────────────┘
                       │
      ┌────────────────▼────────────────┐
      │SubsetTrainer._inner_training_loop
      │ FOR each epoch:
      │   FOR each batch:
      │     Accumulate reps (B batches)
      │     Select K < B via algorithm
      │     Train selected K
      │     Log metrics
      │     (Periodic eval)
      └────────────────┬────────────────┘
                       │
      ┌────────────────▼────────────────┐
      │ MonitoringCallback
      │ - Compute perplexity
      │ - Compute grad norms  
      │ - Per-task eval metrics [NEW]
      │ - GPU/CPU stats
      └────────────────┬────────────────┘
                       │
      ┌────────────────▼────────────────┐
      │ WandB Dashboard
      │ - Training curves
      │ - Eval curves (per-task) [NEW]
      │ - Resource usage
      └────────────────────────────────┘
```

---

## KEY INSIGHTS FOR MODIFICATIONS

### 1. Where's the Selection?
In `subset_trainer_distributed.py` `_inner_training_loop()` (lines 630-950):
- Representations collected in `total_reps` list
- Gathered from all ranks to rank 0
- `select_data()` called to pick K from B
- Broadcast back to all ranks
- Each rank trains its shard of K

### 2. How to Track Loss Per Task?
Add in `MonitoringCallback.on_evaluate()`:
- Filter eval_dataset by task ID
- Call `trainer.evaluate()` on each task subset
- Log `eval_loss_task_{name}` for each

### 3. How to Train Sequentially?
Add in `main()` function:
- Loop over unique tasks in dataset
- Filter train_dataset to 1 task at a time
- Update `trainer.train_dataset` 
- Call `trainer.train()` for each task
- Collect results per task

### 4. What About Selection During Sequential?
Use existing `source_wise_selection` + `keep_sources`:
- Or implement `task_aware_selection` in `select_data()`
- Filter representations by task before selection
- Select K from task subset

---

## IMPORTANT PARAMETERS REFERENCE

| Parameter | Type | Default | Impact |
|-----------|------|---------|--------|
| `data_selection_method` | str | "none" | Which algorithm (greats, fairot, etc) |
| `data_selection_unit` | str | "mezo" | What to extract (rep, grad, etc) |
| `small_batch_ratio` | float | 1.0 | K/B ratio for selection |
| `gradient_accumulation_steps` | int | 1 | Buffer size (B) |
| `source_wise_selection` | str | "proportional" | How to balance sources |
| `wandb_entity` | str | "" | WandB team name |
| `wandb_project` | str | "" | WandB project name |
| `wandb_notes` | str | "" | Run notes/tags |
| `sequential_training` | bool | False | **[NEW]** Train tasks sequentially |
| `per_task_eval` | bool | True | **[NEW]** Log per-task metrics |
| `task_aware_selection` | bool | False | **[NEW]** Balance selection by task |

---

## FILE STRUCTURE

```
colm/train/
├── train.py                      # Single-task entry point
├── train_multitask.py            # Multi-task entry point (modify here 1,2)
├── subset_trainer_distributed.py # Core engine (modify here 3)
├── training_arguments.py         # Config (add args here)
├── data_arguments.py
├── model_arguments.py
├── config_loader.py              # YAML config support
│
├── [Selection Algorithms]
├── greats.py                     # GREATS algorithm
├── fairot.py                     # FairOT v1
├── fairot2.py                    # FairOT v2
├── facility_location.py          # Facility location (submodlib)
├── SPOTgreedy.py                 # SPOT greedy
│
├── [Data Pipeline]
├── data/
│   ├── __init__.py
│   ├── get_training_dataset.py   # Load & preprocess data
│   ├── tasks.py                  # SuperGLUE task definitions
│   └── utils.py
│
└── [Utilities]
    ├── utils.py                  # General utilities
    ├── optimizer_factory.py       # Create optimizers from config
    └── huggingface_trainer.py     # Custom HF Trainer wrapper
```

---

## COMMAND PATTERNS

### Pattern 1: Run Existing Single-Task
```bash
python colm/train/train.py \
    --model_name_or_path <model> \
    --train_files <data_paths> \
    --output_dir <output> \
    --data_selection_method greats \
    --small_batch_ratio 0.5
```

### Pattern 2: Run Existing Multi-Task Mixed
```bash
python colm/train/train_multitask.py \
    --model_name_or_path <model> \
    --dataset_path <dataset> \
    --output_dir <output> \
    --data_selection_method fairot
```

### Pattern 3: NEW - Sequential Tasks
```bash
python colm/train/train_multitask.py \
    --model_name_or_path <model> \
    --dataset_path <dataset> \
    --output_dir <output> \
    --sequential_training \
    --per_task_eval
```

### Pattern 4: NEW - Mixed with Task Balancing
```bash
python colm/train/train_multitask.py \
    --model_name_or_path <model> \
    --dataset_path <dataset> \
    --output_dir <output> \
    --data_selection_method greats \
    --task_aware_selection \
    --per_task_eval
```

---

## TESTING STRATEGY

1. **Test per-task eval** (no selection, simple dataset)
2. **Test sequential mode** (small dataset, 2-3 tasks)
3. **Test with selection** (task-aware selection active)
4. **Full integration** (all features, real datasets)

---

## NEXT STEPS

1. **Read** `COLM_TRAINING_CODEBASE_EXPLORATION.md` for full architecture detail
2. **Study** `COLM_ARCHITECTURE_VISUAL.md` for visual understanding
3. **Implement** using `COLM_SEQUENTIAL_IMPLEMENTATION_GUIDE.md`
4. **Test** with small dataset first (gpt2 + custom data)
5. **Deploy** on full dataset with monitoring

---

## CONCLUSION

The CoLM framework is built on a **batch-wise data selection** approach:

```
STANDARD: Train all B samples per batch
CoLM:     Accumulate B → Select K → Train K
```

Key Files to Understand:
- `subset_trainer_distributed.py` - Where selection happens
- `training_arguments.py` - Configuration options  
- `train_multitask.py` - Multi-task + enhanced logging

For Sequential Training, inject code at:
1. **Per-task eval**: MonitoringCallback (line ~170)
2. **Sequential loop**: main() function (line ~400)
3. **Task-aware selection**: select_data() method (line ~1358)
4. **New args**: TrainingArguments (line ~240)

All references include exact line numbers and complete code examples.

---

**Document Set Generated**: April 15, 2026
**Codebase Location**: `/data/riddhankur/PROJECTS/GREATS_COLM_pytorch/local/llm-experiments`
**Repository**: CoLM (ICLR 2025 - Data-Centric Fine-Tuning)

