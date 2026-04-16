# Dataset Collection & Training Strategy: CoLM vs Riemannian

**Question**: Are they training on entire 14 dataset at once OR one by one with sequential evaluation?

---

## 🎯 Quick Answer

| Approach | Strategy |
|----------|----------|
| **CoLM** | ✅ **ALL 14 sources at once** - Mixed dataset training |
| **Riemannian** | ❌ **One task/dataset at a time** - Sequential training |

---

## 📊 CoLM: Multi-Source Unified Training

### Data Collection Strategy
```
MathInstruct with 14 sources:
├─ Aqua-RAT
├─ MATH
├─ GSM8K
├─ SAT
├─ SciBench
├─ OMW-MathAlpaca
├─ ... (8 more sources)
└─ Source 14 (total 14)

All mixed into ONE unified dataset
with total ~250K-300K samples
```

### How They Load Data

**Location**: `data/get_training_dataset.py`

```python
def load_raw_dataset(train_files: Union[List[str], str], ...):
    # Load ALL 14 sources at once
    processed_datasets = load_dataset(
        "json",
        data_files=train_files,  # ALL jsonl files
    )["train"]
    
    # Result: Single dataset with 300K samples from 14 sources
    return processed_datasets
```

**Key: They specify ONE data_files parameter with ALL sources!**

```python
# In training script
DATA_DIR=./data
PERCENTAGE=1  # 100% of ALL data (all 14 sources)
```

### Training Pipeline

```
One Unified Batch Loop:
├─ Epoch 1: Mix of all 14 sources
│  ├─ Batch 1: [Sample from source-0, source-3, source-7, ...]
│  ├─ Batch 2: [Sample from source-1, source-5, source-12, ...]
│  ├─ ... (per-batch selection happens HERE)
│  └─ Batch N: Mixed sources → Select K best
│
├─ Epoch 2: Same mixed dataset loop
└─ ... (continues for all MAX_STEPS)

Total Training: Single unified training loop on mixed data
```

### Per-Batch Selection With Source Constraints

**Important**: Selection happens per-batch with source awareness:

```python
# In subset_trainer_distributed.py
for batch in train_loader:
    # Batch contains samples from mix of 14 sources
    
    # Source-wise handling option
    SOURCE_WISE=proportional  # Maintain source distribution
    # vs
    SOURCE_WISE=balanced      # Equal samples per source
    # vs
    SOURCE_WISE=none          # Ignore source imbalance
    
    # Selection: K best from this batch (~50% reduction)
    selected_samples = select_method(batch, method=greats)
    
    # Train on selected K samples
    loss = train_on_selected(selected_samples)
```

### Evaluation Strategy

**After training completes**:
```python
# Evaluate on ALL 17 benchmarks at once
6_math_datasets = [MATH, GSM8K, SVGD, ARC, StrategyQA, Minerva]
11_nlu_tasks = [RTE, CB, MultiRC, CoLA, SST-2, QNLI, QQP, MNLI, MRPC, STS-B, BoolQ]

for benchmark in (6_math_datasets + 11_nlu_tasks):
    metrics = evaluate_on(benchmark)
    log_metrics(metrics)
```

**Result**: Single comprehensive evaluation run with all 17 benchmarks

---

## Riemannian: Per-Task Sequential Training

### Data Collection Strategy

```
Load one dataset at a time:
├─ Task 0: Load MathInstruct (all sources combined)
│  ├─ Train on full dataset
│  ├─ Evaluate on task 0
│  └─ Save checkpoint
│
├─ Task 1: Load next dataset (e.g., SuperGLUE subset)
│  ├─ Train on full dataset  
│  ├─ Evaluate on task 1
│  └─ Save checkpoint
│
└─ Task N: Load next dataset
   ├─ Train on full dataset
   ├─ Evaluate on task N
   └─ Final checkpoint
```

### How They Load Data

**Location**: `src/run_experimet.py`

```python
def run_tasks(config):
    tasks = config.tasks  # List of tasks to run
    tokenizer = model_loader.load_tokenizer(config)
    model = model_loader.load_model(config)
    
    dataset = data_preparation.load_dataset(config)
    # dataset has train/validation/test splits
    
    for i, task in enumerate(tasks):
        print(f"Running task {i}: {task}")
        
        if task is Task.FINETUNE:
            # Train on THIS task's data
            run_finetune(config, model, tokenizer, 
                        dataset['train'],        # One dataset
                        dataset['validation'])   # One validation set
        
        elif task is Task.VALIDATE:
            # Evaluate on THIS task
            pl = model_loader.get_pipeline(config, model, tokenizer)
            run_inference(config, pl, dataset['validation'])
        
        elif task is Task.INFERENCE:
            # Inference on THIS task
            run_inference(config, pl, dataset['test'])
```

**Key: They load ONE dataset per task iteration!**

### Training Pipeline

```
Sequential Task Loop:
├─ Task 0: Load dataset-0
│  ├─ for epoch in epochs:
│  │  ├─ for batch in train_loader:
│  │  │  ├─ Forward + Backward (Riemannian manifold)
│  │  │  └─ Optimizer step (RiemannianSGD)
│  │  └─ Validate on task-0 validation
│  └─ Save model after task-0
│
├─ Task 1: Load dataset-1 (fresh data)
│  ├─ for epoch in epochs:
│  │  ├─ for batch in train_loader:
│  │  │  ├─ Forward + Backward (Riemannian manifold)
│  │  │  └─ Optimizer step (RiemannianSGD)
│  │  └─ Validate on task-1 validation
│  └─ Save model after task-1
│
└─ Task N: ...

Total Training: N sequential training loops on different datasets
```

### Per-Task Evaluation

**After training on one task**:
```python
# Evaluate only on THIS task
result = run_inference(config, model, dataset['validation'])
result.to_parquet(f'predictions/task_{task_idx}.parquet')

# Then move to next task
```

---

## 📈 Comparison Table

| Aspect | CoLM | Riemannian |
|--------|------|-----------|
| **Data Collection** | ALL 14 sources at once | One dataset at a time |
| **Dataset Type** | Mixed (MathInstruct all sources) | Per-task specific dataset |
| **Training Loop** | Single unified mixed-data loop | N sequential loops (N=# tasks) |
| **Batch Composition** | Samples from multiple sources | Samples from single source |
| **Source Handling** | Source-wise constraints (proportional/balanced) | N/A (single dataset) |
| **Selection Strategy** | Per-batch coreset selection | No selection (full batch) |
| **Evaluation** | After training: 17 benchmarks at once | After each task: evaluate that task only |
| **Evaluation Loop** | 1 comprehensive evaluation run | N sequential evaluation runs |
| **Model Updates** | Continuous over all 14 sources | Sequential per task (may overfit to task N) |

---

## 📊 Training Timeline Comparison

### CoLM Timeline
```
Time: 4-6 hours
│
├─ Hour 0-0.5: Load all 14 sources (mixed) into memory
├─ Hour 0.5-5.5: Train on mixed dataset with per-batch selection
│  ├─ Epoch 1: Process all 14 sources mixed
│  ├─ Epoch 2: Process all 14 sources mixed
│  └─ Epoch N: Process all 14 sources mixed
├─ Hour 5.5-6: Comprehensive evaluation
│  ├─ Evaluate on MATH
│  ├─ Evaluate on GSM8K
│  ├─ ... (6 math datasets)
│  ├─ ... (11 SuperGLUE tasks)
│  └─ Log all 17 metrics
└─ Total: 1 training run + 1 evaluation run
```

### Riemannian Timeline
```
Time: Varies (if N tasks, scaled by N)
│
├─ Task 0: Load dataset-0 (e.g., MathInstruct)
│  ├─ Train for X hours
│  ├─ Evaluate on task-0 validation
│  └─ Save
├─ Task 1: Load dataset-1 (e.g., different benchmark)
│  ├─ Train for X hours (from checkpoint after task-0)
│  ├─ Evaluate on task-1 validation
│  └─ Save
└─ Task N: Load dataset-N
   ├─ Train for X hours
   ├─ Evaluate on task-N validation
   └─ Final checkpoint

Total: N training runs + N evaluation runs
```

---

## 🔑 Key Differences in Philosophy

### CoLM Philosophy: Multi-Source Unified Learning
```
Problem: How to efficiently train on mixed imbalanced sources?

Solution:
├─ Load all sources together
├─ Handle source imbalance with constraints
├─ Select most informative samples per batch
└─ Evaluate comprehensively after training

Benefit: 
- Learns shared representations across all 14 sources
- Handles source imbalance elegantly with source-wise constraints
- Comprehensive evaluation on all tasks at once
```

### Riemannian Philosophy: Per-Task Optimization
```
Problem: How to optimize each task specifically?

Solution:
├─ Load one task at a time
├─ Train with manifold-aware optimization
├─ Evaluate on that specific task
└─ Move to next task (sequential)

Benefit:
- Task-specific optimization
- Independent evaluation per task
- Can fine-tune for each task specifically
```

---

## 💡 Why This Matters

### CoLM's Unified Approach Enables:
1. **Source balancing**: Handle 300:1 imbalance across 14 sources
2. **Transfer learning**: Learn shared representations
3. **Batch diversity**: Each batch has multiple sources
4. **Efficient selection**: Select K best from diverse samples
5. **Comprehensive metrics**: One evaluation covers all 17 benchmarks

### Riemannian's Sequential Approach Enables:
1. **Task specialization**: Fully optimize for each task
2. **Simple pipeline**: One task at a time
3. **Independence**: Each task trained/evaluated separately
4. **Checkpointing**: Save after each task
5. **Flexibility**: Different hyperparameters per task

---

## 📝 Summary

**Your Question**: Training on all 14 at once vs one by one?

**Answer**:
- ✅ **CoLM**: All 14 sources at once in a unified mixed dataset
- ❌ **Riemannian**: One task/dataset at a time sequentially

This is a fundamental architectural choice reflecting their different goals:
- CoLM: Multi-source efficiency through intelligent selection
- Riemannian: Per-task optimization through manifold geometry

