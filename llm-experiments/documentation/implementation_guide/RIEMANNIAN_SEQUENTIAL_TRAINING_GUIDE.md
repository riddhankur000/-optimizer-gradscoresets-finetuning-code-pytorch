  # Riemannian Sequential Training with Config.yaml and Task-Specific Datasets

## Overview

This guide explains how to configure and run **Riemannian sequential training** using the updated `config.yaml` file with **task-specific datasets**. The Riemannian method trains a model sequentially on different tasks, where the model persists across tasks and accumulates knowledge cumulatively.

**Key Principle**: Model is loaded ONCE before the task loop and trained on different tasks sequentially, with weights persisting from one task to the next.

## Configuration Updates

### 1. Dataset Configuration

The `dataset_config` section now supports two loading strategies:

```yaml
dataset_config:
  # Dataset loading strategy
  loading_strategy: "sequential"  # "single" or "sequential"
  
  # For single dataset (backward compatible)
  dataset_path: "./colm_math_combined_dataset"
  dataset_names:
    - "MetaMathQA"
    - "GSM8K"
  
  # Preprocessing options
  max_seq_length: 512
  num_proc: 4
  val_split_ratio: 0.1  # Used for both strategies
```

### 2. Sequential Tasks Configuration

NEW section for Riemannian training:

```yaml
sequential_tasks_config:
  enabled: true  # Enable sequential task training
  
  # List of tasks to train on sequentially
  tasks:
    - "SST2"           # Task 0: Sentiment classification
    - "RTE"            # Task 1: Textual entailment
    - "BoolQ"          # Task 2: Question answering
  
  # Per-task configuration
  samples_per_task: -1  # -1 means use all samples
  val_split_ratio: 0.1
  task_epochs: 1
  task_max_steps: null
```

### 3. Multi-Task Configuration

Updated to support dynamic task count:

```yaml
multitask_config:
  num_tasks: 3  # Auto-set when using sequential_tasks_config
  track_per_task_loss: true
  eval_per_task: true
```

## Available Tasks

The following tasks are available for sequential training:

| Task ID | Name | Type | Domain |
|---------|------|------|--------|
| 1 | SST2 | Classification | Sentiment Analysis |
| 2 | Copa | Classification | Commonsense Reasoning |
| 3 | BoolQ | Classification | Question Answering |
| 4 | MultiRC | Classification | Multi-choice Reading Comp. |
| 5 | CB | Classification | Commitment Bank |
| 6 | WIC | Classification | Word-in-Context |
| 7 | WSC | Classification | Winograd Schema Challenge |
| 8 | ReCoRD | Classification | Reading Comprehension |
| 9 | RTE | Classification | Textual Entailment |
| 10 | SQuAD | Generation | Question Answering |
| 11 | DROP | Generation | Discrete Reasoning |

## Quick Start Examples

### Example 1: Basic Sentiment + NLI Task Progression

```yaml
sequential_tasks_config:
  enabled: true
  tasks:
    - "SST2"      # Sentiment (easy, 8k samples)
    - "RTE"       # NLI (medium, 2.5k samples)
```

**Training Flow**:
```
Load Model
   ↓
Task 0 (SST2): Train on sentiment
   ↓
Task 1 (RTE): Train on entailment (model has sentiment knowledge)
   ↓
Final Model: Has learned both sentiment and entailment patterns
```

### Example 2: SuperGLUE Tasks

```yaml
sequential_tasks_config:
  enabled: true
  samples_per_task: 1000  # Use 1000 samples per task
  tasks:
    - "Copa"      # Commonsense
    - "MultiRC"   # Reading comprehension
    - "WIC"       # Word sense
    - "CB"        # Commitment bank
```

### Example 3: Reading Comprehension Progression

```yaml
sequential_tasks_config:
  enabled: true
  tasks:
    - "SQuAD"     # Standard QA
    - "DROP"      # Discrete reasoning QA
```

### Example 4: Using Single Dataset (Legacy Mode)

To use the old single dataset approach:

```yaml
sequential_tasks_config:
  enabled: false  # Disable sequential tasks

dataset_config:
  loading_strategy: "single"
  dataset_path: "./colm_math_combined_dataset"
```

## Running Sequential Training

### Option 1: Using Default Config

```bash
cd /data/riddhankur/PROJECTS/GREATS_COLM_pytorch/local/llm-experiments

# Edit config.yaml to set your tasks
nano config.yaml

# Run training
bash train.sh
```

### Option 2: Using Bash Wrapper with Overrides

```bash
# Run with optimizer override
bash train.sh --optimizer adamw --gpu gpu_multi

# Run with custom config
bash train.sh --config ./configs/my_riemannian_setup.yaml
```

### Option 3: Direct Python Execution

```bash
python colm/train/train_sequential_from_config.py ./config.yaml
```

## Configuration Examples

### Config 1: Adaptive Task Progression (Difficulty Increasing)

```yaml
sequential_tasks_config:
  enabled: true
  tasks:
    - "SST2"      # Binary classification (simplest)
    - "RTE"       # Binary entailment
    - "BoolQ"     # Reading comprehension
    - "MultiRC"   # Multi-choice reading
  samples_per_task: 2000
```

### Config 2: Domain-Specific (NLI Focus)

```yaml
sequential_tasks_config:
  enabled: true
  tasks:
    - "CB"        # Commitment Bank (NLI)
    - "RTE"       # Recognizing Textual Entailment
    - "WSC"       # Winograd Schema (NLI-like)
  val_split_ratio: 0.15
```

### Config 3: Fast Experimentation (Limited Samples)

```yaml
sequential_tasks_config:
  enabled: true
  tasks:
    - "SST2"
    - "RTE"
    - "BoolQ"
  samples_per_task: 500  # Quick testing
  task_max_steps: 100     # Limit steps per task
```

## Training Output and Metrics

When using sequential task training, you'll see output like:

```
================================================================================
        SEQUENTIAL TASKS SUMMARY (RIEMANNIAN METHOD)
================================================================================
Task 0: SST2                 | Train: 67349 | Val:  7483 | Total: 74832
Task 1: RTE                  | Train:  2275 | Val:   274 | Total:  2549
Task 2: BoolQ                | Train:  9427 | Val:  1048 | Total: 10475
--------------------------------------------------------------------------------
TOTAL                        | Train: 79051 | Val:  8805 | Total: 87856
================================================================================
Training Strategy: Sequential (model persists across tasks)
Learning Approach: Cumulative (weights updated per task)
================================================================================
```

### Per-Task Metrics

For each task, the system logs and tracks:

```
TASK 0: SST2
  Train loss: 0.4532
  Eval loss: 0.5123
  
TASK 1: RTE  
  Train loss: 0.3456 (starting from SST2-tuned weights)
  Eval loss: 0.4012
  
TASK 2: BoolQ
  Train loss: 0.2890 (starting from SST2+RTE-tuned weights)
  Eval loss: 0.3451
```

### WandB Tracking

Each task's metrics are logged to WandB under:
- `task_0/train_loss` - Task 0 training loss
- `task_0/eval_loss` - Task 0 evaluation loss
- `task_1/train_loss` - Task 1 training loss (with task 0 knowledge)
- ... and so on

## Model Persistence and Cumulative Learning

### How It Works

1. **Model Loaded Once**:
   ```python
   model = AutoModelForCausalLM.from_pretrained(model_path)
   model = get_peft_model(model, lora_config)  # LoRA added once
   ```

2. **Sequential Training**: 
   ```python
   for task_id in range(num_tasks):
       trainer = Trainer(model=model, ...)  # Same model object!
       trainer.train()  # Updates model.parameters() in-place
   ```

3. **Weights Persist**:
   ```
   Initial Weights (W0)
        ↓ (trained on Task 0)
   Updated Weights (W0 + ΔW0) = W1
        ↓ (trained on Task 1, starting from W1)
   Updated Weights (W1 + ΔW1) = W2
        ↓ (trained on Task 2, starting from W2)
   Final Weights (W2 + ΔW2) = W_final
   ```

## Adapting Your Own Dataset Configuration

### Step 1: Modify config.yaml

Replace the `sequential_tasks_config` section:

```yaml
sequential_tasks_config:
  enabled: true
  
  tasks:
    - "Copa"          # Your task 1
    - "MultiRC"       # Your task 2  
    - "WIC"           # Your task 3
  
  samples_per_task: -1  # Use all available
  val_split_ratio: 0.1
  task_epochs: 1
  task_max_steps: null
```

### Step 2: Update Training Config (if needed)

```yaml
training_config:
  output_dir: "./out/riemannian_sequential"
  num_train_epochs: 1  # Per task
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  learning_rate: 0.0002
  
  # Important for sequential training
  save_strategy: "steps"
  save_steps: 256
  eval_strategy: "steps"
  eval_steps: 16
```

### Step 3: Run Training

```bash
bash train.sh
```

## Troubleshooting

### Issue: "Task X not found"

**Solution**: Verify task name in `sequential_tasks_config.tasks`. Available tasks are: SST2, Copa, BoolQ, MultiRC, CB, WIC, WSC, ReCoRD, RTE, SQuAD, DROP.

### Issue: CUDA out of memory

**Solutions**:
1. Reduce `samples_per_task` in config
2. Reduce `per_device_train_batch_size` in training_config
3. Use `gradient_accumulation_steps` to simulate larger batch size
4.  Use single GPU instead of multi-GPU: `gpu: "gpu_0"`

### Issue: Tasks loading very slowly

**Solution**: Reduce `samples_per_task` to fewer samples for testing:
```yaml
sequential_tasks_config:
  samples_per_task: 100  # Test with 100 samples per task
```

### Issue: Different validation losses per task make comparison hard

**Solution**: Use fixed seed and ensure consistent val split:
```yaml
dataset_config:
  val_split_ratio: 0.1  # Consistent across tasks
  
training_config:
  seed: 0
  data_seed: 0
```

## Differences: Single Dataset vs Sequential Tasks

| Aspect | Single Dataset | Sequential Tasks |
|--------|----------------|------------------|
| **Enabled** | `loading_strategy: "single"` | `sequential_tasks_config.enabled: true` |
| **Dataset** | One combined dataset | Different task datasets |
| **Task Count** | `multitask_config.num_tasks` | Auto-detected from tasks list |
| **Per-Task Data** | Same dataset every task | Different dataset per task |
| **Use Case** | Multi-task on same data | Progressive learning/transfer |
| **Riemannian?** | Yes (cumulative learning) | Yes (cumulative + task shift) |

## Advanced Configuration

### Using Subset of Samples

```yaml
sequential_tasks_config:
  enabled: true
  tasks:
    - "SST2"
    - "RTE"
  samples_per_task: 500  # Use only 500 samples per task
```

Calculate expected training time:
```
Samples: 500 * 2 tasks = 1000 total
Batch size: 8
Gradient accumulation: 4
Effective batch: 32
Steps: 1000 / 32 = ~31 steps per task
Time: ~1-2 minutes per task on V100
```

### Custom Task Order

Tasks with recommended progression (easy → hard):

```yaml
tasks:
  - "SST2"      # Easiest: Binary classification
  - "RTE"       # Easy-Medium: Binary entailment
  - "BoolQ"     # Medium: Reading comprehension
  - "MultiRC"   # Medium-Hard: Multi-choice RC
  - "DROP"      # Hardest: Discrete reasoning
```

## Files Involved

1. **config.yaml** - Main configuration (updated)
2. **colm/train/config_parser.py** - Config loader (updated with sequential_tasks_config)
3. **colm/train/train_sequential_from_config.py** - Training script (updated)
4. **data/sequential_task_loader.py** - NEW: Task dataset loader
5. **data/tasks.py** - Task definitions (unchanged)

## Next Steps

1. ✅ Update `config.yaml` with sequential tasks
2. ✅ Select tasks from available list
3. ✅ Adjust `samples_per_task` if needed
4. ✅ Run: `bash train.sh`
5. ✅ Monitor WandB dashboard
6. ✅ Compare metrics across tasks
7. ✅ Fine-tune: Adjust samples/tasks/hyperparams

## Key Concepts

**Riemannian Sequential Training**:
- Model follows a curved trajectory (Riemannian manifold) through task parameter spaces
- Each task's training pushes model along task-specific gradient direction
- Cumulative effect: final model balances all task-specific directions

**Cumulative Learning**:
- Task 0: Learn task 0 patterns
- Task 1: Learn task 1 patterns + retain task 0 knowledge
- Task 2: Learn task 2 patterns + retain tasks 0,1 knowledge

**Model Persistence**:
- Single model instance reused across all tasks
- Weights updated in-place (no reinitialization)
- Clear trajectory of learning

## Support and Debugging

For detailed logs:
```bash
# Run with debug logging
bash train.sh 2>&1 | tee training_debug.log
```

Check logs for:
- Dataset loading summary (task sizes)
- Per-task training progress
- Gradient norms (should stay stable)
- GPU memory usage per task
