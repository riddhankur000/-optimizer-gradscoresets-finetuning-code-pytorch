# CoLM Training Codebase Exploration

**Repository**: `/data/riddhankur/PROJECTS/GREATS_COLM_pytorch/local/llm-experiments`

---

## 1. MAIN TRAINING ENTRY POINTS

### 1.1 Single-Task Training: `colm/train/train.py`

**Main Function**: `main()` (lines 1-400+)

**Flow**:
```
1. Parse arguments (ModelArguments, DataArguments, TrainingArguments)
2. Load tokenizer and model
3. Apply LoRA config
4. Load training dataset (SuperGLUE or custom)
5. Create trainer (SubsetTrainer or HuggingFace Trainer)
6. Execute trainer.train()
7. Save model and metrics
```

**Key Code Sections**:
- **Lines 1-60**: Imports and setup
- **Lines 60-100**: Argument parsing and logging setup
- **Lines 100-150**: Model and tokenizer loading, LoRA configuration
- **Lines 150-300**: Dataset loading (SuperGLUE or generic)
- **Lines 300-350**: Data collator setup, trainer class selection
- **Lines 342-347**: **WANDB SETUP**:
  ```python
  os.environ["WANDB_ENTITY"] = training_args.wandb_entity
  os.environ["WANDB_PROJECT"] = training_args.wandb_project
  os.environ["WANDB_NAME"] = training_args.run_name + f'_{os.uname()[1]}'
  os.environ["WANDB_NOTES"] = training_args.wandb_notes
  ```
- **Lines 350-380**: Trainer initialization and training
- **Lines 380-400**: Model saving and metrics logging

```python
trainer = trainer_class(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=analysis_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
train_result = trainer.train(resume_from_checkpoint=model_args.checkpoint_path)
metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
```

### 1.2 Multi-Task Training: `colm/train/train_multitask.py`

**Main Function**: `main()` (lines 400+)

**Unique Features**:
- YAML config file support (config_loader.py)
- MonitoringCallback for enhanced metrics
- Per-task evaluation metrics
- GPU/CPU resource monitoring
- Gradient norm tracking

**Flow**:
```
1. Load config from YAML or CLI arguments
2. Set GPU configuration (TF32, device_ids)
3. Load tokenizer and combined dataset
4. Identify tasks from dataset
5. Tokenize dataset with per-task tracking
6. Create MultiTaskTrainer with optimizer factory
7. Add MonitoringCallback
8. Execute training
```

**Key Classes**:

```python
class MonitoringCallback(TrainerCallback):
    """Enhances logging with gradient norms, GPU/CPU stats, perplexity"""
    
    def on_backward_end(self, args, state, control, **kwargs):
        # Compute grad norm before they're cleared
        self.last_grad_norm = self._get_grad_norm(model)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Add perplexity, grad_norm, GPU stats, system stats
        logs['train_perplexity'] = math.exp(logs['loss'])
        logs.update(self._get_grad_norm(model))
        logs.update(self._get_gpu_stats())
        logs.update(self._get_system_stats())
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Log eval metrics with perplexity and resource stats
        metrics['perplexity'] = math.exp(metrics['loss'])


class MultiTaskTrainer(Trainer):
    """Custom trainer with optimizer factory support"""
    
    def create_optimizer(self):
        # Uses optimizer factory from config or defaults to AdamW
        if self.config:
            self.optimizer = create_optimizer_from_config(...)
        else:
            super().create_optimizer()
```

---

## 2. CORE TRAINING ENGINE: `subset_trainer_distributed.py`

**Class**: `SubsetTrainer(Trainer)` - Extends HuggingFace Trainer

### 2.1 Data Selection Flow

The key innovation is **batch-wise selection**: 
- Accumulate B small batches → Collect representations
- Select K < B samples via algorithm → Train selected K

**Main Loop**: `_inner_training_loop()` (lines 250-1100+)

```
For each epoch:
  For each batch in dataloader:
    
    [Accumulation Phase]
    if (batch_counter % gradient_accumulation_steps != 0):
      rep = save_select(model, inputs)          # Collect representation
      total_reps.append(rep)
      continue
    
    [Selection Phase]
    rep = save_select(model, inputs)            # Add final representation
    filter_and_normalize(total_reps)            # Process all B reps
    dist.gather(all_reps to rank 0)            # Sync across GPUs
    
    if rank == 0:
      selected_idx, weights = select_data(all_reps, max_K)
    
    dist.broadcast(selected_idx, selected_weights)
    
    [Training Phase]
    for inner_step, selected_input in enumerate(selected_inputs):
      loss = training_step(model, selected_input, weights[inner_step])
      tr_loss += loss
    
    optimizer.step()
    lr_scheduler.step()
```

**Code Locations**:
- Accumulation: lines 630-640
- Representation collection: line 640-650
- Filtering & normalization: lines 650-700
- Distributed gathering: lines 700-760
- Selection call: lines 800-850
- Training loop: lines 900-950

### 2.2 Representation Extraction: `save_select()` (lines 1200-1320)

Computes feature embeddings based on selection unit:

```python
def save_select(self, model, inputs):
    if self.args.data_selection_unit == "rep":
        # Last token hidden state
        hidden_states = model(input_ids, output_hidden_states=True).hidden_states
        res = hidden_states[-1][ids, pos]  # Shape: [batch, hidden_dim]
    
    elif self.args.data_selection_unit == "mezo":
        # Zeroth-order gradient approximation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        projected_grad = (loss1 - loss2) / (2 * eps)
        res = projected_grad * random_z  # Shape: [total_params]
    
    elif self.args.data_selection_unit == "masked_grad":
        # Use backprop on last N layers
        loss = self.training_step(model, inputs)
        res = torch.cat([param.grad.flatten() for param in target_params])
    
    elif self.args.data_selection_unit == "completion_length":
        # Sequence length weighting
        res = inputs["completion_lengths"][0]
    
    return res
```

### 2.3 Selection Algorithms: `select_data()` (lines 1358-1420)

**Supported methods** (set via `--data_selection_method`):

```python
def select_data(self, inputs, max_samples=64, source_list=None, model=None):
    
    if self.method == "greats":
        # Gradient-based coreset selection
        _, sims = compute_cost_matrix(inputs, inputs, metric="cosine")
        _, sims_cross = compute_cost_matrix(inputs, eval_inputs)
        idx = greats.greedy_selection(sims_cross.mean(1), sims, max_samples)
        return idx, weights
    
    if self.method == "fairot":
        # Fair outlier truncation
        dist, sims = compute_cost_matrix(inputs, inputs, metric="cosine")
        idx = fairot2.greedy_fairot(sims, max_samples, dist=dist, iters=500, reg=1e-1)
        return idx, weights
    
    if self.method == "fairot_multisource":
        # Fair outlier truncation with source balancing
        idx, weights = select_data_facloc(inputs, max_samples, source_list, 
                                         optim=lambda S,k: fairot2.greedy_fairot(...))
        return idx, weights
    
    if self.method == "submodlib" or "weightedsubmodlib":
        # Facility location-based selection
        greedy_indices = get_orders_and_weights(max_samples, inputs,
                                               metric=self.args.facility_similarity,
                                               strategy=self.args.source_wise_selection)
        return greedy_indices[0], greedy_indices[1]
    
    if self.method == "spot":
        # SPOT greedy subset selection
        dist = compute_cost_matrix(inputs, inputs, metric="cosine")
        idx = SPOT_GreedySubsetSelection(dist, target_marginal, max_samples)
        return idx, weights
    
    if self.method == "none":
        # No selection, use standard HF Trainer
        # (handled by trainer_class selection in train.py)
```

**Key Parameters**:
- `data_selection_method`: ['submodlib', 'weightedsubmodlib', 'greats', 'fairot', 'fairot_multisource', 'spot', 'none']
- `data_selection_unit`: ['rep', 'mezo', 'masked_grad', 'grad', 'completion_length', 'length_loss_weighted']
- `small_batch_ratio`: K/B ratio (e.g., 0.1 = select 10% of batch)
- `facility_similarity`: ['cosine', 'euclidean', 'l1']
- `mezo_eps`: Perturbation scale
- `mezo_optim`: ['sgd', 'adam']

---

## 3. TRAINING ARGUMENTS & CONFIGURATION

### File: `colm/train/training_arguments.py`

**Extends**: `transformers.TrainingArguments`

**Key Custom Arguments**:

```python
@dataclass
class TrainingArguments(TA):
    # Data Selection
    data_selection_method: str = "none"                    # Selection algorithm
    data_selection_unit: str = "mezo"                      # How to compute features
    small_batch_ratio: float = 1.0                         # K/B ratio
    
    # MeZO (Zeroth-Order) Settings
    mezo_eps: float = 1e-3                                # Perturbation scale
    mezo_optim: str = "sgd"                               # ['sgd', 'adam']
    mezo_selection: str = "mezo_selection"                # Weight×grad weighting
    mezo_topk: str = "largest"                            # Top-k selection
    mezo_transform: str = "none"                          # Normalization type
    
    # Facility Location
    facility_similarity: str = "l1"                       # Distance metric
    source_wise_selection: str = "proportional"           # Source handling
    
    # Representation Dimension
    rep_dim: int = 2560                                   # Rep feature dim
    zo_dim: int = 2560                                    # MeZO feature dim
    proj_dim: int = 2560                                  # Last layer proj dim
    
    # LoRA Settings
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    last_layers: List[str] = ["q_proj", "v_proj"]         # Which layers to select on
    
    # WandB Integration
    wandb_entity: str = ""
    wandb_project: str = ""
    wandb_notes: str = ""
    
    # Analysis
    analysis_mode: bool = True
    keep_sources: str = "0_1_3_5_7_8_9_10_11_13"           # Always keep these sources
```

### File: `colm/train/data_arguments.py`

```python
@dataclass
class DataArguments:
    train_files: List[str]                               # Input data paths
    subset_index_files: List[str]                        # Pre-computed subset indices
    max_seq_length: Optional[int]                        # Tokenization length
    percentage: float = 1.0                              # Sampling percentage
    sample_data_seed: int = 42
```

### File: `colm/train/model_arguments.py`

```python
@dataclass
class ModelArguments:
    model_name_or_path: str                              # Model HF ID or path
    model_max_length: int = 2048
    enable_dropout: bool = True
    lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = ["q_proj", "v_proj"]
```

---

## 4. DATA PIPELINE

### File: `colm/data/get_training_dataset.py`

**Main Function**: `get_training_dataset()`

```python
def get_training_dataset(
    train_files,
    tokenizer,
    max_seq_length,
    sample_percentage=0.9,
    subset_index_files=None,
    seed=42
):
    """
    Load datasets from files, optionally subset, and tokenize.
    
    Returns:
      - SupervisedDataset (custom) or HFDataset
      - Tracks: input_ids, labels, attention_mask, indices
    """
```

**Supported Dataset Types**:
1. **SuperGLUE**: Tasks handled via `colm/data/tasks.py`
   - RTE, COLA, SST-2, MRPC, QQP, MNLI, QNLI, WNLI
   - WNLI, BoolQ, MultiRC, ReCoRD

2. **Custom HF Datasets**: JSON/Parquet format
   - Expects: text, or instruction + input + output
   - Tokenizes with task-specific labels

**Data Collators**:
- `DataCollatorForSupervisedDataset`: Standard LM data collation
- `DataCollatorForSupervisedDatasetWithSource`: Tracks source index
- `DataCollatorWithPaddingAndNesting`: For SuperGLUE classification
- `NondiffCollator`: Non-differentiable loss handling

---

## 5. WANDB INTEGRATION & LOGGING

### 5.1 WandB Setup in `train.py` (lines 342-347)

```python
# Setup wandb
os.environ["WANDB_ENTITY"] = training_args.wandb_entity
os.environ["WANDB_PROJECT"] = training_args.wandb_project
os.environ["WANDB_NAME"] = training_args.run_name + f'_{os.uname()[1]}'
os.environ["WANDB_NOTES"] = training_args.wandb_notes
logger.info('Finished wandb setup.')
```

**Environment Variables**:
- `WANDB_ENTITY`: Your WandB team/user
- `WANDB_PROJECT`: Project name
- `WANDB_NAME`: Run name (appended with hostname)
- `WANDB_NOTES`: Notes/tags for run

**Automatic Logging**:
- HF Trainer logs: loss, learning_rate, epoch, batch_size, gradient_norm
- Custom logging in `train.py`:
  ```python
  trainer.log_metrics("train", metrics)
  trainer.save_metrics("train", metrics)
  ```

### 5.2 Enhanced Logging in `train_multitask.py`

**MonitoringCallback** adds:
- `train_perplexity`: exp(train_loss)
- `eval_perplexity`: exp(eval_loss)
- `grad_norm`: L2 norm of all gradients
- `grad_norm_avg`: Average gradient norm per parameter
- `gpu_memory_used_gb`: GPU memory consumed
- `gpu_memory_utilization_%`: GPU memory %
- `gpu_load_%`: GPU compute load
- `cpu_percent`: CPU utilization
- `cpu_memory_percent`: System memory %

**Example WandB Dashboard Metrics**:
```
Training:
  loss (scalar)
  train_perplexity (scalar)
  learning_rate (scalar)
  grad_norm (scalar)
  gpu_memory_used_gb (scalar)
  cpu_percent (scalar)

Evaluation:
  eval_loss (scalar)
  eval_perplexity (scalar)
  eval_grad_norm (scalar)
  eval_gpu_memory_used_gb (scalar)
```

---

## 6. TRAINER CLASS SELECTION LOGIC

**In `train.py` (lines 340-350)**:

```python
# Trainer class selection based on data_selection_method
if training_args.data_selection_method == "none":
    logger.info("Using HuggingFace Trainer")
    trainer_class = Trainer  # Standard HF Trainer, no selection
    
elif training_args.efficient_mezo:
    logger.info("Using SubsetTrainerEfficient")
    trainer_class = SubsetTrainerEfficient  # Optimized for MeZO
    
else:
    logger.info("Using SubsetTrainer")
    trainer_class = SubsetTrainer  # Full selection pipeline
```

**How Selection is Toggled**:
- `data_selection_method="none"` → Use standard HF Trainer
- `data_selection_method="greats"` → Use SubsetTrainer with greats algorithm
- `data_selection_method="fairot"` → Use SubsetTrainer with fairot algorithm
- etc.

---

## 7. DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│ train.py / train_multitask.py                                   │
│ Main entry point: parse args, load model, tokenizer             │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
    ┌────▼─────────────┐         ┌──────▼──────────┐
    │ get_training_     │         │ training_       │
    │ dataset()         │         │ arguments.py    │
    │ Load & tokenize   │         │ Config args     │
    └────┬─────────────┘         └────────────────┘
         │                              │
         │     ┌───────────────────────┘
         │     │
    ┌────▼─────▼───────────────────────────┐
    │ SubsetTrainer._inner_training_loop()  │
    │ (or Trainer if no selection)         │
    └────┬──────────────────────────────────┘
         │
    FOR each epoch:
         │  
    FOR each batch:
         │
    ┌────▼───────────────────────────────┐
    │ Accumulation Phase (B batches)      │
    │ - save_select() collects reps       │
    │ - Accumulate: total_reps.append(rep)│
    └────┬───────────────────────────────┘
         │ (when buffer full)
    ┌────▼──────────────────────────────────┐
    │ Selection Phase                       │
    │ - Filter & normalize reps             │
    │ - dist.gather() to rank 0             │
    │ - select_data() picks K < B samples   │
    │ - dist.broadcast() selected_idx       │
    └────┬──────────────────────────────────┘
         │
    ┌────▼──────────────────────────────────┐
    │ Training Phase                        │
    │ - training_step() on selected samples │
    │ - optimizer.step()                    │
    │ - Update metrics                      │
    └────┬──────────────────────────────────┘
         │
    ┌────▼──────────────────────────────────┐
    │ Evaluation & Logging (periodic)       │
    │ - trainer.evaluate()                  │
    │ - trainer.log_metrics()               │
    │ - WandB update                        │
    └────────────────────────────────────────┘
```

---

## 8. KEY FUNCTIONS & SIGNATURES

### Critical For Sequential Training

**Training Loop Entry**:
```python
def _inner_training_loop(
    self, 
    batch_size=None, 
    args=None, 
    resume_from_checkpoint=None, 
    trial=None, 
    ignore_keys_for_eval=None
) -> TrainOutput:
    """
    Core training loop with data selection.
    Accumulates B batches, selects K, trains K.
    """
```

**Representation Extraction** (inject sequential task logic here):
```python
def save_select(self, model, inputs):
    """
    Extract representation for selection.
    - Inputs: batch of data
    - Output: feature vector or scalar for selection
    - Currently: hidden states, MeZO gradient, masked gradients, etc.
    
    TO ADD EVAL LOSS TRACKING:
    Insert eval loss computation here on validation set
    """
```

**Data Selection** (inject task-specific selection here):
```python
def select_data(self, inputs, max_samples=64, source_list=None, model=None):
    """
    Select K samples from B based on representations.
    - Inputs: all_reps [B, hidden_dim]
    - Output: selected indices [K], weights [K]
    
    TO ADD SEQUENTIAL TASKS:
    Modify selection to balance tasks
    Use source_list to identify task boundaries
    """
```

**Single Training Step** (already tracks loss):
```python
def training_step(self, model, inputs, inputs_weight=1.0) -> torch.Tensor:
    """
    One training step with loss calculation.
    Loss is already computed here, returned as scalar.
    """
```

---

## 9. INJECTION POINTS FOR SEQUENTIAL TASK TRAINING

### Point 1: Modify `select_data()` for Task Balancing

**Location**: `subset_trainer_distributed.py`, line 1358

**Current**: Selects K samples greedily across all data

**Option A - Task-Proportional Selection**:
```python
def select_data(self, inputs, max_samples=64, source_list=None, model=None):
    # source_list now contains task IDs instead of just source IDs
    task_ids = source_list  # [0,0,1,1,2,2,...] for task IDs
    
    if self.args.sequential_tasks:
        # Get current task from training state
        current_task = self.state.global_step // self.args.steps_per_task
        
        # Select only from current task
        task_mask = (task_ids == current_task)
        task_inputs = inputs[task_mask]
        
        # Apply normal selection on task subset
        idx, weights = self.select_data_facloc(task_inputs, max_samples, ...)
        return idx[task_mask], weights
```

### Point 2: Track Per-Task Eval Loss

**Location**: `train_multitask.py`, MonitoringCallback

**Add**:
```python
def on_evaluate(self, args, state, control, metrics=None, **kwargs):
    if self.trainer.eval_dataset and 'task' in self.trainer.eval_dataset.column_names:
        # Compute per-task eval loss
        tasks = set(self.trainer.eval_dataset['task'])
        for task in tasks:
            task_indices = [i for i, t in enumerate(self.trainer.eval_dataset['task']) if t == task]
            task_eval_dataset = self.trainer.eval_dataset.select(task_indices)
            task_metrics = self.trainer.evaluate(eval_dataset=task_eval_dataset)
            metrics[f'eval_loss_{task}'] = task_metrics['eval_loss']
            metrics[f'eval_perplexity_{task}'] = math.exp(task_metrics['eval_loss'])
```

### Point 3: Sequential Training Loop

**Location**: `train_multitask.py` main()

**Wrapper**:
```python
def main():
    # ... existing setup ...
    
    if training_args.sequential_training:
        tasks = sorted(set(train_dataset['task']))
        for task_idx, task in enumerate(tasks):
            logger.info(f"Training on task {task_idx+1}/{len(tasks)}: {task}")
            
            # Filter to current task
            task_indices = [i for i, t in enumerate(train_dataset['task']) if t == task]
            task_dataset = train_dataset.select(task_indices)
            
            # Update trainer dataset
            trainer.train_dataset = task_dataset
            
            # Train on this task
            task_result = trainer.train(resume_from_checkpoint=...)
            
            # Log task-specific metrics
            logger.info(f"Task {task}: loss={task_result.metrics.get('train_loss')}")
    else:
        # Normal multi-task training (mixed)
        train_result = trainer.train(...)
```

---

## 10. SUMMARY TABLE

| Aspect | Details |
|--------|---------|
| **Entry Points** | `train.py` (single), `train_multitask.py` (multi) |
| **Core Trainer** | `SubsetTrainer(Trainer)` in subset_trainer_distributed.py |
| **Selection Phase** | Lines 600-900, `select_data()` method |
| **Data Selection Methods** | greats, fairot, spot, submodlib, none |
| **Selection Units** | rep, mezo, masked_grad, completion_length, length_loss_weighted |
| **Key Tuning Parameters** | data_selection_method, data_selection_unit, small_batch_ratio, mezo_eps |
| **WandB Integration** | Environment vars (WANDB_ENTITY, WANDB_PROJECT, WANDB_NAME, WANDB_NOTES) |
| **Enhanced Metrics** | MonitoringCallback adds perplexity, grad_norm, GPU stats |
| **Data Flow** | Load → Accumulate B → Select K → Train K → Eval |
| **Distributed Training** | DDP via dist.gather/broadcast for selection sync |
| **Sequential Task Support** | Partial (via source_list), needs extension for per-task tracking |
| **Eval Loss Tracking** | Standard trainer.evaluate(), enhanced in MonitoringCallback |

