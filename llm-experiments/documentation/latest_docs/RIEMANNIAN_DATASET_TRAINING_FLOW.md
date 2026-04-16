# Riemannian Method: Dataset Loading, Processing, and Training Flow

## Overview

This document explains how the original Riemannian training method handles dataset loading, preprocessing, model training, and evaluation for sequential multi-task learning. The Riemannian approach uses a specialized gradient-based data selection method combined with zeroth-order optimization.

## 1. Dataset Loading Architecture

### 1.1 Sequential Task Loader
The `SequentialTaskLoader` class (in `colm/data/sequential_task_loader.py`) manages dataset loading for multiple tasks:

```python
class SequentialTaskLoader:
    def __init__(self, config):
        self.tasks = config.get('tasks', [])  # e.g., ['SST2', 'RTE', 'BoolQ']
        self.samples_per_task = config.get('samples_per_task', -1)  # -1 = all samples
        self.val_split_ratio = config.get('val_split_ratio', 0.1)  # 10% for validation
```

**Key Features:**
- Loads tasks sequentially (one at a time)
- Each task is independent with separate train/val splits
- Supports variable sample size limiting per task
- Validation data is automatically split from training data

### 1.2 Task-Specific Dataset Loading

```python
def load_task(self, task_id: int) -> Tuple[TaskDataset, TaskDataset]:
    task_name = self.tasks[task_id]  # Get task name
    task = get_task(task_name)  # Load task module
    
    train_samples = task.samples.get("train", [])
    val_samples = task.samples.get("valid", [])
    
    # Optional: limit samples
    if self.samples_per_task > 0:
        train_samples = train_samples[:self.samples_per_task]
```

**Data Flow:**
1. Task name from config (e.g., 'SST2')
2. Load task-specific dataset module
3. Extract train/validation samples
4. Apply sample limiting if configured
5. Return `TaskDataset` wrapper objects

## 2. Dataset Preprocessing Pipeline

### 2.1 Tokenization

The Riemannian code applies causal language modeling (CLM) tokenization:

```python
def tokenize_function(examples):
    # Tokenize text with max_length 512
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,
        padding='max_length'
    )
    
    # For CLM: labels = input_ids (predict next token)
    result['labels'] = result['input_ids'].copy()
    
    # Mask padding tokens (-100 means ignore in loss)
    result['labels'] = [
        [-100 if token == tokenizer.pad_token_id else token 
         for token in labels]
        for labels in result['labels']
    ]
    
    return result
```

**Key Points:**
- **Max length**: Fixed at 512 tokens
- **Padding**: 'max_length' ensures consistent batch shapes
- **Labels format**: Same as input_ids for causal LM
- **Padding mask**: -100 value tells loss function to ignore padding

### 2.2 Data Collation

The collator combines tokenized samples into batches:

```python
# Riemannian uses default_data_collator for causal LM format
data_collator = default_data_collator  # Simple stack + pad operation
```

Unlike sequence-to-sequence models, causal LM doesn't need special label handling.

## 3. Sequential Training Architecture

### 3.1 Model State Persistence

The key insight of sequential training is **weight persistence**:

```python
for task_id in range(num_tasks):
    # Load task-specific dataset
    train_subset, val_subset = task_loader.load_task(task_id)
    
    # Same model object used for all tasks!
    # Weights from task_0 → task_1 → task_2
    trainer = Trainer(model=model, ...)
    
    # Train on this task
    train_result = trainer.train()
    
    # Model weights automatically updated in-place
    # Ready for next task without reloading
```

**Weight Flow:**
- Task 0: Random → Task 0 trained weights
- Task 1: Task 0 weights → Task 1 trained weights
- Task 2: Task 1 weights → Task 2 trained weights

### 3.2 Training Configuration

The `TrainingArguments` setup for each task:

```python
training_args = TrainingArguments(
    output_dir='./checkpoints',
    num_train_epochs=1,  # or max_steps
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch = 64
    
    learning_rate=2e-4,
    optim='adamw_torch',  # Standard AdamW optimizer
    
    logging_steps=10,
    eval_steps=16,  # Evaluate frequently
    save_steps=256,
    
    bf16=True,  # bfloat16 precision
    gradient_checkpointing=False,  # Disabled for LoRA
)
```

### 3.3 LoRA Configuration

Applied once, before sequential training:

```python
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=512,
    lora_dropout=0.05,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'up_proj', 'down_proj'],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
# Now 32.5M trainable LoRA parameters out of 8B total
```

## 4. Training Loop for Each Task

### 4.1 Per-Task Training Process

```python
for task_id in range(num_tasks):
    logger.info(f"Training Task {task_id}: {task_name}")
    logger.info(f"Already trained on {task_id} previous tasks")
    
    # 1. Load task dataset
    train_subset, val_subset = task_loader.load_task(task_id)
    
    # 2. Tokenize on-the-fly
    train_subset = train_subset.map(tokenize_function, batched=True)
    val_subset = val_subset.map(tokenize_function, batched=True)
    
    # 3. Create trainer with same model object
    trainer = Trainer(
        model=model,  # Carries weights from previous tasks
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=val_subset,
        data_collator=default_data_collator,
        callbacks=[MonitoringCallback(task_id)],
    )
    
    # 4. Train
    train_result = trainer.train()
    
    # 5. Evaluate
    eval_results = trainer.evaluate()
    
    # Metrics logged for this task
    logger.info(f"Task {task_id} - Train loss: {train_result.training_loss:.4f}")
    logger.info(f"Task {task_id} - Eval loss: {eval_results['eval_loss']:.4f}")
```

### 4.2 Checkpoint Management

- **Save strategy**: Every 256 steps
- **Keep limit**: 3 most recent checkpoints
- **Checkpoint naming**: `checkpoint-{step}`
- **Content**: Model weights + LoRA adapters + training state

## 5. Evaluation Process

### 5.1 Per-Task Evaluation

Evaluation happens **during** training (every 16 steps):

```python
# During trainer.train()
eval_results = trainer.evaluate()

# Returns:
{
    'eval_loss': 0.1234,
    'eval_runtime': 95.23,  # seconds
    'eval_samples_per_second': 9.18,
    'eval_steps_per_second': 1.15,
    'epoch': 0.01,
}
```

### 5.2 Metrics Tracked

For each evaluation:
- **Loss**: Cross-entropy loss on validation set
- **Perplexity**: exp(loss) - measures how well model predicts next token
- **Learning rate**: Current LR (with warmup schedule)
- **Gradient norm**: L2 norm of gradients for stability monitoring

### 5.3 WandB Logging

All metrics logged in real-time:

```
Step 10:  loss=3.73, perplexity=41.6, eval_loss=0.192
Step 20:  loss=0.12, perplexity=1.13,  eval_loss=0.151
Step 30:  loss=0.09, perplexity=1.10,  eval_loss=0.149
...
```

## 6. Task Transition Mechanism

### 6.1 Checkpoint-Based Transition

After completing Task 0:

```python
# Model has Task 0 weights embedded in LoRA adapters
# Checkpoint saved with all weights

# Before Task 1:
# - Same model object is reused
# - LoRA adapters still have Task 0 weights
# - Optimizer state is reset (new Training session)
# - Learning rate warmup restarts
```

### 6.2 Weight Initialization for Next Task

```python
# Task 1 starts with:
# - Base model: Llama-3.1-8B (frozen)
# - LoRA layers: Weights from Task 0
# - This is "transfer learning" - knowledge transfer via initialization

# Gradient flow during Task 1:
# dL/d(LoRA_weight) computed from Task 1 labels
# LoRA adapters updated based on Task 1 + initialized from Task 0
```

## 7. Data Selection in Riemannian (ZerO-Order Optimization)

### 7.1 Gradient-Based Subset Selection

The Riemannian method uses zeroth-order optimization:

```python
# For each batch:
# 1. Compute loss on full batch
# 2. Use gradient to select important samples
# 3. Train on selected subset for parameter updates

# Not used in our simplified version - we use full batches
```

### 7.2 Configuration Parameters

```yaml
small_batch_ratio: 1.0  # Use full batch (1.0 = 100%)
data_selection_method: 'random'  # Or 'importance', 'gradient_based'
data_selection_unit: 'mezo'  # Mean Zero-Order optimization
```

## 8. Complete Training Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ Initialize: Load Model + Apply LoRA                    │
│ (weights: random, LoRA params: random)                 │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │  Task 0 (SST2)          │
        │ ┌──────────────────────┐│
        │ │ Load 67K samples     ││
        │ │ Tokenize → Batch     ││
        │ │ Train 1 epoch        ││
        │ │ Eval every 16 steps  ││
        │ │ Save checkpoints     ││
        │ └──────────┬───────────┘│
        │            │ weights    │
        │  LoRA params from Task0 │
        └────────────┼────────────┘
                     │
        ┌────────────▼────────────┐
        │  Task 1 (RTE)           │
        │ ┌──────────────────────┐│
        │ │ Load 2.5K samples    ││
        │ │ Tokenize → Batch     ││
        │ │ Train 1 epoch        ││
        │ │ Eval every 16 steps  ││
        │ │ Save checkpoints     ││
        │ └──────────┬───────────┘│
        │            │ weights    │
        │  LoRA params from Task1 │
        └────────────┼────────────┘
                     │
        ┌────────────▼────────────┐
        │  Task 2 (BoolQ)         │
        │ ┌──────────────────────┐│
        │ │ Load 9.4K samples    ││
        │ │ Tokenize → Batch     ││
        │ │ Train 1 epoch        ││
        │ │ Eval every 16 steps  ││
        │ │ Save checkpoints     ││
        │ └──────────┬───────────┘│
        │            │ weights    │
        │  LoRA params from Task2 │
        └────────────┼────────────┘
                     │
        ┌────────────▼────────────┐
        │ Final Model State       │
        │ (weights from all tasks)│
        │ Ready for inference     │
        └────────────────────────┘
```

## 9. Key Implementation Details

### 9.1 Tokenization Process
- Real-time during training (no pre-tokenization)
- Consistent 512 token max length
- Causal LM format with labels = input_ids

### 9.2 Batch Formation
- Per-device batch size: 8
- Gradient accumulation: 4 steps
- Effective batch: 32 samples

### 9.3 Evaluation Cadence
- Frequency: Every 16 steps (roughly every 2 minutes)
- Samples used: Full validation set
- Time per eval: ~95 seconds for 872 val samples

### 9.4 Precision & Memory
- Mixed precision: bfloat16
- Gradient checkpointing: Disabled (incompatible with LoRA)
- Memory usage: ~850GB across 8 GPUs (for model + batch)

## 10. Summary

The Riemannian method achieves sequential multi-task learning by:

1. **Persisting model weights** across task boundaries
2. **Using LoRA adapters** to keep base model frozen
3. **Causal LM training** where models predict next tokens
4. **Real-time evaluation** to track learning on each task
5. **Checkpoint-based transitions** ensuring no weight loss between tasks

This approach allows a single model to learn from multiple tasks sequentially while preventing catastrophic forgetting through LoRA's adapter-based approach.
