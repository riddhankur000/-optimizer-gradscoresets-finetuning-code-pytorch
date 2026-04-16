# CoLM Sequential Training Implementation Guide

## QUICK REFERENCE

### Files You'll Modify

| File | What to Change | Why |
|------|---|---|
| `colm/train/subset_trainer_distributed.py` | `select_data()` method | For task-specific selection |
| `colm/train/train_multitask.py` | `MonitoringCallback.on_evaluate()` | For per-task eval loss logging |
| `colm/train/training_arguments.py` | Add custom args | For sequential training flags |
| `colm/train/train_multitask.py` | `main()` function | For sequential training loop |

### Files to REFERENCE (don't modify unless needed)

| File | Purpose | Key Functions |
|------|---------|---|
| `colm/train/subset_trainer_distributed.py` | Core training engine | `_inner_training_loop()`, `save_select()`, `select_data()` |
| `colm/train/training_arguments.py` | Config struct | Defines all CLI args |
| `colm/data/get_training_dataset.py` | Data loading | `get_training_dataset()` |

---

## WHERE TO INJECT: DETAILED CODE LOCATIONS

### 1. EVAL LOSS TRACKING

**File**: `colm/train/train_multitask.py` **Line**: ~170-180

**Current Code**:
```python
def on_evaluate(self, args, state, control, metrics=None, **kwargs):
    """
    Called after evaluation is completed.
    Ensures eval metrics are properly logged to wandb.
    """
    if metrics is None or not self.trainer:
        return
    
    model = self.trainer.model
    
    # Add perplexity calculation
    if 'loss' in metrics and metrics['loss'] is not None:
        try:
            metrics['perplexity'] = math.exp(metrics['loss'])
        except:
            pass
    
    # ... rest of method
```

**To Add Per-Task Eval Loss** (after line 170, before existing code):
```python
def on_evaluate(self, args, state, control, metrics=None, **kwargs):
    if metrics is None or not self.trainer:
        return
    
    model = self.trainer.model
    
    # ============ ADD THIS BLOCK ============
    # Per-task evaluation loss tracking
    eval_dataset = self.trainer.eval_dataset
    if eval_dataset and 'task' in eval_dataset.column_names:
        logger.info("📊 Computing per-task evaluation metrics...")
        
        # Get unique tasks
        tasks = sorted(set(eval_dataset['task']))
        
        for task in tasks:
            # Filter to this task
            task_indices = [i for i, t in enumerate(eval_dataset['task']) 
                           if t == task]
            
            if len(task_indices) == 0:
                continue
            
            task_eval_dataset = eval_dataset.select(task_indices)
            
            # Evaluate on task subset
            try:
                task_metrics = self.trainer.evaluate(
                    eval_dataset=task_eval_dataset,
                    metric_key_prefix=f"eval_task_{task}"
                )
                
                # Extract loss and compute perplexity
                task_loss_key = f"eval_task_{task}_loss"
                if task_loss_key in task_metrics:
                    task_loss = task_metrics[task_loss_key]
                    metrics[task_loss_key] = task_loss
                    try:
                        metrics[f"eval_task_{task}_perplexity"] = math.exp(task_loss)
                    except:
                        pass
                    
                    logger.info(f"  Task '{task}': loss={task_loss:.4f}, "
                              f"samples={len(task_indices)}")
            except Exception as e:
                logger.warning(f"  Could not evaluate task {task}: {e}")
    # ======== END ADD BLOCK ========
    
    # Add perplexity calculation (existing code)
    if 'loss' in metrics and metrics['loss'] is not None:
        try:
            metrics['perplexity'] = math.exp(metrics['loss'])
        except:
            pass
    
    # ... rest of method continues
```

---

### 2. SEQUENTIAL TRAINING LOOP

**File**: `colm/train/train_multitask.py` **Line**: ~400-450 (in main())

**Current Code**:
```python
def main():
    # ... setup code ...
    
    # Create trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        task_names=tasks,
        config=config,
        callbacks=[monitoring_callback],
    )
    
    # Training
    logger.info("Starting training...")
    train_result = trainer.train(
        resume_from_checkpoint=checkpoint_path
    )
```

**To Add Sequential Training** (replace training section):
```python
def main():
    # ... setup code ...
    
    # Create trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        task_names=tasks,
        config=config,
        callbacks=[monitoring_callback],
    )
    
    # ============ ADD THIS BLOCK ============
    # Sequential vs Mixed Training
    if training_args.sequential_training:
        logger.info("🔄 SEQUENTIAL TRAINING MODE")
        logger.info(f"Training on {len(tasks)} tasks sequentially")
        
        all_results = {}
        
        for task_idx, task in enumerate(tasks):
            logger.info(f"\n{'='*60}")
            logger.info(f"Task {task_idx + 1}/{len(tasks)}: {task}")
            logger.info(f"{'='*60}")
            
            # Filter dataset to current task
            task_train_indices = [i for i, t in enumerate(train_dataset['task']) 
                                 if t == task]
            
            if len(task_train_indices) == 0:
                logger.warning(f"No training samples for task {task}, skipping")
                continue
            
            task_train_dataset = train_dataset.select(task_train_indices)
            
            # Optional: also filter eval set to this task
            task_eval_dataset = None
            if eval_dataset:
                task_eval_indices = [i for i, t in enumerate(eval_dataset['task']) 
                                    if t == task]
                if len(task_eval_indices) > 0:
                    task_eval_dataset = eval_dataset.select(task_eval_indices)
            
            # Update trainer with task-specific datasets
            trainer.train_dataset = task_train_dataset
            if task_eval_dataset:
                trainer.eval_dataset = task_eval_dataset
            
            logger.info(f"Training samples: {len(task_train_dataset)}")
            if task_eval_dataset:
                logger.info(f"Eval samples: {len(task_eval_dataset)}")
            
            # Train on this task
            task_result = trainer.train(
                resume_from_checkpoint=checkpoint_path if task_idx == 0 else None
            )
            
            # Store task-specific results
            all_results[task] = {
                'loss': task_result.metrics.get('train_loss'),
                'global_step': task_result.metrics.get('global_step'),
            }
            
            logger.info(f"✓ Task {task} completed")
            logger.info(f"  Final loss: {task_result.metrics.get('train_loss'):.4f}")
        
        # After sequential training, evaluate on full dataset
        logger.info(f"\n{'='*60}")
        logger.info("Final evaluation on full dataset")
        logger.info(f"{'='*60}")
        trainer.train_dataset = train_dataset
        trainer.eval_dataset = eval_dataset
        final_metrics = trainer.evaluate()
        
        logger.info(f"Final metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
        
        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info("Sequential Training Summary:")
        for task, result in all_results.items():
            logger.info(f"  {task}: loss={result['loss']:.4f}")
        logger.info(f"{'='*60}\n")
        
        train_result = task_result  # Last task's result
    
    else:
        # Mixed training (all tasks together)
        logger.info("🔀 MIXED TRAINING MODE")
        train_result = trainer.train(
            resume_from_checkpoint=checkpoint_path
        )
    # ======== END ADD BLOCK ========
    
    # Save results (existing code continues)
    logger.info("Saving model...")
    trainer.save_model()
    # ...
```

---

### 3. OPTIONAL: Task-Aware Selection

**File**: `colm/train/subset_trainer_distributed.py` **Line**: ~1358

**For balanced task representation during mixed training**, modify `select_data()`:

```python
def select_data(self, inputs, max_samples=64, source_list=None, model=None):
    """
    Select a subset of inputs based on model representations.
    
    MODIFIED: Can now balance across tasks during mixed training.
    """
    tocpu = lambda x: x.cpu().numpy()
    
    # ============ ADD THIS BLOCK (at start) ============
    # Optional: Task-aware selection for mixed training
    if hasattr(self.args, 'task_aware_selection') and \
       self.args.task_aware_selection and \
       source_list is not None:
        
        # source_list now contains task IDs: [0, 0, 1, 1, 2, 2, ...]
        # Group inputs by task
        unique_tasks = np.unique(source_list)
        samples_per_task = max(1, max_samples // len(unique_tasks))
        
        all_selected_idx = []
        all_selected_weights = []
        
        for task_id in unique_tasks:
            # Get indices for this task
            task_mask = (np.array(source_list) == task_id)
            task_reps = inputs[task_mask]
            
            if len(task_reps) == 0:
                continue
            
            # Select from this task
            task_selected_idx, task_weights = self._select_from_subset(
                task_reps, 
                min(samples_per_task, len(task_reps)),
                source_list=None,  # Already filtered to task
                model=model
            )
            
            # Map back to original indices
            task_original_indices = np.where(task_mask)[0]
            all_selected_idx.extend(task_original_indices[task_selected_idx])
            all_selected_weights.extend(task_weights)
        
        # Normalize weights
        if len(all_selected_weights) > 0:
            all_selected_weights = np.array(all_selected_weights, dtype=float)
            all_selected_weights /= all_selected_weights.sum()
            return tocpu(np.array(all_selected_idx)), tocpu(all_selected_weights)
    # ======== END ADD BLOCK ========
    
    # Existing selection code continues as-is
    if(self.method in ["submodlib", "weightedsubmodlib"]): 
        return self.select_data_facloc(inputs, max_samples, source_list, metric=self.args.facility_similarity)
    
    # ... rest of existing method
```

---

### 4. ADD TRAINING ARGUMENTS

**File**: `colm/train/training_arguments.py` **Line**: ~240 (end of TrainingArguments)

**Add these fields**:
```python
@dataclass
class TrainingArguments(TA):
    # ... existing fields ...
    
    # ============ ADD THIS BLOCK ============
    # Sequential training configuration
    sequential_training: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, train on each task sequentially. "
                "If False, train on all tasks mixed together."
            )
        },
    )
    
    steps_per_task: int = field(
        default=1000,
        metadata={
            "help": (
                "Number of training steps per task in sequential mode. "
                "Used for task cycling scheduler."
            )
        },
    )
    
    task_aware_selection: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, balance data selection across tasks during mixed training. "
                "Only used when data_selection_method != 'none'."
            )
        },
    )
    
    per_task_eval: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, log per-task evaluation loss metrics. "
                "Only applicable for multi-task training."
            )
        },
    )
    # ======== END ADD BLOCK ========
```

---

## COMMAND LINE EXAMPLES

### Example 1: Sequential Training (One Task at a Time)
```bash
python colm/train/train_multitask.py \
    --model_name_or_path meta-llama/Llama-2-7b \
    --dataset_path /path/to/combined_dataset \
    --output_dir ./results_sequential \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --max_seq_length 512 \
    --sequential_training \
    --per_task_eval \
    --wandb_entity your_entity \
    --wandb_project colm_sequential \
    --report_to wandb
```

### Example 2: Mixed Training with Task-Aware Selection
```bash
python colm/train/train_multitask.py \
    --model_name_or_path meta-llama/Llama-2-7b \
    --dataset_path /path/to/combined_dataset \
    --output_dir ./results_mixed \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --data_selection_method greats \
    --data_selection_unit rep \
    --small_batch_ratio 0.5 \
    --source_wise_selection proportional \
    --task_aware_selection \
    --per_task_eval \
    --wandb_entity your_entity \
    --wandb_project colm_mixed_tasks \
    --report_to wandb
```

### Example 3: Via YAML Config
**config.yaml**:
```yaml
model:
  model_id: meta-llama/Llama-2-7b

training:
  output_dir: ./results_sequential
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 5e-5
  sequential_training: true
  per_task_eval: true
  
data:
  dataset_path: /path/to/combined_dataset
  max_seq_length: 512

logging:
  report_to: wandb
  wandb_entity: your_entity
  wandb_project: colm_sequential
  
lora:
  lora_r: 128
  lora_alpha: 512
  lora_dropout: 0.05

gpu:
  device_ids: "0,1,2,3"
  use_tf32: true
```

```bash
python colm/train/train_multitask.py config.yaml
```

---

## EXPECTED OUTPUT

### Sequential Training Console Output:
```
====================================================
Task 1/3: task_math
====================================================
Training samples: 5000
Eval samples: 500
Epoch 1/3:  45%|████▌    | 2250/5000 [15:32<18:45, 2.05it/s, loss=2.45]
...
✓ Task task_math completed
  Final loss: 1.89

====================================================
Task 2/3: task_logic
====================================================
Training samples: 3000
Eval samples: 300
...
✓ Task task_logic completed
  Final loss: 1.75

====================================================
Final evaluation on full dataset
====================================================
Final metrics:
  eval_loss: 1.82
  eval_perplexity: 6.17
  eval_loss_task_math: 1.85
  eval_loss_task_logic: 1.79

====================================================
Sequential Training Summary:
  task_math: loss=1.89
  task_logic: loss=1.75
  task_qa: loss=1.92
====================================================
```

### WandB Metrics:
```
Training:
- loss (global, per-step)
- train_perplexity
- learning_rate
- grad_norm

Evaluation (periodic):
- eval_loss (global)
- eval_loss_task_math
- eval_loss_task_logic
- eval_loss_task_qa
- eval_perplexity_task_math
- eval_perplexity_task_logic
- eval_perplexity_task_qa
- gpu_memory_used_gb
- cpu_percent
```

---

## TESTING YOUR CHANGES

### 1. Test Per-Task Eval Logging (Small Dataset)
```bash
# Create small test dataset
python -c "
import datasets
from datasets import DatasetDict

# Create minimal dataset
data = {
    'text': ['Hello'] * 100,
    'task': ['task1'] * 50 + ['task2'] * 50,
    'source': [0] * 100,
}
dataset = datasets.Dataset.from_dict(data)
dataset_dict = DatasetDict({
    'train': dataset,
    'validation': dataset,
})
dataset_dict.save_to_disk('/tmp/test_dataset')
"

# Run training with per-task eval
python colm/train/train_multitask.py \
    --model_name_or_path gpt2 \
    --dataset_path /tmp/test_dataset \
    --output_dir ./test_output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --max_seq_length 128 \
    --per_task_eval \
    --evaluation_strategy steps \
    --eval_steps 10
```

### 2. Test Sequential Training
```bash
# Same dataset as above, but sequential
python colm/train/train_multitask.py \
    --model_name_or_path gpt2 \
    --dataset_path /tmp/test_dataset \
    --output_dir ./test_sequential \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --max_seq_length 128 \
    --sequential_training \
    --per_task_eval
```

### 3. Verify Loss Tracking
```python
# Python script to check metrics were logged
import json
import os

output_dir = "./test_sequential"
metrics_file = os.path.join(output_dir, "trainer_state.json")

if os.path.exists(metrics_file):
    with open(metrics_file) as f:
        state = json.load(f)
    print("Training metrics logged:")
    print(json.dumps(state["log_history"][-5:], indent=2))
else:
    print("metrics file not found")
```

---

## DEBUGGING CHECKLIST

- [ ] Verify datasets have 'task' column
- [ ] Check `task` column contains task names/IDs
- [ ] Verify eval_dataset is not None before per-task eval
- [ ] Check WandB credentials are set
- [ ] Monitor first eval step for per-task metrics
- [ ] Test with small `--max_steps 10` first
- [ ] Verify loss is computed without NaNs
- [ ] Check GPU memory with task-specific eval

