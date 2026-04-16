# Checkpoint & Training Flow: CoLM vs Riemannian

## 🎯 Quick Answer

| Aspect | CoLM | Riemannian |
|--------|------|-----------|
| **Training Style** | Single unified run on mixed data | Sequential multi-task |
| **Checkpoint Handling** | Save once after training | Save after each task |
| **Next Task Start** | N/A (only one training run) | Start from previous task's weights! |
| **Cumulative Learning** | No (single unified training) | **YES** (accumulates across tasks) |

---

## 🔄 Riemannian Multi-Task Training Flow

### Code Structure (src/run_experimet.py)

```python
def run_tasks(config):
    tasks = config.tasks
    
    # STEP 1: Load model ONCE at the start
    tokenizer = model_loader.load_tokenizer(config)
    model = model_loader.load_model(config)  # ← Load base model ONCE
    model = model_loader.get_peft(config, model)  # ← Add LoRA ONCE
    
    dataset = data_preparation.load_dataset(config)
    
    # STEP 2: Loop through tasks with the SAME model object
    for i, task in enumerate(tasks):
        print(f"Running task {i}: {task}")
        
        if task is Task.FINETUNE:
            # Train with model from previous iterations
            run_finetune(config, model, tokenizer, 
                        dataset['train'],        # Task i's training data
                        dataset['validation'])   # Task i's validation data
            # ↓ model gets updated in-place with task i weights
            
        elif task is Task.VALIDATE:
            # Evaluate with accumulated weights
            run_inference(config, pl, dataset['validation'])
```

### The Critical Point: Model is Loaded ONCE

```
Timeline of model object:

┌─────────────────────────────────────────────────────────┐
│ model = load_model(config)  # Load base pretrained once │
│ model = get_peft(config, model)  # Add LoRA once        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓ (Same model object enters loop)
    ┌────────────────────────────────────────────────────┐
    │ Iteration 0 (Task 0):                              │
    │  run_finetune(model, ...)  # Update model in-place │
    │  model.save_pretrained(...)  # Save task 0 weights  │
    └────────────────┬───────────────────────────────────┘
                     │ (model now has task 0 weights)
                     ↓
    ┌────────────────────────────────────────────────────┐
    │ Iteration 1 (Task 1):                              │
    │  run_finetune(model, ...)  # Update model again     │
    │  model.save_pretrained(...)  # Save task 0+1 weights│
    └────────────────┬───────────────────────────────────┘
                     │ (model now has task 0 + task 1 weights!)
                     ↓
    ┌────────────────────────────────────────────────────┐
    │ Iteration 2 (Task 2):                              │
    │  run_finetune(model, ...)  # Update from task 0+1  │
    │  model.save_pretrained(...)  # Save task 0+1+2!    │
    └────────────────┬───────────────────────────────────┘
                     │ (Cumulative!)
                     ↓
           ... continues ...
```

---

## 📊 What This Means

### Task 0 → Task 1 Transition

```python
# Task 0 completes
run_finetune(config, model, tokenizer, task0_data, task0_val)
model.save_pretrained(...)  # Saves task 0 weights

# Next iteration starts (Task 1)
# model STILL HAS task 0 weights in memory!

# Task 1 training starts FROM task 0 weights
run_finetune(config, model, tokenizer, task1_data, task1_val)
# ↑ Training continues from accumulated weights
# ↑ Updates are applied ON TOP of task 0 weights
```

### Cumulative Weight Accumulation

```
Initial state: model = base pretrained (e.g., Phi-2)

After Task 0:   model = base + Δ_task0
After Task 1:   model = base + Δ_task0 + Δ_task1
After Task 2:   model = base + Δ_task0 + Δ_task1 + Δ_task2
...
Final:          model = base + Σ(Δ_all_tasks)
```

---

## Comparison: CoLM vs Riemannian Checkpoint Strategy

### CoLM: Single Unified Training

```
┌─────────────────────────────────────────────┐
│ Load all 14 sources mixed into one dataset  │
└────────────────────┬────────────────────────┘
                     │
                     ↓
    ┌────────────────────────────────────────┐
    │ Train on mixed data (hours 0-5.5)      │
    │ Single training loop, all sources      │
    │ model = base + Δ_all_14_sources        │
    │ save_checkpoint()                      │
    └────────────────┬───────────────────────┘
                     │
                     ↓
    ┌────────────────────────────────────────┐
    │ Evaluate on all 17 benchmarks (once)   │
    │ MATH, GSM8K, SuperGLUE tasks, etc.     │
    └────────────────────────────────────────┘

Result: One checkpoint with all 14 sources learned
```

### Riemannian: Sequential Multi-Task Training

```
┌────────────────────────────────────────────┐
│ Load base model + LoRA (once)              │
│ model = Phi-2 (base pretrained)            │
└────────────────┬───────────────────────────┘
                 │
        ┌────────┴──────────┐
        │                   │
        ↓                   ↓
   Task 0 Loop         Task 1 Loop
   ┌──────────────┐    ┌──────────────┐
   │ Load task 0  │    │ Load task 1  │
   │ Train:       │    │ Train:       │  ← Starting from task 0 weights!
   │ base+Δ0      │    │ base+Δ0+Δ1   │
   │ Save         │    │ Save         │
   │ Eval task 0  │    │ Eval task 1  │
   └──────────────┘    └──────────────┘
                           ↓
                        Task 2 Loop
                        ┌──────────────┐
                        │ Load task 2  │
                        │ Train:       │  ← From task 0+1 weights!
                        │ base+Δ0+Δ1+Δ2
                        │ Save         │
                        │ Eval task 2  │
                        └──────────────┘

Result: Cumulative checkpoints - each contains all previous tasks!
```

---

## ⚠️ Important Implications

### Riemannian: Catastrophic Forgetting Risk

```
Task 0 training: Learns MathInstruct well
  ↓ (weights saved)
  
Task 1 training: Learns Task 1, BUT might FORGET MathInstruct
  ↓ (weights saved with task 1 favored)
  
Task 2 training: Learns Task 2, might FORGET both previous
  ↓ (final model specialized for task 2, worse on 0 and 1)

Result: Final model might be BEST for last task, WORSE for earlier tasks
This is called "catastrophic forgetting"
```

### To Prevent Catastrophic Forgetting, Riemannian Could Use:

1. **Replay**: Mix in old task data during new task training
2. **Continual Learning**: Special loss functions (e.g., EWC - Elastic Weight Consolidation)
3. **Task-Specific Heads**: Different output layers per task
4. **Multi-Head Experts**: Different LoRA modules per task

---

## 💾 Checkpoint Saving Comparison

### CoLM Checkpoints
```
checkpoints/
├─ checkpoint-256/      # Intermediate (optional)
├─ checkpoint-512/      # Intermediate
└─ final_checkpoint/    # Final model
    └─ (Contains: base + learned from all 14 sources)
        └─ Evaluable on all 17 benchmarks
```

### Riemannian Checkpoints
```
peft_pretrained_path_0/    # After task 0
  └─ adapter_model.bin (base + Δ_task0)
  
peft_pretrained_path_1/    # After task 1
  └─ adapter_model.bin (base + Δ_task0 + Δ_task1)
  
peft_pretrained_path_2/    # After task 2
  └─ adapter_model.bin (base + Δ_task0 + Δ_task1 + Δ_task2)
  
...

Final checkpoint: Contains ALL accumulated weights
Performance: Best on last task, may degrade on earlier tasks
```

---

## 🔑 Key Question: Fresh Start or Cumulative?

```
Q: Does each task start from exactly the same checkpoint?
A: NO! Each task starts from the accumulated weights of ALL previous tasks

Q: Is this intentional or a side effect?
A: Likely INTENTIONAL for continual learning / multi-task learning

Q: Is this good or bad?
A: MIXED:
   ✅ Pros:
      - Transfer learning: Earlier tasks help later tasks
      - Parameter efficiency: Single LoRA module per layer
      - Faster training: Initialized from warmer weights
      
   ❌ Cons:
      - Catastrophic forgetting risk
      - Evaluation biased toward final tasks
      - Need special techniques to prevent forgetting
```

---

## 📈 Cumulative Effect Visualization

If we measure accuracy on each task:

```
Multi-task training with cumulative weights:

        Task 0    Task 1    Task 2    Task 3
After 0: 85%      N/A       N/A       N/A
After 1: 82%      88%       N/A       N/A      ← Slight drop on task 0!
After 2: 78%      85%       90%       N/A      ← Forgetting task 0 & 1!
After 3: 70%      80%       87%       92%      ← Task 3 best, others worse!

⚠️ Catastrophic forgetting pattern!
```

vs Separate Training:

```
Training each task independently from fresh:

        Task 0    Task 1    Task 2    Task 3
Trained: 85%      88%       90%       92%      ← All tasks fine!
```

---

## 🎯 Summary

**Your question**: "After saving checkpoint from task 1, does task 2 training start from that checkpoint?"

**Answer**: 
✅ **YES, exactly!** 

The model object in memory is never reset. Each task:
1. Loads the same model object (from previous task's weights)
2. Trains on new data (starting from accumulated weights)
3. Saves the checkpoint
4. **SAME MODEL object continues to next task**

This creates **cumulative learning** where:
- Task 0 weights: base + Δ_task0
- Task 1 weights: base + Δ_task0 + Δ_task1 ← Includes task 0!
- Task 2 weights: base + Δ_task0 + Δ_task1 + Δ_task2 ← Includes all!

**Benefit**: Transfer learning across tasks  
**Risk**: Catastrophic forgetting (final task overwrites earlier learning)

