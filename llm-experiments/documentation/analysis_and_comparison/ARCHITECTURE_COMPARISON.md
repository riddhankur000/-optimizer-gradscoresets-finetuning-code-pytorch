# Visual Architecture Comparison

## 🏗️ **ARCHITECTURE DIAGRAMS**

### **YOUR CURRENT PIPELINE (Single Combined Dataset)**

```
┌─────────────────────────────────────────────────────────────────┐
│                   Training Pipeline Flow                        │
└─────────────────────────────────────────────────────────────────┘

  MathInstruct.jsonl (235K examples)
         │
         ├─ COMBINED DATA
         │  (14 math sources mixed)
         │
         ▼
  ┌──────────────────────┐
  │  load_raw_dataset()  │
  │ (subset_selection=   │
  │  use_small_sources)  │
  └──────────┬───────────┘
             │
         + ─ ─ + ─ ─ +
         │           │      Group by source
    Small Sources Full    (e.g., TheoremQA=100%)
    (100% used)          Large Sources Partial
                         (e.g., MATH=86%)
             │
             ▼
  ┌──────────────────────┐
  │  Generic Formatting  │ ─── Same prompt template
  │  (SupervisedDataset) │     for all 14 datasets
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ Tokenization         │ ─── Pre-tokenized
  │ (input_ids, labels)  │     into JSON
  └──────────┬───────────┘
             │
         ┌───┴────────────────────┐
         │ Metadata Preserved:    │
         │ - source (0-13)        │
         │ - weight               │
         │ - completion_length    │
         │ - original_index       │
         └───┬────────────────────┘
             │
             ▼
  ┌──────────────────────────────┐
  │  DataCollatorForSupervised   │
  │  DatasetWithSource           │
  │  (Batch size=2)              │
  └──────────┬───────────────────┘
             │
      ┌──────┴──────┐
      │ Batch       │
      ├─ Pad        │
      ├─ Mask       │
      ├─ Sources    │
      └─ Metadata   │
             │
             ▼
  ┌──────────────────────┐
  │  Model Forward Pass  │
  │  (Phi-2 LoRA)        │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Loss Calculation    │ ─── GLOBAL LOSS ONLY
  │  (HuggingFace Loss)  │     No per-source tracking
  └──────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  EVALUATION METRICS (Global Only)                               │
  ├─────────────────────────────────────────────────────────────────┤
  │  Accuracy: 0.72 (combined)                                      │
  │  Loss: 0.38 (combined)                                          │
  │  Perplexity: 1.46 (combined)                                    │
  │                                                                 │
  │  ✗ Cannot analyze per-source performance                       │
  │  ✗ Cannot identify weak sources                                │
  │  ✗ Cannot do per-source early stopping                         │
  └─────────────────────────────────────────────────────────────────┘

```

---

### **RIEMANNIAN PROPOSED PIPELINE (Multi-Source with Per-Task Tracking)**

```
┌─────────────────────────────────────────────────────────────────┐
│                   Training Pipeline Flow                         │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐  ┌──────────────┐
  │ MetaMathQA (~300K)   GSM8K (~8K)  │
  │ (From HF Hub)    │  │ (From HF Hub)│
  └──────┬───────────┘  └──────┬───────┘
         │                     │
         ├─────────┬───────────┤
         │         │           │
         │    SEPARATE DATASETS
         │    (Independent loading)
         │
         ▼
  ┌────────────────────────────────────────┐
  │ Dataset-Specific Formatters             │
  ├────────────────────────────────────────┤
  │  MetaMathQA         GSM8K              │
  │  _get_MetaMath_     _get_GSM8K_       │
  │  instructions()     instructions()    │
  │  (Chat template)    (Chat template)   │
  └────────┬───────────────────────┬──────┘
           │                       │
           ▼                       ▼
  ┌──────────────────┐  ┌──────────────────┐
  │ Formatted Texts  │  │ Formatted Texts  │
  │ + task='MetaMath'│  │ + task='GSM8K'   │
  │ + task metadata  │  │ + task metadata  │
  └────────┬─────────┘  └────────┬─────────┘
           │                     │
           ├─────────┬───────────┤
           │         │           │
    ┌──────────────────────────────────┐
    │  Split into Train/Val (80/20 ea) │
    └──────────────┬───────────────────┘
                   │
                   ▼
    ┌────────────────────────────────┐
    │  Concatenate All Datasets      │
    │  (Mixed batch training)        │
    │  Combined Train Dataset        │
    │  (MetaMathQA.train +           │
    │   GSM8K.train)                 │
    └──────────────┬─────────────────┘
                   │
                   ▼
    ┌────────────────────────────────┐
    │  Tokenize on-the-fly OR Cache  │
    │  (Text-based format)           │
    └──────────────┬─────────────────┘
                   │
                   ▼
    ┌────────────────────────────────┐
    │  Data Collator                 │
    │  (Batch size=2, Mixed tasks)   │
    │                                │
    │  Example 1: MetaMathQA sample  │
    │  Example 2: GSM8K sample       │
    └──────────────┬─────────────────┘
                   │
         ┌─────────┴──────────┐
         │ Track in Batch:    │
         │ - task_names       │
         │ - sources          │
         │ - original indices │
         └─────────┬──────────┘
                   │
                   ▼
    ┌────────────────────────────────┐
    │  Model Forward Pass            │
    │  (Phi-2 LoRA)                  │
    └──────────────┬─────────────────┘
                   │
                   ▼
    ┌────────────────────────────────┐
    │  Per-Task Loss Tracking        │
    │  (PerTaskMetricsTracker)       │
    │                                │
    │  if task=='MetaMathQA':        │
    │    task_tracker.record(loss)   │
    │  elif task=='GSM8K':           │
    │    task_tracker.record(loss)   │
    └──────────────┬─────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │    EVALUATION METRICS (Per-Task)     │
    ├──────────────────────────────────────┤
    │  MetaMathQA:                         │
    │    Accuracy: 0.74                    │
    │    Loss: 0.36                        │
    │    Perplexity: 1.43                  │
    │                                      │
    │  GSM8K:                              │
    │    Accuracy: 0.68                    │
    │    Loss: 0.42                        │
    │    Perplexity: 1.52                  │
    │                                      │
    │  Average:                            │
    │    Accuracy: 0.71                    │
    │    Loss: 0.39                        │
    │    Perplexity: 1.48                  │
    │                                      │
    │  ✓ Per-task performance visible      │
    │  ✓ Identify which dataset helps more │
    │  ✓ Task-specific early stopping      │
    │  ✓ Dataset contribution analysis     │
    └──────────────────────────────────────┘

```

---

## 🔄 **BATCH COMPOSITION COMPARISON**

### Your Method During Training:

```
Step 100 - Batch 1:
┌─────────────────────────────────────────┐
│ Example 1:                              │
│  - Source: MATH (source_id=0)          │
│  - Weight: 0.92                         │
│  - Completion Len: 145 tokens          │
│  - Input: "Solve this algebraic..."    │
│  - Loss: 0.35                           │
│                                         │
│ Example 2:                              │
│  - Source: TheoremQA (source_id=1)     │
│  - Weight: 0.8                          │
│  - Completion Len: 89 tokens           │
│  - Input: "Prove this theorem..."      │
│  - Loss: 0.42                           │
│                                         │
│ Batch Loss: (0.35 + 0.42) / 2 = 0.385  │
│                                         │
│ ✗ Cannot determine why loss differs    │
│ ✗ Aggregated metrics only              │
└─────────────────────────────────────────┘

Logged to W&B:
- loss: 0.385
- learning_rate: 2e-05
- epoch: 0.01
```

### RiemanianFinetune Method During Training:

```
Step 100 - Batch 1:
┌──────────────────────────────────────────────┐
│ Example 1:                                   │
│  - Task: MetaMathQA                         │
│  - Source: MetaMathQA                       │
│  - Input: "Solve this algebraic..."         │
│  - Loss: 0.35                                │
│                                              │
│ Example 2:                                   │
│  - Task: GSM8K                              │
│  - Source: GSM8K                            │
│  - Input: "Grade school math problem..."    │
│  - Loss: 0.42                                │
│                                              │
│ Batch Loss: (0.35 + 0.42) / 2 = 0.385       │
│                                              │
│ ✓ Task info tracked for each example        │
│ ✓ Can compute per-task statistics           │
│ ✓ Can identify task-specific issues         │
└──────────────────────────────────────────────┘

Logged to W&B:
- loss: 0.385
- MetaMathQA/loss: 0.35
- GSM8K/loss: 0.42
- learning_rate: 2e-05
- epoch: 0.01
```

---

## 📊 **EVALUATION FRAMEWORK COMPARISON**

### Your Current Evaluation:

```
metrics.compute(predictions, labels)
        │
        └─► global_accuracy = 0.72
        └─► global_loss = 0.38
        └─► global_perplexity = 1.46

✗ Single number for all 14 math datasets
✗ Cannot identify MATH vs TheoremQA performance
✗ Cannot optimize per-source
```

### Proposed Per-Task Evaluation:

```
metrics.compute(predictions, labels, per_task=True)
        │
        ├─► MetaMathQA_accuracy = 0.74 ─────┐
        ├─► MetaMathQA_loss = 0.36          │
        ├─► MetaMathQA_perplexity = 1.43    │ Dataset 1
        │                                    │
        ├─► GSM8K_accuracy = 0.68 ─────────┤ Dataset 2
        ├─► GSM8K_loss = 0.42              │
        ├─► GSM8K_perplexity = 1.52        │
        │                                   │
        └─► Global: Average of above       ─┴─ Global

✓ Per-task metrics tracked
✓ Can see which dataset needs help
✓ Can optimize specific weak areas
```

---

## 🎯 **WORKFLOW COMPARISON**

### Your Workflow:

```
1. ONE TIME: Create MathInstruct.jsonl
   └─ Combine 14 datasets manually
   └─ Apply generic formatting
   └─ Pre-tokenize
   └─ Fix for life (or restart)

2. TRAINING: Load pre-combined data
   └─ Fast loading
   └─ Global evaluation
   └─ Generic metrics

3. ANALYSIS: Global results only
   └─ What worked overall?
   └─ ???
```

### Proposed Workflow:

```
1. SETUP: Create dataset loaders
   └─ One loader per dataset
   └─ One formatter per dataset
   └─ Reusable, modular

2. TRAINING: Load & combine dynamically
   └─ Easy to add/remove datasets
   └─ Per-task tracking
   └─ Task-specific metrics

3. ANALYSIS: Detailed per-task insights
   └─ Which datasets help most?
   └─ Which tasks are hardest?
   └─ What to prioritize?
   └─ Task-specific improvements
```

---

## 💾 **DISK STORAGE COMPARISON**

### Your Method:
```
MathInstruct.jsonl
  ├─ Raw format: ~500 MB
  ├─ Pre-tokenized: ~2 GB (input_ids + labels)
  ├─ + metadata: ~50 MB
  └─ Total: ~2.55 GB
  
  Pros: ✓ Fast loading, ✓ Compact
  Cons: ✗ Not modular, ✗ Hard to update
```

### Proposed Method:
```
combined_math_dataset/ (from save_to_disk)
  ├─ train/
  │  ├─ data-00000-of-00003.arrow
  │  └─ dataset_info.json
  ├─ validation/
  │  ├─ data-00000-of-00001.arrow
  │  └─ dataset_info.json
  └─ test/
     ├─ data-00000-of-00001.arrow
     └─ dataset_info.json
  
  Total: ~3.2 GB (slightly larger, text-based)
  
  Pros: ✓ Modular, ✓ Easy to update, ✓ Reusable
  Cons: ✗ Slightly larger, ✗ Requires tokenization at load
```

---

## 🚀 **MIGRATION PATH**

### Phase 1: Add Per-Task Tracking (Easy)
```python
# Minimal changes to existing code
# Just add task_name to each batch
# No changes to model or training logic
```

### Phase 2: Add Multiple Dataset Support (Medium)
```python
# Add dataset loaders
# Add formatters
# Add combining logic
# Update evaluation
```

### Phase 3: Full RiemanianFinetune Adoption (Complete)
```python
# Complete per-task evaluation
# Task-specific early stopping
# Task-specific hyperparameter tuning
# Full metrics dashboard
```

