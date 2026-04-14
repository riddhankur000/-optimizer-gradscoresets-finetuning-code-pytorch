# Train Dataset Sampling: The 2nd Method (Multi-Task)

## 📋 Quick Reference

### **How Data is Mixed in Combined Dataset**

```
┌─────────────────────────────────────────────────────────┐
│         MULTI-TASK TRAINING DATASET STRUCTURE           │
└─────────────────────────────────────────────────────────┘

STEP 1: Load Separate Datasets
├─ MetaMathQA:  300,000 examples  ┐
├─ GSM8K:         8,536 examples  │ Each kept separate
├─ MATH:          7,500 examples  ├─ train_test_split()
└─ Other:         Variable        │ shuffle=True, seed=42
                                   ┘

STEP 2: Split Each Dataset (85% train / 15% val)
├─ MetaMathQA train:  255,000 examples ┐
├─ MetaMathQA val:     45,000 examples │
├─ GSM8K train:         7,256 examples │ Each split independently
├─ GSM8K val:           1,280 examples │ with shuffle=True
├─ MATH train:          6,375 examples │
└─ MATH val:            1,125 examples ┘

STEP 3: Concatenate All Train Sets (SHUFFLE OFF)
├─ MetaMathQA train (255K)
├─ GSM8K train (7.2K)  
├─ MATH train (6.4K)
└─ ✓ RESULT: 268,631 examples in random order
              (randomized in step 2, not step 3)

STEP 4: During Training (Dataloader Shuffling)
├─ Epoch 0: Dataloader shuffles again with shuffle=True
├─ Epoch 1: Different shuffle, new random order
└─ ✓ Each epoch gets different task mixing
```

---

## 🎲 Whether Tasks are Randomly Mixed

### **YES - But with Important Details:**

| Aspect | Details |
|--------|---------|
| **Local Shuffling (Per-Dataset)** | Each dataset shuffled independently before concatenation |
| **Global Mixing** | All datasets concatenated without further shuffling |
| **Dataloader Shuffling** | PyTorch DataLoader shuffles again during training |
| **Per-Batch Composition** | Each batch gets random mix of tasks |
| **Reproducibility** | seed=42 makes split reproducible, but dataloader shuffle varies per epoch |

### **Example Scenario:**

```
After concatenation, combined train dataset looks like:
Index  Task        Source
─────  ────────    ──────────────
0      MetaMathQA  "Solve: x^2 + ..."
1      GSM8K       "Grade school: There are..."
2      MATH        "Prove: If a > b then..."
3      MetaMathQA  "Simplify: (a+b)/(c-d)..."
4      MetaMathQA  "Factor: 12x^2 - 18x..."
5      GSM8K       "Grade school: Alice has..."
...

✓ Tasks appear in random order
✓ More MetaMathQA because it's the largest dataset (255K vs 7K vs 6K)
✓ Probability of task in batch ≈ (size_of_task / total_size)
```

---

## 📊 Example Task Distribution

Based on the code, with MetaMathQA (255k) + GSM8K (7.2k) + MATH (6.4k):

```
Total: 269K examples

MetaMathQA: 255K examples │████████████████████│ 94.8%
GSM8K:       7.2K examples │██████                │  2.7%
MATH:        6.4K examples │██████                │  2.5%
```

**Batch Composition (batch_size=2):**
- Probability both MetaMathQA: 0.948² = 89.8%
- Probability both GSM8K: 0.027² = 0.07%
- Probability mixed tasks: ~10%

---

## 🔧 How to View Random Samples

### **Option 1: Run Visualization Script**

```bash
cd /home1/adamw_vs_muon_2/llm-experiments

# First, create the combined dataset
python colm/data/load_math_datasets.py

# Then visualize samples
python scripts/visualize_multi_task_samples.py ./colm_math_combined_dataset
```

**Output shows:**
- Random samples from each task
- Task distribution pie chart
- Batch composition simulation
- Consecutive samples (to see task switching)

### **Option 2: Quick Python Script**

```python
from datasets import load_from_disk
import random

# Load
dataset = load_from_disk('./colm_math_combined_dataset')['train']

# Show 5 random samples
indices = random.sample(range(len(dataset)), 5)

for i, idx in enumerate(indices, 1):
    example = dataset[idx]
    print(f"\n--- SAMPLE {i} ---")
    print(f"Task: {example['task']}")
    print(f"Source: {example['source']}")
    print(f"Text preview: {example['text'][:300]}...\n")
```

### **Option 3: Check Distribution**

```python
from collections import Counter

# Sample 1000 examples
indices = random.sample(range(len(dataset)), 1000)
tasks = [dataset[idx]['task'] for idx in indices]
counter = Counter(tasks)

for task, count in counter.most_common():
    pct = (count / 1000) * 100
    print(f"{task}: {pct:.1f}%")
```

---

## 📝 Key Differences: Single vs Multi-Task

### **Your Current Method (Single Combined):**

```
MathInstruct.jsonl
    ↓
Pre-combined + pre-tokenized
    ↓
Load once → Use as-is
    ↓
Global metrics only
    ↓
Task composition: FIXED (whatever was pre-combined)
```

### **2nd Method (Multi-Task):**

```
MetaMathQA dataset ┐
GSM8K dataset      ├─ Load separately
MATH dataset       ┘
    ↓
Apply per-task formatters
    ↓
Split each (85/15)
    ↓
Concatenate (shuffled order from step 2)
    ↓
Dataloader shuffles again (each epoch!)
    ↓
Random task mixing every batch + epoch
    ↓
Per-task metrics (see which task helps most)
```

---

## ✅ Summary

| Question | Answer |
|----------|--------|
| **Are tasks randomly sampled?** | ✓ Yes, but weighted by dataset size |
| **Is the order fixed?** | ✗ No - shuffled during split + dataloader |
| **Can I see different batch mixes?** | ✓ Yes - each epoch and batch varies |
| **Do I get per-task metrics?** | ✓ Yes - that's the whole point! |
| **How to view samples?** | Run `visualize_multi_task_samples.py` |
| **Will MetaMathQA dominate?** | ✓ Yes (94.8% of data) - might want weighted sampling instead |

---

## 🎯 Next Step (Optional Enhancement)

If MetaMathQA is too dominant (94.8%), you can add **weighted random sampling**:

```python
# In training loop
from torch.utils.data import WeightedRandomSampler

# Give equal weight to each task
task_weights = []
for example in dataset:
    task = example['task']
    if task == 'MetaMathQA':
        weight = 1.0 / 255000  # Downweight large dataset
    elif task == 'GSM8K':
        weight = 1.0 / 7200    # Keep small datasets
    else:
        weight = 1.0 / 6400
    task_weights.append(weight)

sampler = WeightedRandomSampler(
    weights=task_weights,
    num_samples=len(dataset),
    replacement=True
)

dataloader = DataLoader(
    dataset,
    sampler=sampler,
    batch_size=2
)
```

This would make batches have ~33% MetaMathQA, 33% GSM8K, 33% MATH instead of 95/2/3.

