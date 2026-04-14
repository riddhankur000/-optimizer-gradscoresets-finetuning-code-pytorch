# Dataset Structure & Format Comparison

## 📊 **PART 1: Current MathInstruct Pipeline (Your Method)**

### Dataset Structure:
```
MathInstruct.jsonl (235,833 examples)
├── instruction: str
├── input: str (optional)
├── output: str
├── source: str (14 different math datasets)
├── completion_length: int
└── weight: float
```

### Training Pipeline Flow:

```
1. LOAD RAW DATA
   MathInstruct.jsonl (235,833 lines)
   ↓

2. SAMPLING (subset_selection="use_small_sources")
   - Group by source (14 datasets)
   - Keep 100% of small sources (e.g., TheoremQA: 500 examples)
   - Take longest 86% of large sources (e.g., MATH: 20000 → 17200 examples)
   ↓

3. FORMAT TO PROMPT
   Template: "Below is an instruction...\n\n### Instruction:\n{instruction}\n..."
   ↓

4. TOKENIZE (SupervisedDataset)
   input_ids: [token_ids...]
   labels: [token_ids...]
   metadata: {
       'sources': source_id (0-13),
       'indices': original_index,
       'weights': importance_weight,
       'completion_lengths': output_length
   }
   ↓

5. BATCH & COLLATE (DataCollatorForSupervisedDatasetWithSource)
   Batch size: 2 examples
   ↓ Pad sequences, merge metadata
   ↓

6. MODEL FORWARD PASS
   input_ids → model → logits
   labels → loss calculation (ignores padding with IGNORE_INDEX=-100)
```

### Data Format After Loading:
```python
{
    'input_ids': tensor([[token_ids...], [token_ids...]]),
    'labels': tensor([[label_ids...], [label_ids...]]),
    'attention_mask': tensor([[1,1,1,0,0], [1,1,1,1,1]]),
    'sources': [2, 5],  # Dataset source IDs
    'indices': [99085, 52001],  # Original indices
    'weights': [0.185, 0.320],  # Importance weights
    'completion_lengths': [81, 120]  # Response lengths
}
```

### Evaluation Pipeline:
```
- Single dataset: MathInstruct
- Single metric: Global loss/perplexity/accuracy
- NO per-source evaluation
- NO task-specific analysis
- Cannot separately evaluate MATH vs TheoremQA performance
```

---

## 🔄 **PART 2: RiemanianFinetune Pipeline (Commonsense Strategy)**

### Dataset Structure:
```
8 Separate Datasets (from HF Hub)
├── BoolQ (bool-based QA)
│   ├── train: {question, answer}
│   ├── validation: {question, answer}
│   └── test: {question, answer}
├── PIQA (physical intuition)
│   ├── train: {goal, sol1, sol2, label}
│   └── ...
├── SIQA (social intelligence)
├── HellaSwag (prompt completion)
├── WinoGrande (coreference resolution)
├── ARC-Easy (multiple choice)
├── ARC-Challenge (multiple choice)
└── OBQA (open book QA)
```

### Training Pipeline Flow:

```
1. LOAD SEPARATE DATASETS
   BoolQ_ds = load_dataset('google/boolq')
   PIQA_ds = load_dataset('ybisk/piqa')
   SIQA_ds = load_dataset('allenai/social_i_qa')
   ... (8 total)
   ↓

2. APPLY DATASET-SPECIFIC FORMATTERS
   ┌─────────────────────┬──────────────────┐
   │ BoolQ               │ PIQA             │
   ├─────────────────────┼──────────────────┤
   │ _get_BoolQ_         │ _get_PIQA_       │
   │ instructions()      │ instructions()   │
   │                     │                  │
   │ "Answer True/False" │ "Choose         │
   │                     │  Solution1 or 2" │
   └─────────────────────┴──────────────────┘
   ↓

3. APPLY CHAT TEMPLATE (Per-dataset formatter)
   BoolQ:
   [system]: "Answer True/False"
   [user]: "Question: Is X true?"
   [assistant]: "The answer is True"
   
   PIQA:
   [system]: "Choose solution"
   [user]: "Goal: X\nSol1: Y\nSol2: Z"
   [assistant]: "The answer is Solution1"
   ↓

4. ADD TASK METADATA
   dataset['train'] = dataset['train'].add_column(
       'task', 
       ['BoolQ'] * len(dataset['train'])
   )
   ↓

5. PROCESS ALL DATASETS
   for each dataset:
       map(formatter, dataset, num_proc=8)
   ↓

6. SPLIT TRAIN/VAL/TEST
   train_portion = 90% of each dataset.train
   val_portion = 10% of each dataset.train
   test_portion = all of each dataset.validation
   ↓

7. CONCATENATE ALL DATASETS
   Combined_train = concat(BoolQ_train, PIQA_train, ..., OBQA_train)
   Combined_val = concat(BoolQ_val, PIQA_val, ..., OBQA_val)
   Combined_test = concat(BoolQ_test, PIQA_test, ..., OBQA_test)
   ↓

8. TOKENIZE & FORMAT FOR TRAINING
   (Similar to your approach, but text-based not pre-tokenized)
   ↓

9. BATCH & COLLATE
   Batch size: 2 examples
   (ALL from potentially different tasks)
   ↓

10. MODEL FORWARD PASS + TASK TRACKING
    model(batch) → logits
    Track metrics per task separately
```

### Data Format After Loading:
```python
{
    'text': "full conversation with answer",
    'text_wa_answer': "conversation without answer",
    'correct_answer': "ground truth",
    'task': 'PIQA'  # ← Dataset source
}
```

### Evaluation Pipeline:
```
Per-Task Evaluation:
├── BoolQ metrics (accuracy on boolean questions)
├── PIQA metrics (accuracy on physical intuition)
├── SIQA metrics (accuracy on social intelligence)
├── HellaSwag metrics
├── WinoGrande metrics
├── ARC-Easy metrics
├── ARC-Challenge metrics
└── OBQA metrics

Analysis:
- Identify which tasks the model performs well on
- Early stopping based on specific task metrics
- Dataset-specific loss tracking
- Cross-task generalization analysis
```

---

## 🔑 **KEY DIFFERENCES SUMMARY**

| Aspect | Your Method | RiemanianFinetune |
|--------|------------|-------------------|
| **Data Source** | 1 pre-combined file | 8 separate HF datasets |
| **Format** | Pre-tokenized JSON | Text with chat template |
| **Sampling** | Source-aware weighted | Simple concatenation |
| **Task ID** | `source` (math dataset type) | `task` (reasoning type) |
| **Per-example metadata** | weight, completion_length, index | None (handled via task column) |
| **Formatter Strategy** | Generic prompt template | 8 dataset-specific formatters |
| **Training** | Balanced source sampling | Mixed task batches |
| **Evaluation** | Global metrics only | Per-task breakdown |
| **Batch Composition** | Same source preference | All tasks mixed |
| **Flexibility** | Add new data = re-prepare entire JSONL | Add new data = add new dataset loader |

---

## 📈 **TRAINING DYNAMICS COMPARISON**

### Your Method (MathInstruct):
```
Batch Example:
├── Example 1: From MATH (source=0)
└── Example 2: From TheoremQA (source=1)

Loss: L_batch = (L_math + L_theoremqa) / 2

Per-source effectiveness:
- MATH: High-quality, long explanations → high weight
- TheoremQA: Small dataset → used fully (100%)
- Implicit assumption: All math tasks benefit mutually
```

### RiemanianFinetune Method:
```
Batch Example:
├── Example 1: From PIQA (task='PIQA')
└── Example 2: From SIQA (task='SIQA')

Loss: L_batch = (L_piqa + L_siqa) / 2

Per-task effectiveness:
- PIQA: Physical scene understanding → measure separately
- SIQA: Social reasoning → measure separately
- Explicit goal: Multi-task learning across different reasoning types
```

---

## 💾 **STORAGE & EFFICIENCY**

### Your Method:
```
File: MathInstruct.jsonl (on disk)
├── Single file, easy to distribute
├── Pre-computed: sampling already done
├── No preprocessing at training time = FAST
└── Fixed: Cannot easily add/remove sources

In Memory During Training:
- Tokenized: ~50GB (FP32 embeddings)
- Metadata kept: sources, weights, indices
- Batch preparation: Merge metadata + pad sequences
```

### RiemanianFinetune Method:
```
Saved Dataset (on disk as shards)
├── Multiple split directories
├── Pre-computed: formatters already applied
├── No preprocessing at training time = FAST (but slower than pre-tokenized)
└── Flexible: Each dataset can be updated independently

In Memory During Training:
- Text-based: ~100GB (larger than tokenized)
- Metadata kept: task name + answer
- Batch preparation: Tokenize on-the-fly (if not cached)
```

---

## 🎯 **WHICH IS BETTER FOR WHAT?**

### Use Your Method When:
✓ Single domain (math, code, etc.)
✓ Want weighted importance sampling
✓ Need maximum speed/efficiency
✓ Dataset already curated + combined
✓ Per-source analysis less important

Examples: Math fine-tuning, code generation, domain-specific tasks

### Use RiemanianFinetune Method When:
✓ Multiple diverse domains
✓ Want task-specific evaluation
✓ Need flexibility to add/modify datasets
✓ Cross-task generalization important
✓ Per-task performance tracking critical

Examples: General-purpose LLM, multi-task learning, commonsense reasoning
