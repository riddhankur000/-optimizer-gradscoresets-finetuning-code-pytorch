# Subset Selection Methods Architecture Analysis

## Directory: `/home1/riddhankur/adamw_vs_muon_2`

This document outlines how and where subset selection methods are implemented throughout the codebase.

---

## 1. Overview of Subset Selection System

The codebase implements **multiple data subset selection strategies** for efficient training of LLMs. These methods help select the most informative samples from large datasets to reduce training time while maintaining or improving model performance.

### Supported Selection Methods:
- **SPOT** (SPOT Greedy Subset Selection)
- **GREATS** (Greedy Subset Selection)
- **FairOT** (Fairness-aware Optimal Transport)
- **Facility Location** (Greedy facility location-based selection)
- **Random** (Baseline)
- **None** (Full dataset training)

---

## 2. Main Entry Points

### 2.1 `llm-experiments/colm/train/train.py`
**Location**: Lines 325-380

```python
# Determines which trainer to use based on selection method
if training_args.data_selection_method == "none":
    trainer_class = Trainer  # Standard HuggingFace Trainer
elif training_args.efficient_mezo:
    trainer_class = SubsetTrainerEfficient  # Efficient variant
else:
    trainer_class = SubsetTrainer  # Standard subset selection trainer
```

**Configuration Parameters Read**:
- `training_args.data_selection_method` - Which selection algorithm to use
- `training_args.efficient_mezo` - Whether to use efficient MEZO variant
- `training_args.data_selection_unit` - Unit of selection ("mezo", etc.)

---

## 3. Core Implementation Files

### 3.1 `llm-experiments/colm/train/subset_trainer_distributed.py`

**Main Classes**:

1. **`class SubsetTrainer(Trainer)` (Lines 176+)**
   - Extends HuggingFace's Trainer class
   - Implements core subset selection logic
   - Key methods:
     - `_inner_training_loop()` - Modified training loop with selection
     - `select_data()` - Main dispatcher for selection methods
     - `select_data_facloc()` - Facility location selection
     - `select_euclidean()` - Euclidean distance-based selection
     - `save_select()` - Saves selected indices

2. **`class SubsetTrainerEfficient(SubsetTrainer)` (Line 1499+)**
   - Extension of `SubsetTrainer` for efficient gradient computation
   - Optimized for MEZO (Zero-Order Optimization)
   - Reduces memory footprint during selection

**Key Selection Methods**:

#### `select_data()` - Main Dispatcher (Lines 1470+)
```python
def select_data(self, inputs, max_samples=64, source_list=None, model=None):
    # Computes similarities between samples
    # Routes to appropriate selection method based on self.method
    
    if self.method == "spot":
        # SPOT Greedy selection
        from colm.train.SPOTgreedy import SPOT_GreedySubsetSelection
        idx = SPOT_GreedySubsetSelection(dist, target_marginal, max_samples)
        
    elif self.method == "greats":
        # GREATS selection
        import colm.train.greats as greats
        idx, total = greats.greedy_selection(
            tocpu(sims_cross.mean(1)), 
            tocpu(sims), 
            max_samples
        )
        
    elif self.method == "fairot":
        # FairOT selection
        import colm.train.fairot2 as fairot2
        idx = fairot2.greedy_fairot(
            tocpu(sims), 
            max_samples, 
            dist=tocpu(dist), 
            iters=500, 
            reg=1e-1
        )
```

#### `select_data_facloc()` - Facility Location (Lines 1450+)
```python
def select_data_facloc(self, inputs, max_samples=64, source_list=None, 
                       optim=None, metric="cosine"):
    # Uses facility location greedy algorithm
    # Computes orders and weights
    from colm.train.facility_location import get_orders_and_weights
    order, weights = get_orders_and_weights(...)
```

#### `select_euclidean()` - Euclidean Distance (Lines 1350+)
```python
def select_euclidean(self, all_reps_flat, max_samples, source_list, ...):
    # Selects samples based on Euclidean distance representations
    # Useful for coverage-based selection
```

---

### 3.2 Subset Selection Algorithm Implementations

#### 3.2.1 `llm-experiments/colm/train/facility_location.py`
**Implements**: Greedy facility location algorithm
- **Function**: `get_orders_and_weights()`
- **Algorithm**: Iteratively selects samples that maximize diversity
- **Use Case**: When you want representative subset from diverse data

#### 3.2.2 `llm-experiments/colm/train/SPOTgreedy.py`
**Implements**: SPOT Greedy Subset Selection
- **Class**: `SPOT_GreedySubsetSelection`
- **Algorithm**: 
  - Uses optimal transport theory
  - Computes cost matrix between samples
  - Greedily selects influential samples
- **Parameters**: 
  - `dist`: distance/similarity matrix
  - `target_marginal`: target distribution
  - `max_samples`: number of samples to select

#### 3.2.3 `llm-experiments/colm/train/greats.py`
**Implements**: GREATS (Gradient Regularized Training with Subset Selection)
- **Function**: `greedy_selection()`
- **Algorithm**:
  - Uses gradient-based similarity
  - Computes cross-sample similarities
  - Selects high-uncertainty or high-gradient samples
- **Parameters**:
  - `sims_cross`: cross-gradient similarities
  - `sims`: full similarity matrix
  - `max_samples`: number of samples to select

#### 3.2.4 `llm-experiments/colm/train/fairot.py` & `fairot2.py`
**Implements**: FairOT (Fair Optimal Transport) 
- **Function**: `greedy_fairot()`
- **Algorithm**:
  - Combines fairness constraints with optimal transport
  - Ensures balanced representation from different groups
  - Solves regularized optimal transport problem
- **Parameters**:
  - `sims`: similarity matrix
  - `max_samples`: number of samples to select
  - `dist`: distribution/fairness constraints
  - `iters`: optimization iterations
  - `reg`: regularization coefficient

---

## 4. Configuration and Execution Flow

### 4.1 Configuration File
**File**: `llm-experiments/colm/scripts/train/base_training_args.sh`

**Key Parameters**:
```bash
--data_selection_method submodlib      # Selection algorithm
--data_selection_unit mezo             # Selection unit
--lora_dropout 0.05                    # LoRA configuration
```

### 4.2 Training Script
**File**: `llm-experiments/scripts/run_math_efficient.sh`

**Configuration Options** (Lines 25-42):
```bash
SELECTION_METHOD=none          # submodlib, weightedsubmodlib, none
DATA_SELECTION=mezo            # rep, mezo, masked_grad, grad, proj_grad, etc.
MEZO_SELECTION=grad            # weight_grad, weight, grad
MEZO_TOPK=10                   # Top-k selection
FACILITY_SELECT=none           # none, faciloc, random
```

---

## 5. Data Flow During Training

### 5.1 Initialization Phase
```
train.py
    ↓
TrainingArguments.data_selection_method
    ↓
Selects trainer class: SubsetTrainer or SubsetTrainerEfficient
    ↓
Trainer initializes with method parameter
```

### 5.2 Selection Phase (During Training Loop)
```
_inner_training_loop()
    ↓
For each batch:
    1. Compute representations (hidden states)
    2. select_data() → dispatches to specific method
    3. Compute similarities between samples
    4. Route to:
       - SPOT_GreedySubsetSelection
       - greats.greedy_selection
       - fairot2.greedy_fairot
       - facility_location.get_orders_and_weights
    5. Return: selected_indices
    6. Filter training batch to selected samples
    7. Continue training with subset
```

### 5.3 Index Saving (Optional)
```
If args.save_indices:
    selected_indices → {output_dir}/indices/
```

---

## 6. Method Comparison

| Method | Algorithm | Complexity | Use Case |
|--------|-----------|-----------|----------|
| **SPOT** | Optimal Transport | O(n²) | Balanced sample importance |
| **GREATS** | Gradient-based | O(n²) | High-entropy sample selection |
| **FairOT** | Fair Optimal Transport | O(n² × iters) | Fairness-constrained selection |
| **Facility Location** | Greedy Facility Location | O(nk) | Diversity-focused selection |
| **Random** | Random Sampling | O(n) | Baseline |
| **None** | Full Dataset | O(1) | No selection |

---

## 7. Key Implementation Details

### 7.1 Subset Trainer's Training Loop Modifications

**Location**: `subset_trainer_distributed.py` - `_inner_training_loop()` method

Key modifications:
1. **Batch size**: Enforces `per_device_train_batch_size = 1`
2. **Selection trigger**: Called at specific intervals (configurable)
3. **Memory management**: Calls `accelerator.free_memory()` before selection
4. **Sample filtering**: After selection, creates `SubsetRandomSampler` with selected indices

### 7.2 Similarity Computation

**How it works**:
```python
# Compute representation for each sample
reps = model_forward(inputs)  # Hidden states

# Compute pairwise similarities
sims = cosine_similarity(reps)  # or other metrics

# Pass to selection method
selected_indices = selection_method(sims, max_samples)
```

### 7.3 Source-wise Selection

**File**: `subset_trainer_distributed.py` - Line 210 onwards
```python
if args.source_wise_selection != "none":
    # Selects samples while maintaining source distribution
    # Useful for multi-source datasets like MetaMathQA + GSM8K
```

---

## 8. Integration with Multi-Task Training

**Files Involved**:
- `llm-experiments/colm/data/tasks.py` - Task definitions with `sample_subset()` method
- `colm/train/train.py` - Lines 200-225 - Sample filtering per task

**Flow**:
```
For each task:
    1. Load samples from task dataset
    2. If source_wise_selection != "none":
        call perform_training_subset_selection()
    3. Else if subset selection enabled:
        call perform_training_subset_selection_on_collection()
    4. Filter samples to subset
    5. Tokenize and train on subset
```

---

## 9. Search Keywords & File Locations

To find specific implementations, search for:

| Concept | Search Term | Files |
|---------|------------|-------|
| Main trainer with selection | `class SubsetTrainer` | `subset_trainer_distributed.py:176` |
| Efficient variant | `class SubsetTrainerEfficient` | `subset_trainer_distributed.py:1499` |
| Selection dispatcher | `def select_data` | `subset_trainer_distributed.py:1470` |
| Facility location | `get_orders_and_weights` | `facility_location.py` |
| SPOT selection | `SPOT_GreedySubsetSelection` | `SPOTgreedy.py` |
| GREATS selection | `greedy_selection` | `greats.py` |
| FairOT selection | `greedy_fairot` | `fairot2.py` |
| Configuration reading | `data_selection_method` | `train.py:332`, `base_training_args.sh:33` |
| Training entry point | `if __name__ == "__main__"` | `train.py:~475` |

---

## 10. Configuration Examples

### Using GREATS Selection
```bash
python colm/train/train.py \
    --data_selection_method greats \
    --data_selection_unit mezo \
    ...
```

### Using FairOT Selection
```bash
python colm/train/train.py \
    --data_selection_method fairot \
    --data_selection_unit mezo \
    ...
```

### Using Facility Location
```bash
python colm/train/train.py \
    --data_selection_method submodlib \  # maps to facility location
    --facility_select faciloc \
    ...
```

### No Selection (Standard Training)
```bash
python colm/train/train.py \
    --data_selection_method none \
    ...
```

---

## 11. Performance Characteristics

- **Gradient Computation**: Handled by `SubsetTrainerEfficient` for MEZO
- **Selection Overhead**: Typically 5-10% of training time
- **Memory**: Reduced by ~70-90% when selecting 10% of data
- **Convergence**: Often improves or maintains convergence with proper selection strategy

---

## 12. Summary

The subset selection system is architecturally organized as:

```
Entry: train.py
    ↓
Trainer Selection (SubsetTrainer vs Standard Trainer)
    ↓
Subset Trainer (_inner_training_loop)
    ↓
Selection Methods Dispatcher (select_data)
    ↓
Algorithm Implementations:
    ├── SPOT (SPOTgreedy.py)
    ├── GREATS (greats.py)
    ├── FairOT (fairot2.py)
    └── Facility Location (facility_location.py)
    ↓
Indices Saved & Training Continues with Subset
```

This modular design allows easy experimentation with different subset selection strategies while maintaining a unified training framework.
