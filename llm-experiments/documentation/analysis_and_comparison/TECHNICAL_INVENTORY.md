# CoLM: Technical Inventory & Codebase Analysis

**Project**: Mini-batch Coresets for Memory-efficient Language Model Training  
**Organization**: ICLR 2025 (Official Implementation)  
**Base Version**: Python 3.10 | PyTorch 2.2.1 | Transformers 4.43.2

---

## Executive Summary

This codebase implements **CoLM (Coreset for Language Models)**, a memory-efficient training method that selects representative mini-batches from large, imbalanced datasets using facility location algorithms with zeroth-order gradient estimation (MeZO). The project integrates multiple data selection algorithms (facility location, GREATS, FairOT, SPOT) and supports both efficient and full training modes.

---

## 1. Directory Structure & File Organization

```
llm-experiments/
├── colm/                          # [416 KB] Core training implementation
│   ├── train/                     # [7,322 lines total] Training algorithms & utilities
│   │   ├── train.py               # [405 lines] Main training entry point (Phi models)
│   │   ├── train_multitask.py     # [586 lines] Multi-task training (MetaMathQA + GSM8K)
│   │   ├── subset_trainer_distributed.py  # [2,392 lines] Core trainer with data selection
│   │   ├── huggingface_trainer.py # [838 lines] Modified HF Trainer with custom metrics
│   │   ├── facility_location.py   # [127 lines] Facility location greedy selection
│   │   ├── greats.py              # [34 lines] GREATS algorithm (validation-based selection)
│   │   ├── fairot.py              # [352 lines] FairOT algorithm (optimal transport v1)
│   │   ├── fairot2.py             # [484 lines] FairOT algorithm v2 (improved OT)
│   │   ├── SPOTgreedy.py           # [81 lines] SPOT algorithm (OT prototype selection)
│   │   ├── sinkhorn.py            # [197 lines] Partial optimal transport solver
│   │   ├── custom_phi.py          # [267 lines] Phi model decomposition for layerwise forward
│   │   ├── optimizer_factory.py   # [372 lines] Custom optimizer creation (Adam/Muon/SGD)
│   │   ├── config_loader.py       # [265 lines] YAML config parsing and validation
│   │   ├── training_arguments.py  # [320 lines] Extended training args (CoLM-specific)
│   │   ├── data_arguments.py      # [80 lines] Data loading arguments
│   │   ├── model_arguments.py     # [94 lines] Model configuration arguments
│   │   ├── utils.py               # [270 lines] Gradient computation, similarity metrics
│   │   ├── plot.py                # [116 lines] Visualization utilities
│   │   └── buffsub.py, buffhf.py  # Buffer utility modules
│   └── data/                      # [88 KB] Dataset utilities
│       ├── get_training_dataset.py # [150+ lines] Dataset loading with source awareness
│       ├── get_validation_dataset.py
│       ├── tasks.py               # [100+ lines] Task definitions (SST2, Copa, etc.)
│       ├── templates.py           # [150+ lines] Prompt templates for SuperGLUE
│       └── utils.py               # [50+ lines] Data utilities & encoding
│
├── data/                          # Data files (requires external download)
│   └── *.jsonl files (MathInstruct, SuperGLUE with metadata)
│
├── scripts/                       # [~12 KB] Training orchestration
│   ├── run_math_efficient.sh      # CoLM with efficient MeZO (GAS=8, ~0.1s selection overhead)
│   ├── run_math.sh                # CoLM standard training (GAS=64, FairOT/GREATS)
│   ├── run_superglue.sh           # SuperGLUE evaluation setup
│   └── (subdirectory scripts in ./colm/scripts/train/)
│
├── math_eval/                     # [37 MB] Math evaluation module
│   ├── run_open.py                # [11 KB] Evaluation runner with vLLM support
│   ├── eval_pretrained.sh         # Pretrained model evaluation
│   ├── eval_finetuned.sh          # Fine-tuned model evaluation (6 datasets × 4 GPUs)
│   ├── data_loader.py             # [7.3 KB] Batch dataset loader
│   ├── prompt_utils.py            # [38 KB] Prompt engineering & few-shot examples
│   ├── utils.py                   # [18 KB] Answer extraction & validation
│   └── dataset/                   # Dataset files for GSM8K, MATH, NUMGLUE, etc.
│
├── superglue_eval/                # [24 KB] SuperGLUE evaluation
│   ├── eval_superglue.py          # [100+ lines] SuperGLUE tasks (SST2, Copa, BoolQ, etc.)
│   ├── eval_pretrained.sh
│   └── eval_finetuned.sh
│
├── vllm/                          # [20 MB + 308 Python files] LLM inference engine
│   ├── vllm/                      # Core serving implementation
│   ├── examples/                  # API client examples
│   ├── benchmarks/                # Throughput & latency benchmarks
│   ├── csrc/                      # CUDA/HIP kernels (PUNICA, PagedAttention)
│   └── setup.py                   # Installation config
│
├── docs/                          # Documentation
├── config.yaml                    # [6.7 KB] Master configuration (Llama 3.1 8B example)
├── requirement.txt                # Exact dependencies (torch 2.2.1, transformers 4.43.2)
├── setup.py                       # Package installation
├── README.md                      # Quick overview
├── IMPLEMENTATION_GUIDE.md        # [16 KB] Features & algorithm details
├── QUICK_REFERENCE.md             # [15 KB] Commands & troubleshooting
└── README_ANALYSIS.md             # [16 KB] Analysis & experimental setups
```

---

## 2. New Integration: vLLM (LLM Serving Engine)

### What is vLLM?
**vLLM** is a high-performance LLM inference and serving library that powers CoLM's evaluation module.

### Integration Points
- **Math Evaluation**: `math_eval/run_open.py` uses vLLM's LLM class for fast inference
- **LoRA Support**: vLLM includes multi-LoRA support via `vllm.vllm.lora.request.LoRARequest`
- **Key Feature**: PagedAttention for memory-efficient KV cache management
- **Installation**: `VLLM_INSTALL_PUNICA_KERNELS=1 pip install -e vllm/`

### Key Modules in vLLM
```
vllm/
├── vllm/                    # Core inference engine
│   ├── engine/              # LLM inference orchestration
│   ├── model_executor/      # Model execution with paged attention
│   ├── worker/              # Distributed worker processes
│   ├── lora/                # LoRA adapter management
│   ├── utils/               # Utilities
│   └── sampling_params.py   # Decoding parameters
├── benchmarks/
│   ├── benchmark_throughput.py
│   ├── benchmark_latency.py
│   ├── benchmark_serving.py
│   ├── benchmark_prefix_caching.py
│   └── kernels/             # Kernel-specific benchmarks
├── examples/
│   ├── api_client.py        # OpenAI-compatible API client
│   └── fp8/                 # FP8 quantization examples
└── csrc/
    ├── punica/              # PUNICA kernels for LoRA
    └── (CUDA/HIP kernels)
```

### vLLM Usage in Evaluation
```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="microsoft/phi-2", max_model_len=2048)
sampling_params = SamplingParams(temperature=0.5, top_p=0.95)

# Generate with LoRA adapter
lora_request = LoRARequest("adapter", 1, "/path/to/lora/model")
outputs = llm.generate(input_strs, sampling_params, lora_request=lora_request)
```

---

## 3. Training Algorithms: From fairot.py to fairot2.py to SPOTgreedy.py

### 3.1 FairOT Algorithm Family

#### `fairot.py` (352 lines) - Initial Implementation
**Purpose**: Optimal transport-based data selection with fairness constraints

**Key Functions**:
- `greedy_fairot(S, k, reg=1e-2)`: Greedy selection via approximate gain
  - Uses `pot_partial_extended()` from Sinkhorn algorithm
  - Computes optimal alpha for marginal distributions
  - Selects k prototypes iteratively
  
- `optimal_alpha(S_a, b, reg, tol=1e-8, max_iter=100)`: KKT-based solution
  - Solves: maximize $\sum S_a \cdot α + λ·H(α)$ subject to $0 ≤ α ≤ b$, $\sum α = 1$
  - Interior/boundary partition determines soft-max scaling

**Limitation**: Single gain computation per iteration (slower on large batches)

#### `fairot2.py` (484 lines) - Improved Implementation  
**Purpose**: Vectorized optimal transport with efficiency improvements

**Key Improvements**:
- `greedy_fairot()` v2 with **vectorized gain computation**
  - Precomputes sorted indices `sorted_indices_all` and similarities `sorted_S_all`
  - Computes **all gains in parallel**: `np.array([exact_gain(...) for candidate in candidates])`
  - Uses `pot_partial_library()` for more stable OT computation
  
- `optimal_alpha_vectorized()`: Faster alpha computation
  - Uses cumsum for partition finding instead of loop
  - Complexity: O(n log n) vs O(n²) iteration
  
- Distance matrix support: `dist` parameter enables alternative metrics

**Usage Difference**:
```python
# fairot.py - slower
gains = []
for v in candidates - set(P):
    gain = approx_gain(P, gamma_P, v, S, k, reg)  # Recompute each time
    gains.append(gain)

# fairot2.py - faster (vectorized)
gains = np.array([
    exact_gain(P, gamma_P, v, S, sorted_S_candidates[i], ...)
    for i, v in enumerate(candidates)  # All in parallel
])
```

### 3.2 SPOT Algorithm (`SPOTgreedy.py`, 81 lines)

**Purpose**: Prototype selection for optimal transport

**Algorithm**: `SPOT_GreedySubsetSelection(C, target_marginal, m)`
```python
def SPOT_GreedySubsetSelection(C, target_marginal, m):
    # C: cost matrix [num_sources × num_targets]
    # target_marginal: row distribution across targets
    # m: number of prototypes to select
    
    for step in range(m):
        # Compute gain: improvement in coverage
        gain = gain_matrix @ target_marginal.t()
        # Select best candidate
        best_idx = argmax(gain[remaining_sources])
        selected_indices[step] = best_idx
        # Update min costs
        better_mask = (min_costs - C[chosen, :]) > 0
        min_costs[better_mask] = C[chosen, better_mask]
```

**Key Difference from FairOT**:
- FairOT: Solves partial OT problem iteratively (exact marginals)
- SPOT: Greedy matching on cost matrix (faster, approximate)

### 3.3 GREATS Algorithm (`greats.py`, 34 lines)

**Purpose**: Validation-based submodular selection

**Core Function**: `greedy_selection(scores, interaction_matrix, K)`
```python
def greedy_selection(scores, interaction_matrix, K):
    """
    scores: [train_bs × val_bs] - validation scores for each training example
    interaction_matrix: [train_bs × train_bs] - pairwise interactions
    K: cardinality constraint
    
    for each step:
        1. Select highest score: idx_max = argmax(scores)
        2. Update scores: scores -= interaction_matrix[idx_max, :]
        3. Mark selected: scores[idx_max] = -inf
    """
```

**Relationship to FairOT & SPOT**:
| Algorithm | Input | Selection Criterion | Speed |
|-----------|-------|-------------------|-------|
| **FairOT** | Similarity matrix + labels | Facility location + OT fairness | Slow (iterative OT solve) |
| **SPOT** | Cost matrix + target distribution | Greedy gain on transport cost | Medium |
| **GREATS** | Validation scores + interactions | Submodular greedy with coverage | Fast |

---

## 4. Data Selection: Facility Location & Source-Wise Handling

### 4.1 Facility Location (`facility_location.py`, 127 lines)

**Purpose**: Greedy coreset selection with support for imbalanced sources

**Algorithm**: `get_orders_and_weights(B, X, metric, y=None, strategy="proportional")`

```
Input:
  - X: features [N × d]
  - B: budget (coreset size)
  - y: source labels for N examples (e.g., [0,0,...,1,1,...,2,2,...])
  - strategy: "none", "proportional", or "balanced"

Process:
  1. Group examples by source/class
  2. Allocate budget per source based on strategy:
     - "none": Select B from all examples uniformly
     - "proportional": $B_i = floor(N_i / N × B)$, adjust to sum to B
     - "balanced": Ensure min≤B_i≤max calculated from proportions
  3. For each source: Run facility location on that subset
  4. Weight based on cluster size:
     - Selected examples: weight = 1
     - Unselected examples: weight = similarity to nearest selected
```

**Key Features**:
- **Small Source Preservation**: If source has <B_i examples, include ALL
- **Distance-Aware Weighting**: Unselected samples weighted by similarity to selected medoids
- **Optimizer Support**: Can use `greaty_fairot()` or standard `FacilityLocationFunction`

### 4.2 Source-Wise Selection in Training

**Location**: `subset_trainer_distributed.py` line 1289

**Concept**: Apply selection **independently per source**, then combine results

```
Training Data:
  ├─ Source 0 (MathQA): 250K examples ─→ Select 32 (0.5% of batch)
  ├─ Source 1 (MATH): 11K examples ─→ Select 32
  ├─ Source 2 (GSM8K): 8K examples ─→ SELECT ALL (small source)
  └─ ...
  └─ Coreset: 64 selected + 32 small-source = 96 total

Benefits:
  - Handles 300:1 imbalance in MathInstruct (Theorem 4.1 in paper)
  - Prevents small sources from being drowned out
  - Maintains source-level diversity
```

---

## 5. Zeroth-Order Gradient Estimation (MeZO)

### 5.1 Core Concept

**Location**: `subset_trainer_distributed.py` lines 1420-1430 + `utils.py`

**Problem**: Full gradient requires 2 forward passes (inefficient for large models)

**Solution**: Last-layer zeroth-order (ZO) gradient approximation
```
Standard gradient:
  ∇L = ∂L/∂θ

MeZO approximation (last layer only):
  1. Sample random perturbation: z ~ N(0,I) of dimension d
  2. Forward passes:
     - f(θ + εz): Get logits with perturbed last layer
     - f(θ - εz): Get logits with negated perturbation
     - Compute loss L+ and L-
  3. Gradient estimate: ∇̂L ≈ (L+ - L-) / (2ε) × z
  4. Sparse projection: Keep top-k dimensions (0.7% sparsity → 2560 dims)
```

### 5.2 Efficient Implementation

**Configuration Options**:
```yaml
efficient_mezo: True           # Use efficient variant
zo_dim: 2560                   # Final sparse dimension (0.7% of full)
mezo_eps: 1e-3                 # Perturbation scale
mezo_topk: "largest"           # Selection: random/largest/smallest/sampling
mezo_selection: "grad"         # Use gradient for topk selection
mezo_optim: "adam"             # Normalize by Adam moments (untested) or "sgd"
mezo_transform: "none"         # Apply normalization or clipping
```

**Memory Savings**:
- Full gradient: ~2× forward passes + full storage
- MeZO with sparsity: ~1× forward pass + sparse storage (0.7% density)
- Training time overhead: <0.1s for selection + synchronization

### 5.3 Adam-Aware Normalization

**Location**: `facility_location.py` lines 98-130

**Concept**: Normalize selected gradients by exponential moving average (EMA) of gradient norms

```python
if mezo_optim == "adam":
    # Compute EMA of gradient norm per neuron
    m_hat = ema_of_norms  # Over training history
    # Scale gradient by 1/sqrt(m_hat + eps) to normalize
    normalized_grad = grad / sqrt(m_hat + 1e-8)
```

**Purpose**: Compensate for optimizer-level differences in gradient magnitudes

---

## 6. Custom Model Architectures

### 6.1 Phi Model Decomposition (`custom_phi.py`, 267 lines)

**Purpose**: Enable last-layer MeZO gradient computation

**Implementation**: `DecomposedPhiCausalLM`

```python
class DecomposedPhiCausalLM:
    def forward_till_penultimate(self, input_ids, ...):
        """
        Run forward through all layers EXCEPT the last one.
        Returns: {
            'hidden_states': [batch, seq_len, d_model],
            'attention_mask': [...],
            'past_key_values': [...],  # For caching
            'cache_position': [...]
        }
        """
        # Process embeddings + first (num_layers - 1) layers
        for decoder_layer in self.layers[:-1]:
            hidden_states = decoder_layer(hidden_states, ...)
        return hidden_states
    
    def forward_last_layer(self, hidden_states):
        """Apply only the last transformer layer + lm_head"""
        final_hidden_states = self.layers[-1](hidden_states)
        logits = self.lm_head(final_hidden_states)
        return logits
```

**Benefits for MeZO**:
- Can optimize only `(d_model × vocab_size)` parameters in last layer
- Gradients for full model approximated via ZO on this subset
- Reduces computation & memory by ~20-30%

---

## 7. Training Pipelines

### 7.1 Main Training Scripts

#### `run_math_efficient.sh` - **Efficient CoLM**
```bash
GAS=8                           # Gradient accumulation (small per-device batch)
BATCH_RATIO=0.5                 # Select 50% of accumulated batch
SELECTION_METHOD=greats         # Use GREATS algorithm
DATA_SELECTION=mezo             # Zeroth-order gradients
EFF_MEZO=True                   # Memory-efficient MeZO
ZO_DIM=2560                     # Sparse dimension (0.7%)
MEZO_TOPK=largest               # Select largest magnitude gradients
MAX_STEPS=1024                  # 1K update steps
```

**Effective Batch Size**: 4 (device) × 8 (GAS) = 32 per step  
**Coreset Size**: 32 × 0.5 = 16 selected samples  
**Training Time**: ~ideal forward (B=32) + backward (B=16) + <0.1s selection

#### `run_math.sh` - **Standard Training**
```bash
GAS=64                          # Higher accumulation
BATCH_RATIO=0.5                 # Still select 50%
SELECTION_METHOD=fairot_multisource  # FairOT with source awareness
DATA_SELECTION=masked_grad      # Use masked/projected gradients
EFF_MEZO=False                  # Full MeZO computation
MEZO_TRANSFORM=self_normalize   # Normalize by max gradient
MAX_STEPS=2048                  # 2K update steps
```

**Effective Batch Size**: 1 (device) × 64 (GAS) = 64 per step  
**Coreset Size**: 64 × 0.5 = 32  
**Trade-off**: Slower training, potentially better convergence

#### `run_superglue.sh` - **Evaluation Benchmark**
```bash
DEVICE_BS=1                     # Single example per device
GAS=32                          # 32-step accumulation → B=32 equiv
MAX_STEPS=1024                  # Shorter training for classification
SELECTION_METHOD=submodlib      # Facility location
```

### 7.2 Multi-Task Training (`train_multitask.py`, 586 lines)

**Features**:
- Loads pre-created combined dataset (MetaMathQA + GSM8K)
- Per-task metrics logging via `MonitoringCallback`
- Comprehensive GPU/memory monitoring with `GPUtil`
- Perplexity & gradient norm tracking
- YAML configuration support

**Config Example** (in `config.yaml`):
```yaml
dataset_config:
  dataset_names:
    - "MetaMathQA"
    - "GSM8K"
  train_split: "train"
  eval_split: "validation"
  train_ratio: 0.85

training_config:
  learning_rate: 0.0002
  max_steps: 4096
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  # Effective batch = 8 × 4 × num_gpus
```

---

## 8. Data Management

### 8.1 Dataset Loading (`get_training_dataset.py`, 150+ lines)

**Main Function**: `get_training_dataset(train_files, tokenizer, max_seq_length, ...)`

**Supported Input Formats**:
1. **JSONL files** (MathInstruct, SuperGLUE)
   - Format: `{"instruction": "...", "output": "...", "source": "source_name"}`
2. **HF Datasets** (e.g., "load-superglue-sst2")
   - Loaded dynamically

**Subset Selection Strategies**:
- `"random"`: Random sampling
- `"balanced_longest_selection"`: Longest examples per source, balanced
- `"longest_sourcewise_selection"`: Source-wise longest sampling
- `"longest_selection"`: Global longest examples
- `"use_small_sources"` (default): Use ALL small sources, sample large ones

**Source Handling**:
```python
# MathInstruct example (14 sources)
source_groups = {
    "GSM8K": [idx1, idx2, ...],          # ~8K
    "MATH": [idx3, idx4, ...],            # ~11K
    "MetaMathQA": [idx5, idx6, ...],      # ~241K
    ...
}

# With proportional strategy: sample proportionally from each source
```

### 8.2 Validation Dataset (`get_validation_dataset.py`)

**Variants**:
- **Validation**: For early stopping & metrics
- **Test**: Final evaluation
- **Per-source**: Optional source-wise validation

### 8.3 Dataset-Specific Templates (`templates.py`, 150+ lines)

**Supported Tasks** (SuperGLUE):
```python
class SST2Template(Template):
    # Input: "sentence"
    # Output: "terrible" or "great"
    # Prompt: "{sentence} It was"
    # Verbalize: "{sentence} It was {great/terrible}"

class CopaTemplate(Template):
    # Causal/effect reasoning
    # Prompt: "{premise} {so/because}"

class BoolQTemplate(Template):
    # Yes/No questions
    # Prompt: "{passage} {question}?"

class MultiRCTemplate(Template):
    # Multiple choice reading comprehension
```

---

## 9. Evaluation Modules

### 9.1 Math Evaluation (`math_eval/`, 37 MB)

**Supported Datasets**:
```
Primary:        Secondary:
├─ GSM8K        ├─ SVAMP
├─ MATH         ├─ Deepmind (hard)
└─ NUMGLUE      └─ SimulEq (synthetic)
```

**Evaluation Script** (`run_open.py`, 11 KB):

```python
def run_question_answer(
    args,  # model, dataset, etc.
    questions: list,
    groundtruths: list,
    lora_path=None
):
    # 1. Load few-shot examples
    used_examples = get_examples(args.dataset, args.shots, args.stem_flan_type)
    
    # 2. Generate answers
    if args.use_vllm:
        # Batch inference via vLLM (fast)
        outputs = llm.generate(input_strs, sampling_params, lora_request=lora_request)
    else:
        # Single-pass inference (slow but memory-efficient)
        outputs = utils.get_answer(examples, questions, model, tokenizer)
    
    # 3. Extract & validate answers
    for output, question, groundtruth in zip(outputs, questions, groundtruths):
        if 'print(' in output:
            # Executable code detected → execute
            tmp = execute_with_timeout(output)
        else:
            # Extract final answer
            answer = answer_clean(args.dataset, ('####', 'The answer is'), output)
```

**Command Line Interface**:
```bash
python math_eval/run_open.py \
    --model /path/to/model \
    --dataset gsm8k \
    --shots 0 \
    --stem_flan_type "pot_prompt" \
    --batch_size 8 \
    --model_max_length 2048 \
    --use_vllm \
    --enable_lora \
    --dtype bfloat16
```

**Prompt Formats** (`prompt_utils.py`, 38 KB):
- **Chain-of-Thought (CoT)**: "Let's think step by step..."
- **Program-of-Thought (PoT)**: Generates executable Python code
- **Few-shot examples**: Task-specific demonstrations

### 9.2 SuperGLUE Evaluation (`superglue_eval/`, 24 KB)

**Supported Tasks**:
```
SST2      (2-class sentiment)
Copa      (Causal reasoning)
BoolQ     (Yes/No QA)
MultiRC   (Multiple choice reading)
CB        (Entailment)
WIC       (Word sense)
WSC       (Coreference)
ReCoRD    (Reading comprehension)
RTE       (Text entailment)
SQuAD     (Span extraction)
DROP      (Discrete reasoning)
```

**Evaluation Method** (`eval_superglue.py`):
```python
def forward(model, tokenizer, input_ids, option_len=None, generation=False):
    """
    Compute log-likelihood of options.
    
    For classification (BoolQ, SST2):
        1. Unroll all options: "{context} {option}"
        2. Compute log-prob of each
        3. argmax → predicted class
    
    For generation (SQuAD, DROP):
        1. Generate text until eos_token
        2. Compare with reference answer
    """
```

---

## 10. Integration Points & Data Flow

### 10.1 Training Pipeline Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      DATASET LOADING                         │
│ load_raw_dataset(train_files) → raw_datasets (JSONL, HF)    │
│ SupervisedDataset(raw_datasets) → tokenized examples        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING BATCH LOOP                       │
│ for step in num_steps:                                      │
│   ├─ Load batch B (128 examples)                            │
│   ├─ Forward pass: compute gradients g (last layer only)    │
│   └─ Sparse projection: keep top 0.7% → g_sparse           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│              DATA SELECTION (CoLM Core)                      │
│ 1. Compute similarity: S = cosine(g_sparse, g_sparse)        │
│ 2. Group by source:                                         │
│    ├─ Small sources (e.g., GSM8K): include ALL             │
│    └─ Large sources: run facility location                  │
│ 3. Greedy selection: select K medoids (32 from 128)        │
│ 4. Return: selected_indices, weights                        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│                  FORWARD & BACKWARD PASS                     │
│ batch_selected = batch[selected_indices]                     │
│ loss = model(batch_selected)                                 │
│ loss.backward()                                              │
│ optimizer.step()   ← Update weights                          │
└──────────────────────────────────────────────────────────────┘
```

### 10.2 Configuration Hierarchy

```
config.yaml (main config)
    ├─ active_profiles:
    │   ├─ optimizer: "adamw"
    │   └─ gpu: "gpu_multi"
    │
    ├─ model_config:
    │   ├─ model_id: "meta-llama/Llama-3.1-8B"
    │   ├─ torch_dtype: "bfloat16"
    │   └─ device_map: "auto"
    │
    ├─ lora_config:
    │   ├─ enabled: true
    │   ├─ lora_rank: 16
    │   ├─ target_modules: [q_proj, v_proj, ...]
    │   └─ lora_alpha: 512
    │
    ├─ training_config:
    │   ├─ learning_rate: 0.0002
    │   ├─ max_steps: 4096
    │   ├─ per_device_train_batch_size: 8
    │   └─ optim: "adamw_torch"
    │
    └─ dataset_config:
        ├─ dataset_path: "./colm_math_combined_dataset"
        ├─ dataset_names: [MetaMathQA, GSM8K]
        └─ train_ratio: 0.85
```

---

## 11. Configuration Options Available

### 11.1 Core CoLM Parameters

| Parameter | Type | Default | Options | Purpose |
|-----------|------|---------|---------|---------|
| `data_selection_method` | str | "none" | submodlib, greats, fairot_multisource, weightedsubmodlib, none | Selection algorithm |
| `data_selection_unit` | str | "rep" | mezo, rep, masked_grad, grad, proj_grad, mezo_rep, completion_length | Gradient type |
| `efficient_mezo` | bool | False | True/False | Use efficient sparse ZO |
| `small_batch_ratio` | float | 1.0 | 0.5, 0.25, etc. | Fraction of batch to select |
| `zo_dim` | int | 2560 | 512, 1024, 2560 | Sparse dimension (d/2560 ≈ 0.7%) |
| `mezo_eps` | float | 1e-3 | 1e-3 to 1e-1 | ZO perturbation scale |
| `mezo_topk` | str | "largest" | random, largest, smallest, sampling | Top-k selection method |
| `mezo_selection` | str | "grad" | weight_grad, weight, grad | Gradient selection criterion |
| `mezo_optim` | str | "sgd" | adam, sgd, muon | Optimizer type |
| `mezo_transform` | str | "none" | none, self_normalize, normalize, clip_full, clip_last | Gradient normalization |

### 11.2 Source & LoRA Parameters

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `source_wise_selection` | str | "proportional" | How to balance sources: none/proportional/balanced |
| `keep_sources` | str | "" | Force keep specific sources (e.g., "0_1_3") |
| `lora` | bool | True | Enable LoRA finetuning |
| `lora_r` | int | 128 | LoRA rank |
| `lora_alpha` | int | 512 | LoRA scaling factor |
| `last_layers` | str | "v_proj" | Which layer for ZO: v_proj, output, full_last |

### 11.3 Training Parameters

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `learning_rate` | float | 2e-5 | Initial learning rate |
| `per_device_train_batch_size` | int | 1 | Examples per GPU |
| `gradient_accumulation_steps` | int | 64 | Accumulation steps |
| `max_steps` | int | 2048 | Total gradient updates |
| `warmup_ratio` | float | 0.1 | 10% warmup steps |
| `lr_scheduler_type` | str | "linear" | Scheduler: linear, cosine, constant |
| `weight_decay` | float | 0.01 | L2 regularization |
| `max_grad_norm` | float | 1.0 | Gradient clipping |

---

## 12. New Algorithms & Methods Added

### 12.1 Comparison Table

| Algorithm | Location | Date | Type | Key Innovation |
|-----------|----------|------|------|-----------------|
| **CoLM (Main)** | `facility_location.py` | 2024 | Facility Location | Mini-batch coresets with MeZO |
| **FairOT v1** | `fairot.py` | 2024 | Optimal Transport | Iterative partial OT selection |
| **FairOT v2** | `fairot2.py` | 2024 | Optimal Transport | Vectorized gain computation (*new*) |
| **SPOT** | `SPOTgreedy.py` | 2024 | Prototype Selection | Low-rank OT approximation |
| **GREATS** | `greats.py` | 2024 | Submodular| Validation-based with interactions |
| **MeZO (Efficient)** | `utils.py`, `subset_trainer_distributed.py` | 2024 | Zeroth-Order | Sparse last-layer ZO gradient (*new implementation*) |

### 12.2 Novel Contributions

1. **Source-Wise Selection** (Theorem 4.1 in paper)
   - Handles 300:1 imbalance in MathInstruct
   - Guarantees each source contributes to coreset

2. **Efficient MeZO Integration**
   - Combined facility location + ZO gradient for memory efficiency
   - <0.1s overhead vs 10-30% computation savings

3. **Adaptive Decomposition Model** (`custom_phi.py`)
   - Layer-wise forward pass for partial gradient computation
   - Enables arbitrary model architectures

4. **Multi-Algorithm Support**
   - Pluggable selection methods (FairOT, GREATS, SPOT, Facility Location)
   - Compare algorithms on same task

---

## 13. Key Files Summary

### Tier 1: Core Training
| File | Lines | Purpose |
|------|-------|---------|
| `subset_trainer_distributed.py` | 2,392 | Main trainer with data selection (largest) |
| `train.py` | 405 | Entry point (Phi-2 tuned) |
| `train_multitask.py` | 586 | Multi-task training with monitoring |

### Tier 2: Selection Algorithms  
| File | Lines | Purpose |
|------|-------|---------|
| `facility_location.py` | 127 | Facility location + source handling |
| `fairot2.py` | 484 | Optimized OT selection |
| `fairot.py` | 352 | Original OT selection |

### Tier 3: Supporting Infrastructure
| File | Lines | Purpose |
|------|-------|---------|
| `config_loader.py` | 265 | YAML config parsing |
| `huggingface_trainer.py` | 838 | Modified HF Trainer |
| `sinkhorn.py` | 197 | Partial OT solver (POT library) |
| `optimizer_factory.py` | 372 | Custom optimizer creation |

### Tier 4: Data & Models
| File | Lines | Purpose |
|------|-------|---------|
| `get_training_dataset.py` | 150+ | Dataset loading & sampling strategies |
| `custom_phi.py` | 267 | Phi model decomposition |
| `templates.py` | 150+ | Prompt templates (14 tasks) |

### Tier 5: Evaluation
| File | Lines | Purpose |
|------|-------|---------|
| `math_eval/run_open.py` | 11K | Math evaluation (GSM8K, MATH, etc.) |
| `superglue_eval/eval_superglue.py` | 100+ | SuperGLUE task evaluation |
| `math_eval/prompt_utils.py` | 38K | CoT & PoT prompts + examples |

---

## 14. Advanced Features

### 14.1 Gradient Computation Methods (`utils.py`)

**Available Gradient Types** for similarity computation:
```python
# 1. Representation (default)
grad = hidden_states[-1]  # Last layer activations

# 2. MeZO (zeroth-order)
z ~ N(0,I)
loss_plus = model(x, params + eps*z)
loss_minus = model(x, params - eps*z)
grad = (loss_plus - loss_minus) / (2*eps) * z

# 3. Masked Gradient
grad = full_gradient * mask  # Keep specific dimensions

# 4. Projected Gradient
grad = full_gradient @ projection_matrix  # Project to lower rank

# 5. MeZO + Representation
grad = mezo_grad * hidden_states
```

### 14.2 Similarity Metrics (`facility_location.py`)

```python
# 1. Cosine Similarity (default)
S = cosine(X, X)  # Normalized dot product

# 2. Euclidean Distance (converted to similarity)
dists = cdist(X, X, p=2)
S = max_dist - dists  # Invert distance

# 3. L1 Distance
dists = cdist(X, X, p=1)
S = max_dist - dists
```

### 14.3 Distributed Training Support

**Features** (in `subset_trainer_distributed.py`):
- Multi-GPU gradient synchronization before selection
- Per-GPU batch selection + reshuffling
- Distributed facility location via submodlib
- FSDP (Fully Sharded Data Parallel) support

---

## 15. Available Training Pipelines

### Pipeline 1: **Efficient Math Training** (Recommended)
```bash
# Run: scripts/run_math_efficient.sh
Duration:          ~4-6 hours (on 1x A100)
Effective Batch:   32 samples (4 per GPU × 8 GAS)
Coreset Size:      16 samples (50% of batch)
Method:            GREATS + MeZO sparse (0.7%)
```

### Pipeline 2: **Optimal Transport Training**
```bash
# Run: scripts/run_math.sh
Duration:          ~10-12 hours (on 1x A100)
Effective Batch:   64 samples (1 per GPU × 64 GAS)
Coreset Size:      32 samples (50% of batch)
Method:            FairOT + Full gradient computation
```

### Pipeline 3: **SuperGLUE Evaluation Training**
```bash
# Run: scripts/run_superglue.sh
Duration:          ~2-3 hours (on 4x GPUs)
Effective Batch:   32 samples (1 per GPU × 32 GAS)
Coreset Size:      16 samples (50% of batch)
Method:            Facility Location CoLM
```

### Pipeline 4: **Multi-Task Training** (New)
```bash
# Run: python train_multitask.py --config config.yaml
Datasets:          MetaMathQA + GSM8K (combined)
Metrics:           Per-task loss, perplexity, grad norms
Duration:          Configurable (max_steps based)
```

---

## 16. Dependency Ecosystem

### Core ML Stack
```
torch==2.2.1                      # Base framework
torchvision==0.17.1               # Vision utilities
torch-geometric (implicit)        # For submodular ops

transformers==4.43.2              # HF models (pinned version!)
peft==0.7.1                       # LoRA adapter
accelerate==0.33.0                # Distributed training

datasets==latest                  # Data loading
auto-gptq                         # GPTQ quantization
```

### Selection & Optimization
```
submodlib                          # Facility location algorithms
POT (Python Optimal Transport)     # Sinkhorn solver
scikit-learn==1.4.2                # Utilities
```

### Evaluation & Monitoring
```
wandb                             # Experiment tracking
torchmetrics                      # Metrics computation
rouge-score, bert_score           # NLG metrics
nltk                              # NLP utilities
```

### Inference
```
vllm/                             # Built-in serving (308 Python files)
(vLLM supports: CUDA, Hip, CPU backends)
```

---

## 17. Key Metrics & Experiments

### 17.1 Evaluation Metrics

**Math Reasoning** (GSM8K, MATH, etc.):
- Exact Match (EM)
- Program Validity (for PoT)
- CoT Accuracy (intermediate reasoning)

**Classification** (SuperGLUE):
- Accuracy
- F1-score
- Matthews Correlation (for imbalanced tasks)

**Tracked During Training**:
- Training loss
- Perplexity: $\exp(\text{loss})$
- Gradient norm (L2)
- GPU memory usage
- Selection time (overhead measurement)

### 17.2 Supported Model Families

| Model | Parameters | Supported | Notes |
|-------|-----------|-----------|-------|
| Phi-2 | 2.7B | ✓ | Primary testing (scripts optimized) |
| Phi-3-mini | 3.8B | ✓ | InstructPT variant |
| LLaMA-2 | 7B–70B | ✓ | Via HF model hub |
| LLaMA-3.1 | 8B–70B | ✓ | config.yaml example |
| StableLM | 3B–7B | ✓ | Alternative base model |
| Qwen | 7B–72B | ~ | May need `custom_phi.py` updates |

---

## 18. Experimental Reproducibility

### Key Files for Reproduction
```
requirement.txt              # Exact dependencies
config.yaml                  # Main configuration
setup.py                     # Package installation
IMPLEMENTATION_GUIDE.md      # Algorithm details
QUICK_REFERENCE.md           # Command reference
```

### Data Preparation
```bash
# Download from: https://drive.google.com/file/d/1kpYMJ0xrn0eLyv-uwhUZCTjFWT6Zlb-Q/
# Extract to: /data/*.jsonl

# Or use built-in datasets:
python data/get_training_dataset.py --dataset load-superglue-sst2
```

### Running Experiments
```bash
# 1. Install
pip install -r requirement.txt --no-cache-dir --no-build-isolation

# 2. Setup vLLM (if evaluating)
VLLM_INSTALL_PUNICA_KERNELS=1 pip install vllm/

# 3. Train
bash scripts/run_math_efficient.sh

# 4. Evaluate
cd math_eval && bash eval_finetuned.sh /path/to/model
```

---

## 19. Known Limitations & Future Work

### Current Limitations
1. **Transformers Version Lock**: Code is tied to `transformers==4.43.2`
   - Different versions may require updates to `custom_phi.py`
   
2. **vLLM Evaluation**: Requires GPU for inference (no CPU-only mode)
   - Fallback to single-pass inference in `run_open.py`

3. **Source-Wise Selection**: Requires `source` field in dataset
   - Basic dataset may fail without metadata

4. **LoRA Specific**: Primarily tested with LoRA finetuning
   - Full parameter training supported but less optimized

### Potential Extensions
1. Support additional models (Mistral, Qwen, etc.) via `custom_phi.py`
2. Quantization integration (GPTQ, AWQ with CoLM selection)
3. Multi-node distributed training enhancements
4. Streaming/online data selection for continuous learning
5. GPU-accelerated Sinkhorn solver (POT CUDA backend)

---

## 20. File Statistics Summary

| Metric | Value |
|--------|-------|
| **Total Python Files** | ~350+ |
| **Main Training Code** | 7,322 LOC (colm/train/) |
| **Total Project LOC** | ~12,000+ (excluding vLLM) |
| **vLLM Size** | 20 MB, 308 Python files |
| **Largest File** | subset_trainer_distributed.py (2,392 lines) |
| **Configuration Files** | YAML, JSON, shell scripts |
| **Largest Dataset** | MathInstruct (~250K examples) |
| **Supported Benchmarks** | 6 math + 11 classification tasks |

---

## Conclusion

CoLM represents a sophisticated integration of:

1. **Modern ML** (LoRA, distributed training, HF ecosystem)
2. **Advanced Algorithms** (Facility Location, Optimal Transport, MeZO)
3. **Production Infrastructure** (vLLM serving, comprehensive evaluation, config management)
4. **Research Reproducibility** (detailed documentation, modular design, multiple ablations)

The codebase is production-ready for memory-efficient LLM training on imbalanced data mixtures, with extensive evaluation on both mathematical reasoning and natural language understanding tasks.

