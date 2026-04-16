# LLM Fine-Tuning Codebases: Detailed Comparison

## Executive Summary

This document compares two advanced LLM fine-tuning approaches:

- **Codebase 1 (CoLM)**: Mini-batch Coresets for Memory-efficient Language Model Training (ICLR 2025)
  - Focuses on **data coreset selection** (CoLM, GREATS, FairOT, SPOT algorithms)
  - Uses standard PEFT LoRA with Muon optimizer
  - Employs efficient zeroth-order gradient estimation for selection

- **Codebase 2 (Riemannian)**: Fixed Rank LoRA with Riemannian Geometry Optimization
  - Focuses on **Riemannian manifold optimization** of LoRA parameters
  - Uses custom Riemannian optimizers (RiemannianLora, RiemannianSGD)
  - Employs specialized LoRA initialization on fixed-rank manifolds

---

## 1. Training Architecture

### 1.1 Entry Points and Configuration

#### CoLM (Codebase 1)
```
train.py → HfArgumentParser → ModelArguments, DataArguments, TrainingArguments
         → SubsetTrainer or SubsetTrainerEfficient (custom trainer)
         → JSON or CLI-based argument parsing
```

**Key Features:**
- Uses Hugging Face transformers architecture
- CLI arguments: `ModelArguments`, `DataArguments`, `TrainingArguments`
- Extends HF Trainer with custom subset selection logic
- Distributed training via `torch.distributed`

#### Riemannian (Codebase 2)
```
run.py → OmegaConf.load(config.yaml) → config structure
      → run_experimet.py (run function) → Task enum
      → SFTTrainer (from TRL library)
      → YAML-based configuration
```

**Key Features:**
- Uses OmegaConf for configuration management
- Task-based architecture (INFERENCE, FINETUNE, VALIDATE, INITIALIZE)
- Built on top of TRL's SFTTrainer (Supervised Fine-Tuning)
- More modular with separate functions: `model_loader`, `data_preparation`, `finetune`

### 1.2 Data Loading and Processing

#### CoLM Data Pipeline
```python
# From train.py
1. Load dataset (SuperGLUE or custom task)
2. Create SupervisedDataset with custom collator
3. DataCollatorForSupervisedDataset with source information
4. Distributed sampling with SubsetTrainer

# Key data formats:
- input_ids + attention_mask
- labels (completion tokens marked, rest masked with -1)
- sources (for multi-dataset training)
- indices (for tracking original sample indices)
```

**Data Selection Integration:**
- Batches are first collected to reach gradient accumulation size
- Representations are extracted via forward passes
- Selection algorithm runs on rank 0, broadcasts selected indices
- Only selected samples are trained on

#### Riemannian Data Pipeline
```python
# From data_preparation.py and run_experimet.py
1. Load dataset from disk (DatasetDict.load_from_disk)
2. Train/val/test splits
3. DataCollatorForCompletionOnlyLM (from TRL)
4. No explicit per-batch selection

# Dataset format:
- text_wa_answer: complete text with answer
- standard HF dataset format
```

**Key Difference:**
- Simplified, static dataset loading
- No per-batch selection mechanism
- Fixed dataset composition across epochs

### 1.3 Optimizer Selection and Configuration

#### CoLM Optimizer Setup
```python
# Default: Muon optimizer (from HF transformers)
optimizer = AdamW or custom optimizer
```

**Key aspects:**
- Flexible optimizer through HF trainer
- Different learning rates for LoRA A and B matrices possible
- Integrates with gradient clipping and warmup scheduling

#### Riemannian Optimizer Setup
```python
# Multiple optimizer options available:
def get_optimizer_cls_and_kwargs(config, model):
    name2optim = {
        'LoRA_plus': get_LoRA_plus,           # AdamW for LoRA
        'LoRA_RITE': get_LoRA_RITE,          # Not implemented
        'LoRA_Riemanian': get_Riemannian_LoRA,  # Riemannian geometry
        'SGD_Riemannian': get_Riemannian_SGD,   # Riemannian SGD
        'PseudoAdaGrad': get_pseudo_adagrad,    # Adaptive method
    }
```

**Key Parameter Grouping:**
```python
# Separates LoRA_A and LoRA_B into distinct parameter groups
param_groups = [
    {"params": [lora_A_layer1, lora_A_layer2, ...]},
    {"params": [lora_B_layer1, lora_B_layer2, ...]},
]
```

### 1.4 Gradient Computation Strategy

#### CoLM Gradient Computation
```python
# From subset_trainer_distributed.py::_inner_training_loop

# Two modes of gradient computation:

# Mode 1: Representation-based selection
if self.args.data_selection_unit == "rep":
    # Extract hidden states from last layer
    hidden_states = model(..., output_hidden_states=True).hidden_states
    rep = hidden_states[-1][ids, pos]  # Last token representation

# Mode 2: MeZO (Zeroth-order) gradient
elif self.args.data_selection_unit == "mezo":
    # Efficient zeroth-order gradient estimation
    z ~ N(0, I)  # Random perturbation
    loss1 = forward(model + eps*z)
    loss2 = forward(model - eps*z)
    projected_grad = (loss1 - loss2) / (2*eps)

# Mode 3: Masked gradients (backprop)
elif self.args.data_selection_unit == "masked_grad":
    # Standard backpropagation on selected layers only
    loss.backward()
    grads = extract_gradients(model.selected_layers)
```

**Key Components:**
- Representations aggregated across all gradient accumulation steps
- Distributed gathering across multiple GPUs
- Sync point before selection

#### Riemannian Gradient Computation
```python
# Standard backpropagation through SFTTrainer
# No explicit zeroth-order estimation
# Gradients flow through entire model

# Gradient flow:
loss.backward() → all gradients computed
optimizer.step() → custom optimizer applies manifold operations
```

### 1.5 Loss Computation

#### CoLM Loss Computation
```python
# From subset_trainer_distributed.py::compute_loss

if isinstance(model, PeftModel):
    outputs = model(**inputs)  # LoRA-wrapped forward pass
else:
    outputs = model(**inputs)

# Label smoothing applied if configured
if label_smoother:
    loss = label_smoother(outputs, labels, shift_labels=True)
else:
    loss = outputs["loss"]

# Per-sample weighting based on selection weights
loss = loss * input_weight  # weights from selection algorithm
```

**Loss Features:**
- Per-sample weighting based on coreset selection
- Normalized by gradient accumulation steps
- Support for label smoothing

#### Riemannian Loss Computation
```python
# Standard SFTTrainer loss computation
# No custom per-sample weighting
loss = model(**inputs)["loss"]
```

### 1.6 Training Loop Structure

#### CoLM Training Loop
```
for epoch in epochs:
    for outer_step, batch in enumerate(train_dataloader):
        # Phase 1: Representation collection
        for inner_step in range(gradient_accumulation_steps):
            rep = save_select(model, inputs)  # Extract representation
            total_reps.append(rep)
            continue  # No gradient update yet

        # Phase 2: Selection (on rank 0)
        all_reps = gather_from_all_ranks(total_reps)
        
        if rank == 0:
            selected_idx = select_data(
                all_reps,
                max_samples=gradient_accumulation_steps,
                method=data_selection_method  # greats, fairot, etc.
            )
        
        selected_idx = broadcast_from_rank_0(selected_idx)

        # Phase 3: Gradient computation and update
        for inner_step, selected_input in enumerate(selected_inputs):
            loss = compute_loss(model, selected_input, weight)
            backward(loss)
        
        optimizer.step()
        model.zero_grad()
```

**Key Phases:**
1. **Collection**: Gather representations from unselected samples
2. **Selection**: Run coreset selection algorithm on rank 0
3. **Training**: Train only on selected samples
4. **Update**: Optimizer step

**Memory Efficiency:**
- Selection happens BEFORE backprop on most samples
- Only selected samples go through backprop
- Significant reduction in GPU memory usage

#### Riemannian Training Loop
```
for epoch in epochs:
    for batch in train_dataloader:
        # Standard training loop
        outputs = model(**batch)
        loss = outputs["loss"]
        
        accelerator.backward(loss)
        
        # Optional: QR decomposition after each step
        if config.lora_qr:
            for module in lora_modules:
                Q, R = qr(A.T)
                A = Q.T
                B = B @ R.T
        
        optimizer.step()
        model.zero_grad()
```

**Training Characteristics:**
- Standard supervised fine-tuning loop
- All samples in batch are trained
- QR orthogonalization optional (LoraQR callback)

---

## 2. Data Selection Strategy

### 2.1 CoLM: Multi-Algorithm Coreset Selection

#### Overview
CoLM implements **5 coreset selection algorithms** that operate per mini-batch:

#### Algorithm 1: Facility Location (CoLM)
```python
# From colm/train/facility_location.py
# Function: get_orders_and_weights(max_samples, inputs, metric, y, per_class_start, strategy)

# Process:
1. Compute similarity matrix between all representations
2. Select K closest centers (facility location centers)
3. Assign remaining samples to nearest center
4. Weight samples based on cluster membership

# Metric options: "cosine", "euclidean"
```

**Properties:**
- Selects diverse, representative samples
- Greedy algorithm with O(n²) complexity
- Returns indices + weights for weighted training

#### Algorithm 2: GREATS (Gradient Relevance)
```python
# From colm/train/greats.py
# Function: greedy_selection(norms, similarities, max_samples)

# Process:
1. Compute gradient norms of representations
2. Compute similarity matrix
3. Greedily select samples with high gradient norm
4. Prioritize samples dissimilar from already selected

# Formula:
score = norm_i + similarity_penalty
```

**Properties:**
- Balances gradient magnitude with diversity
- Efficient greedy implementation
- Good for identifying high-impact samples

#### Algorithm 3: FairOT
```python
# From colm/train/fairot2.py
# Function: greedy_fairot(S, max_samples, dist, iters, reg)

# Process:
1. Formulate as optimal transport problem
2. Use regularized Wasserstein distance
3. Balance similarity (S matrix) with fairness constraints (dist)
4. Iteratively optimize subset selection

# Parameters:
- S: similarity matrix
- dist: fairness/distribution constraint
- reg: regularization strength (~1e-1)
- iters: optimization iterations (~500)
```

**Properties:**
- Considers fairness across data sources
- Handles multi-source datasets
- More computationally expensive than others

#### Algorithm 4: FairOT Multi-source
```python
# Per-source application of FairOT
# Ensures representation from multiple data sources
# Lambda function wrapper: lambda S,k, dist: fairot2.greedy_fairot(S, k, reg=1e-1, dist=dist, iters=500)
```

#### Algorithm 5: SPOT (Streaming subset selection)
```python
# Less frequently used in current implementation
# Greedy streaming algorithm for online subset selection
```

### 2.2 Riemannian: No Explicit Data Selection

**Key Difference:**
- No per-batch selection mechanism
- **Static dataset composition**: same samples used across epochs
- All samples in mini-batch are trained equally

**Batching Strategy:**
```python
# Standard random sampling from fixed dataset
dataset.sort_by_random(seed)
train_loader = DataLoader(dataset, batch_size=B)
```

### 2.3 Batch Creation Comparison

#### CoLM Batch Creation
```
Step 1: Load B samples (gradient accumulation size)
Step 2: Extract representations for all B samples
Step 3: Run selection algorithm → select K < B samples
Step 4: Create training batches from K selected samples
Step 5: Train on K samples (weighted by selection)
```

**Effective Batch Efficiency:**
- Trains on smaller effective batch (K samples)
- Skips uninformative samples
- Gradient accumulation keeps batch semantics

#### Riemannian Batch Creation
```
Step 1: Load B samples (batch size)
Step 2: Create training batch directly
Step 3: Train on all B samples
```

**Simplicity:**
- Direct, no intermediate steps
- Clearer data flow

---

## 3. LoRA Implementation

### 3.1 CoLM: Standard PEFT LoRA

#### Configuration
```python
# From colm/train/train.py
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=model_args.lora_r,                    # Rank (e.g., 32, 64)
    lora_alpha=model_args.lora_alpha,       # Scaling factor
    lora_dropout=model_args.lora_dropout,   # Dropout (often 0)
    target_modules=model_args.lora_target_modules,  # Which layers to adapt
    modules_to_save=modules_to_save         # Additional modules
)

model = get_peft_model(model, lora_config)
```

#### Structure
```
For each target layer (e.g., q_proj, v_proj):
    Original: Y = W·X
    With LoRA: Y = W·X + B·A·X
    
    Where:
    - A ∈ ℝ^(r × d_in): projection to lower rank
    - B ∈ ℝ^(d_out × r): projection back to original rank
    - r << min(d_in, d_out): rank parameter
```

#### Initialization Strategy
```python
# Default PEFT initialization:
# A: Kaiming uniform distribution
# B: Zero initialization
# Result: Identity transformation at start (B·A = 0)
```

#### Training Updates
```python
# Standard gradient descent on parameters:
dA ← ∂L/∂A
dB ← ∂L/∂B
A ← A - lr·dA
B ← B - lr·dB
```

### 3.2 Riemannian: Fixed-Rank LoRA

#### Key Concept: Fixed-Rank Manifold
```
Standard LoRA operates in Euclidean space: ℝ^(r×d) × ℝ^(d×r)

Fixed-Rank LoRA operates on Riemannian manifold:
    M_{r} = {(A, B) : rank(A·B) = r}
    
Representation: Doubles dimension
    A_actual = [A_ortho | A_orth] ∈ ℝ^(2r × d_in)
    B_actual = [B; B_orth] ∈ ℝ^(d_out × 2r)
    
Where:
    - A_ortho, B_ortho: orthogonal components
    - Used for Riemannian geometry operations
```

#### Configuration
```python
# From src/initializers/LoRA_init.py
def _init_riemannian(init_args, model):
    for lora_A_layer in model.modules():
        r, m = lora_A_weight.shape
        r = r // 2  # Account for doubled rank
        
        # Initialize orthogonal component
        A_params = torch.empty(r, m)
        nn.init.orthogonal_(A_params, gain=1.0)
        
        # Concatenate with zero component
        A_params = torch.cat([A_params, torch.zeros_like(A_params)], dim=0)
        lora_A_weight.copy_(A_params)
```

#### Initialization Details
```
LoRA_A (size: 2r × d_in):
    [Orthogonal matrix U ∈ O(r, d_in)]
    [Zero matrix 0 ∈ ℝ^(r × d_in)]

LoRA_B (size: d_out × 2r):
    [Orthogonal matrix V ∈ O(d_out, r) scaled by d_out^(-0.5)]
    [Orthogonal matrix V_orth ∈ O(d_out, r)]
```

#### Representation on Manifold
```
Point on manifold: (A, B) with A ∈ St(m, r), B ∈ ℝ^(n, r)

Class FixedRank maintains:
    - Retraction: maps tangent space back to manifold
    - Transport: moves vectors between tangent spaces
    - Embedding: embeds tangent vectors in ambient space
    - Euclidean to Riemannian projection
```

#### Key Riemannian Operations

**1. Tangent Vector Projection (Euclidian → Riemannian)**
```python
def euclidian_to_reimanian(point, vec):
    A, B = point
    dot_A, dot_B = vec
    
    # Project dot_A onto orthogonal complement of span(A)
    dot_A -= A @ (A.T @ dot_A)
    
    return dot_A, dot_B
```

**2. Retraction (Manifold update)**
```python
def retraction(point, tangent_vec):
    A_part, B_part = embedding(point, tangent_vec, ambient=False)
    
    # QR decomposition
    Q_l, R_l = torch.linalg.qr(A_part)
    Q_r, R_r = torch.linalg.qr(B_part.T)
    T = R_l @ R_r.T
    
    # SVD on T
    U, S, Vt = torch.linalg.svd(T)
    U = U[:, :r]
    S = S[:r]
    Vt = Vt[:r, :]
    
    new_A = Q_l @ U
    new_B = torch.diag(S) @ Vt @ Q_r.T
    
    return [new_A, new_B]  # Both on manifold
```

**3. Parallel Transport (moving vectors between tangent spaces)**
```python
def transport(point_X, point_Y, vec_X):
    # Project and adapt vector from X's tangent space to Y's
    U_l, _ = point_Y
    _, Vt_r = to_right(point_Y)
    
    A_part, B_part = embedding(point_X, vec_X, ambient=False)
    dot_U = (A_part - U_l @ (U_l.T @ A_part)) @ (B_part @ Vt_r.T)
    dot_V = (U_l.T @ A_part) @ B_part
    
    return [dot_U, dot_V]
```

#### Parameter Updates via Retraction
```python
# Instead of Euclidean gradient descent:
# θ_new = θ_old - lr·∇L(θ_old)

# Use Riemannian gradient descent:
# θ_new = Retract_θ_old(-lr·∇L(θ_old))

# This ensures the new parameters remain on the fixed-rank manifold
```

### 3.3 Comparison: Standard vs Fixed-Rank LoRA

| Aspect | Standard LoRA | Fixed-Rank LoRA |
|--------|---------------|-----------------|
| **Parameter Space** | Euclidean: ℝ^(r×d) × ℝ^(d×r) | Riemannian: St(m,r) × ℝ^(n,r) |
| **Rank Constraint** | Soft/implicit | Hard/explicit (manifold) |
| **Initialization** | Kaiming(A), Zero(B) | Orthogonal(A), Orthogonal(B) |
| **Update Rule** | Standard SGD | Riemannian retraction |
| **Memory** | Standard | ~2× (doubled rank in params) |
| **Convergence** | Euclidean convergence | Geodesically optimal |

---

## 4. Optimizer Differences

### 4.1 CoLM Optimizer: Standard with Potential Modifications

#### Default Setup
```python
# From HF Transformers
optimizer = AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

# Optional: Different LR for LoRA A vs B (not yet implemented in shown code)
param_groups = [
    {"params": lora_A_params, "lr": 5e-4},
    {"params": lora_B_params, "lr": 5e-4 * scale_factor},
]
```

#### Optimizer Step
```python
# Standard PyTorch SGD with momentum
m_t = β₁·m_{t-1} + (1-β₁)·∇L
v_t = β₂·v_{t-1} + (1-β₂)·(∇L)²
θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)

# Optional gradient clipping
if args.max_grad_norm > 0:
    grad_norm = clip_grad_norm(model.parameters(), max_grad_norm)
```

### 4.2 Riemannian Optimizer Variants

#### Optimizer A: LoRA_plus (Standard AdamW with LoRA-specific parameter groups)

```python
def get_LoRA_plus(optimizer_config, model):
    lora_A_params = extract_lora_A_params(model)  # lr = α
    lora_B_params = extract_lora_B_params(model)  # lr = α·B_lr_scale
    
    param_groups = [
        {"params": lora_A_params, "lr": α},
        {"params": lora_B_params, "lr": α·B_lr_scale},
    ]
    
    optimizer = torch.optim.AdamW
    return optimizer, {"params": param_groups}
```

**Key Feature:** Differential learning rates for A and B matrices

#### Optimizer B: RiemannianLora (Manifold-aware optimization)

```python
class RiemannianLora(Optimizer):
    def step(self):
        for group in param_groups:
            A = group["params"][A_idx]  # Orthogonal matrix
            B = group["params"][B_idx]
            r = A.shape[0] // 2
            
            # 1. Extract deltas (gradients) from tangent space
            point, grad_vec = self.get_deltas(A, B, manifold)
            
            # 2. Apply first momentum (velocity)
            momentum_vec = self.apply_momentum(
                point=point,
                vec=grad_vec,
                state=state,
                manifold=manifold,
                beta=betas[0]
            )
            
            # 3. Apply second momentum (adaptive learning rate like AdamW)
            dot_A, dot_B = self.apply_second_momentum(
                grad_vec=grad_vec,
                momentum_vec=momentum_vec,
                state=state,
                beta=betas,
                manifold=manifold
            )
            
            # 4. Update via Riemannian retraction
            new_A, new_B = manifold.retraction(
                point=point,
                tangent_vec=[-lr * dot_A, point[1] - lr * dot_B]
            )
            
            # 5. Maintain orthogonality structure
            A_zero = torch.zeros_like(new_A)
            B_orth = torch.linalg.qr(new_B.T).Q.T
            
            new_A = torch.cat([new_A, A_zero], dim=1)
            new_B = torch.cat([new_B, B_orth], dim=0)
            
            group["params"][A_idx].copy_(new_A)
            group["params"][B_idx].copy_(new_B)
```

**Key Features:**
- Operates directly on Riemannian manifold
- Uses manifold's retraction for updates
- Maintains orthogonal structure
- First momentum: standard exponential moving average
- Second momentum: adaptive scaling per component

#### Optimizer C: RiemannianSGD (Manifold SGD without adaptive rate)

```python
class RiemannianSGD(Optimizer):
    # Same setup as RiemannianLora but without second momentum
    # Uses only first momentum for velocity
    # Fixed learning rate without adaptive scaling
```

#### Optimizer D: PseudoAdaGrad (Adaptive gradient scaling)

```python
class PseudoAdaGrad(Optimizer):
    def step(self):
        for group in param_groups:
            A = group["params"][A_idx]
            B = group["params"][B_idx]
            
            # Compute gradients
            grad_vec = [A.grad, B.grad]
            
            # Apply first momentum
            m_t[A] = β₀·m_{t-1}[A] + (1-β₀)·A.grad
            m_t[B] = β₀·m_{t-1}[B] + (1-β₀)·B.grad
            
            # Apply second momentum (without RMS, just variance)
            v_t[A] = β₁·v_{t-1}[A] + (1-β₁)·||A.grad||²
            v_t[B] = β₁·v_{t-1}[B] + (1-β₁)·||B.grad||²
            
            # Update (Euclidean, not manifold-aware)
            A_new = A - lr · m_t[A] / (√v_t[A] + ε)
            B_new = B - lr · m_t[B] / (√v_t[B] + ε)
```

**Key Features:**
- Similar to AdamW but uses norm-based second moment
- Not manifold-aware
- Works in Euclidean space

#### Optimizer E: NoOptimizer (for initialization only)

```python
class NoOptimizer:
    # Placeholder optimizer with no-op step()
    # Used during initialization phase (INITIALIZE task)
```

### 4.3 Optimizer Comparison Matrix

| Feature | CoLM | LoRA_plus | RiemannianLora | RiemannianSGD | PseudoAdaGrad |
|---------|------|-----------|----------------|---------------|---------------|
| **Geometry** | Euclidean | Euclidean | Riemannian | Riemannian | Euclidean |
| **First Momentum** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Second Momentum** | ✓ (adaptive) | ✓ | ✓ (adaptive on manifold) | ✗ | ✓ (norm-based) |
| **Parameter Groups** | ✓ | ✓ (A vs B) | ✓ (per layer) | ✓ | ✓ |
| **Manifold-aware** | ✗ | ✗ | ✓ | ✓ | ✗ |
| **Retraction** | ✗ | ✗ | ✓ | ✓ | ✗ |
| **Transport** | ✗ | ✗ | ✓ | ? | ✗ |

### 4.4 Riemannian Geometry in Optimization

#### Why Manifold Operations Matter

**Problem:**
- Standard LoRA: parameters can move off the fixed-rank manifold
- Result: rank constraints become soft/approximate

**Solution:**
- Use retraction: maps Euclidean updates back to manifold
- Use parallel transport: ensures consistent momentum across tangent spaces
- Maintain orthogonality: A ∈ St(m,r) (Stiefel manifold)

#### Computational Overhead

```
Additional cost per step:
1. QR decomposition: O(rm² + rn²)  [for m×r and n×r matrices]
2. SVD: O(r³)
3. Parallel transport: O(rm + rn)

Total: ~O(rm² + rn²) vs O(rmn) for standard update
Roughly 1-3% overhead for modern matrix sizes
```

---

## 5. Evaluation Pipeline

### 5.1 CoLM Evaluation: Comprehensive Multi-Task Setup

#### Evaluation Infrastructure
```python
# From math_eval/ and superglue_eval/ directories
Supported Tasks:
  - Math: 6 datasets (MATH, GSM8K, SVAMP, etc.)
  - SuperGLUE: 11 tasks (RTE, CB, CoLA, SST-2, MRPC, QQP, STS-B, MNLI, QNLI, AXb, AXg)
  - Custom: any HF dataset compatible
```

#### Inference Strategy: vLLM-based
```python
# Custom vLLM integration for fast batch inference

# High-level flow:
1. Load fine-tuned model via vLLM
2. Create batches of prompts
3. Parallel inference with vLLM optimization
   - Paged attention: efficient memory management
   - Token-level batching: maximize GPU utilization
   - Speculative decoding: optional speedup
4. Parse outputs
5. Calculate metrics
```

#### Key Evaluation Components
```bash
# math_eval/eval_finetuned.sh
${VLLM_PATH}/python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --tensor-parallel-size ${NUM_GPUS} \
    --max-model-len 4096 \
    --dtype float16
```

#### Evaluation Metrics
```
For Math Tasks:
- Accuracy (exact match after majority voting)
- Pass@1, Pass@K metrics

For SuperGLUE:
- Accuracy, F1, Matthews Correlation, Spearman correlation (task-specific)
- Overall SuperGLUE score (13-task average)
```

### 5.2 Riemannian Evaluation: Pipeline Inference Setup

#### Evaluation Setup
```python
# From src/eval.py
def _inference_model(eval_cfg, pl, test_dataset):
    model_preds = []
    
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        # Split dataset into chunks for memory efficiency
        for i, split in enumerate(np.array_split(
            np.arange(len(test_dataset)), 
            eval_cfg.num_splits
        )):
            # Batch inference using HF pipeline
            model_pred = pl(
                test_dataset.select(split)['text_wa_answer'],
                return_full_text=False,
                max_new_tokens=eval_cfg.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                batch_size=eval_cfg.batch_size,
            )
            model_preds += model_pred
            
            # Clear GPU memory after each split
            if eval_cfg.empty_cache:
                torch.cuda.empty_cache()
    
    return model_preds_merged
```

#### Inference Method
```python
pl = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Features:
- HF pipeline (simpler than vLLM)
- Auto mixed precision (AMP)
- Manual dataset splitting for memory bounds
- GPU cache clearing between splits
- Batch processing
```

#### Output Processing
```python
def make_preds(config, pl, test_dataset):
    model_preds = _inference_model(eval_cfg, pl, test_dataset)
    
    # Create DataFrame with predictions
    preds_df = test_dataset.to_pandas()
    preds_df["model_pred"] = model_preds
    
    # Save to parquet
    preds_df.to_parquet(config.evaluation_config.dump_path)
    
    return preds_df
```

### 5.3 Evaluation Comparison

| Aspect | CoLM | Riemannian |
|--------|------|-----------|
| **Inference Framework** | vLLM (optimized) | HF pipeline (simpler) |
| **Tasks Covered** | 6 math + 11 SuperGLUE (17 total) | Custom tasks (configurable) |
| **Parallelization** | Tensor parallelism via vLLM | Pipeline built-in batching |
| **Memory Strategy** | Paged attention | Manual split + cache clear |
| **Metrics Calculation** | Integrated in eval scripts | Output saved for external eval |
| **Output Format** | Aggregated scores | Dataframe with predictions |
| **Speed** | Highly optimized (paged attention, speculative decoding) | Moderate (standard pipeline) |

### 5.4 Configuration for Evaluation

#### CoLM Evaluation Config
```bash
math_eval/eval_finetuned.sh /path/to/model
# Automatically evaluates on all 17 benchmarks
# Reports detailed metrics per task
```

#### Riemannian Evaluation Config
```yaml
# In config.yaml
evaluation_config:
  num_splits: 10           # Memory splits
  batch_size: 32           # Inference batch size
  max_new_tokens: 512      # Generation limit
  empty_cache: true        # GPU memory reset
  dump_path: "predictions/{task_idx}.parquet"
```

---

## 6. Key Differences Summary Table

| Dimension | CoLM (Coresets) | Riemannian (Manifold) |
|-----------|-----------------|----------------------|
| **Primary Innovation** | Data coreset selection | Riemannian LoRA optimization |
| **Entry Point** | train.py (args) | run.py (YAML config) |
| **Configuration** | CLI + JSON | YAML via OmegaConf |
| **Data Selection** | Per-batch coreset (GREATS, FairOT) | None (static composition) |
| **Selection Methods** | 5 algorithms + fairness | N/A |
| **Batch Size Effective** | K << B | B (all samples) |
| **LoRA Type** | Standard (PEFT) | Fixed-Rank Manifold |
| **LoRA Initialization** | Kaiming(A) + Zero(B) | Orthogonal(A,B) |
| **LoRA Parameters** | r × d + d × r | 2r × d + d × 2r (doubled) |
| **Optimizer** | AdamW (standard) | RiemannianLora, RiemannianSGD, etc. |
| **Update Rule** | Euclidean: θ - lr∇L | Riemannian: Retract(θ - lr∇L) |
| **Geometry** | Euclidean space | Fixed-rank manifold |
| **Retraction** | None | QR + SVD per step |
| **Parallel Transport** | None | Vector transport between tangent spaces |
| **Memory Usage** | Lower (selective training) | Higher (doubled LoRA rank) |
| **Inference** | vLLM (faster) | HF Pipeline (simpler) |
| **Evaluation Scope** | 17 benchmarks (6 math + 11 GLUE) | Custom (configurable) |
| **Training Overhead** | Selection overhead (~0.1s/step) | Manifold operations (~1-3% overhead) |
| **Suited For** | Large, mixed datasets | Manifold-based optimization |

---

## 7. Code Flow Diagrams

### 7.1 CoLM Training Flow

```
╔════════════════════════════════════════════════════════════════╗
║                    CoLM TRAINING FLOW                          ║
╚════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────┐
│ train.py                                                        │
│ - Load model + tokenizer                                        │
│ - Setup LoRA (standard PEFT)                                    │
│ - Create SubsetTrainer or SubsetTrainerEfficient                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ FOR each epoch:                                                 │
│   FOR each outer_step (mini-batch):                             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        │                            │
        ▼                            ▼
   [Phase 1]                    [Phase 2-3]
   Representation            Selection & Training
   Collection                
        │                            │
        ├─ FOR each gradient        │
        │  accumulation step:        │
        │  │                         │
        │  ├─ inputs = next_batch   │
        │  │                         │
        │  ├─ Extract representation│
        │  │  (hidden state or      │
        │  │   MeZO gradient)       │
        │  │                         │
        │  ├─ total_reps.append(rep)
        │  │                         │
        │  └─ continue (no backprop)
        │                            │
        └────────┬───────────────────┘
                 │
                 ▼
        ┌──────────────────────────┐
        │ Gather all_reps from all │
        │ ranks to rank 0          │
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │ IF rank == 0:            │
        │  Run selection algorithm:│
        │  - Facility Location     │
        │  - GREATS                │
        │  - FairOT                │
        │  → selected_idx[]        │
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │ Broadcast selected_idx   │
        │ to all ranks             │
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │ FOR each selected sample:│
        │  - compute_loss()        │
        │  - backward()            │
        │  - accumulate gradients  │
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │ optimizer.step()         │
        │ model.zero_grad()        │
        │ epoch += 1               │
        └──────────────────────────┘
```

### 7.2 Riemannian Training Flow

```
╔════════════════════════════════════════════════════════════════╗
║               RIEMANNIAN TRAINING FLOW                         ║
╚════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────┐
│ run.py                                                          │
│ - Load config (YAML)                                            │
│ - Load model + tokenizer                                        │
│ - Create Fixed-Rank LoRA                                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ IF task == INITIALIZE:                                          │
│  - Use RSVD initialization callback                             │
│  - Train with NoOptimizer (frozen)                              │
│  - Initialize LoRA parameters on manifold                       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ IF task == FINETUNE:                                            │
│  - Setup SFTTrainer                                             │
│  - Create optimizer (RiemannianLora, etc.)                      │
│  - Optional: LoraQR callback                                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ FOR each epoch:                                                 │
│   FOR each batch:                                               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        │                            │
        ▼                            ▼
   [Forward Pass]              [Backward Pass]
        │                            │
        ├─ inputs = next_batch      │
        │                            │
        ├─ outputs = model(**inputs)
        │                            │
        ├─ loss = outputs["loss"]   │
        │                            │
        └─ return loss             ◄┴─ loss.backward()
                                    │
                                    ▼
                          ┌──────────────────────────┐
                          │ RiemannianOptimizer.step │
                          └──────────┬───────────────┘
                                    │
                          ┌─────────▼─────────────┐
                          │ FOR each param group: │
                          │  A, B = params       │
                          └──────────┬────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
              [Extract deltas]  [First Momentum] [Second Momentum]
                    │               │               │
              point, vec =      momentum_vec =  dot_A, dot_B =
              get_deltas()      apply_momentum  apply_second_momentum
                    │               │               │
                    └───────┬───────┴───────┬───────┘
                            │               │
                            ▼               │
                      ┌───────────────────────────┐
                      │ Retraction (manifold map) │
                      │ QR + SVD per step         │
                      │ new_A, new_B =            │
                      │   ...manifold.retraction()
                      └──────────┬────────────────┘
                                 │
                                 ▼
                      ┌────────────────────────────┐
                      │ Update parameters:         │
                      │ A_new, B_new               │
                      │ Maintain orthogonal struct │
                      └──────────┬─────────────────┘
                                 │
                                 ▼
                      ┌────────────────────────────┐
                      │ IF lora_qr callback:       │
                      │  A_new, B_new = QR(A, B)  │
                      └────────────────────────────┘

                      ▼
                   ┌──────────────────┐
                   │ model.zero_grad()│
                   │ epoch += 1       │
                   └──────────────────┘
```

### 7.3 Data Selection Algorithm Detail (CoLM)

```
┌─────────────────────────────────────────────────────────────────┐
│           CORESET SELECTION (Rank 0 only)                      │
└─────────────────────────────────────────────────────────────────┘

Input: all_reps [N × D]     (N representations, D dimensions)
       max_samples (K)      (desired subset size)
       method (string)      (algorithm choice)

┌─────────────────────────────────────────────────────────────────┐
│ IF method == "facility_location":                              │
│                                                                  │
│  1. Compute similarity matrix:                                  │
│     S = cosine_similarity(all_reps, all_reps)  [N × N]        │
│                                                  │
│  2. Facility location greedy:                    │
│     FOR i in 1..K:                               │
│       Select center c_i with max marginal value │
│       Mark samples assigned to c_i                │
│                                                  │
│  3. Weight by cluster size:                      │
│     w_j = |cluster(j)| / N                       │
│                                                  │
│  Output: selected_idx[], selected_weights[]      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ ELIF method == "greats":                                        │
│                                                                  │
│  1. Compute gradient norms:                                     │
│     norms = ||all_reps[i]||  [N]               │
│                                                │
│  2. Compute similarity matrix:                 │
│     S = cosine_similarity(all_reps, all_reps)  │
│                                                │
│  3. Greedy selection:                          │
│     FOR i in 1..K:                              │
│       score[j] = norms[j] - λ·S[j, selected] │
│       Select argmax(score)                      │
│                                                  │
│  Output: selected_idx[]                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ ELIF method == "fairot":                                        │
│                                                                  │
│  1. Compute similarity matrix:                                  │
│     S = cosine_similarity(all_reps, all_reps)  │
│                                                │
│  2. Extract fairness constraint:                │
│     dist = distribution_per_source              │
│                                                │
│  3. Solve regularized OT problem:              │
│     minimize: -S·x + reg·W(dist, x)            │
│     subject to: sum(x) = K                      │
│                 0 ≤ x ≤ 1                       │
│                                                │
│     (via greedy iterations, ~500 iters)        │
│                                                │
│  Output: selected_idx[]                         │
└─────────────────────────────────────────────────────────────────┘

All outputs then:
  - Broadcast to all ranks
  - Used for training batch creation
```

### 7.4 Riemannian Manifold Update Detail

```
┌─────────────────────────────────────────────────────────────────┐
│     RIEMANNIAN PARAMETER UPDATE (per layer)                    │
└─────────────────────────────────────────────────────────────────┘

State: A ∈ St(m, r), B ∈ ℝ^(n, r)  [on fixed-rank manifold]
       A.shape = [2r, d_in]
       B.shape = [d_out, 2r]

Gradient: ∂L/∂A, ∂L/∂B  [computed via backprop]

┌─────────────────── STEP 1: EXTRACT TANGENT VECTOR ──────────────┐
│                                                                  │
│ Extract deltas (gradients from extended representation):        │
│   dot_A = (∂L/∂A).T from rows [r:2r]  (skip orthogonal part)  │
│   dot_B = (∂L/∂B).T from rows [0:r]                            │
│                                                                  │
│ Project to Euclidean subspace:                                  │
│   A ← A.T[:, :r]        [actual orthogonal part]               │
│   B ← B.T[:r, :]                                                │
│   point = [A, B]                                                │
│                                                                  │
│ Project gradient to Riemannian subspace:                        │
│   dot_A ← dot_A - A @ (A.T @ dot_A)  [orthogonalize]          │
│   vec = [dot_A, dot_B]                                          │
│                                                                  │
│ Output: point, vec  [point on manifold, tangent vector]        │
└──────────────────────────────────────────────────────────────────┘

┌─────────────── STEP 2: APPLY FIRST MOMENTUM ─────────────────────┐
│                                                                   │
│ velocity[t] = β₁ · velocity[t-1] + (1-β₁) · vec                 │
│             (with parallel transport for consistency)             │
│                                                                   │
│ Output: momentum_vec  [accumulated velocity]                      │
└──────────────────────────────────────────────────────────────────┘

┌──────────── STEP 3: APPLY SECOND MOMENTUM (ADAPTIVE) ────────────┐
│                                                                   │
│ norm[A] = ||dot_A||²                                             │
│ norm[B] = ||dot_B||²                                             │
│                                                                   │
│ second_momentum[A] = β₂·second_mom[A] + (1-β₂)·norm[A]          │
│ second_momentum[B] = β₂·second_mom[B] + (1-β₂)·norm[B]          │
│                                                                   │
│ adaptive_lr[A] = 1 / (√second_momentum[A] + ε)                   │
│ adaptive_lr[B] = 1 / (√second_momentum[B] + ε)                   │
│                                                                   │
│ dot_A ← momentum_vec[A] · adaptive_lr[A]  [adaptive step]        │
│ dot_B ← momentum_vec[B] · adaptive_lr[B]                         │
│                                                                   │
│ Output: [dot_A, dot_B]  [adaptive momentum step]                │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────── STEP 4: RETRACTION ──────────────────────────┐
│                                                                   │
│ Map from tangent space back to manifold:                         │
│                                                                   │
│ A_part, B_part = embedding([A, B], [-lr·dot_A, point[1]-lr·dot_B], ambient=False)
│                                                                   │
│ [QR decompositions]                                              │
│ Q_l, R_l = qr(A_part)                                            │
│ Q_r, R_r = qr(B_part.T)                                          │
│ T = R_l @ R_r.T                                                  │
│                                                                   │
│ [SVD to ensure rank-r structure]                                 │
│ U, S, Vt = svd(T)                                                │
│ U = U[:, :r]   [top r components]                                │
│ S = S[:r]                                                        │
│ Vt = Vt[:r, :]                                                   │
│                                                                   │
│ [Construct new point on manifold]                                │
│ new_A = Q_l @ U                [m × r orthogonal]               │
│ new_B = diag(S) @ Vt @ Q_r.T   [n × r with appropriate norm]   │
│                                                                   │
│ Output: [new_A, new_B]  [guaranteed on manifold]                │
└──────────────────────────────────────────────────────────────────┘

┌──────────── STEP 5: MAINTAIN ORTHOGONAL STRUCTURE ────────────────┐
│                                                                   │
│ Extend back to doubled representation:                           │
│                                                                   │
│ A_zero = zeros_like(new_A)        [r × d_in]                    │
│ A_new = [new_A; A_zero]           [2r × d_in]                   │
│                                                                   │
│ B_orth = qr(new_B.T).Q.T          [d_out × r orthogonal]        │
│ B_new = [new_B; B_orth]           [d_out × 2r]                  │
│                                                                   │
│ Update parameters:                                                │
│ weight[A].copy_(A_new)                                           │
│ weight[B].copy_(B_new)                                           │
│                                                                   │
│ ✓ Parameters remain on fixed-rank manifold                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 8. Technical Insights and Tradeoffs

### 8.1 CoLM Strengths

1. **Data Efficiency**
   - Trains only on informative samples
   - Reduces effective batch size
   - Can handle larger full batches for selection

2. **Generalization**
   - Coreset selection often improves generalization
   - Diverse subset selection (facility location, FairOT)
   - Handles multi-source data with fairness constraints

3. **Flexibility**
   - Multiple selection algorithms to choose from
   - Easy to add new selection methods
   - Per-source fairness control

4. **Existing Infrastructure**
   - Leverages HF transformers
   - Standard PEFT LoRA
   - Standard optimizers (AdamW)

### 8.2 CoLM Limitations

1. **Selection Overhead**
   - Representation extraction for all samples
   - Selection algorithm computation (~0.1s per step)
   - Synchronization across distributed GPUs
   - CPU-GPU data transfers

2. **Memory Complexity**
   - Still needs to load representations for all samples
   - Gathers all representations to rank 0
   - Synchronization points

3. **Convergence**
   - Selection introduces stochasticity
   - May require more training steps to converge
   - Hyperparameter tuning for batch sizes

### 8.3 Riemannian Strengths

1. **Theoretical Foundation**
   - Manifold-aware optimization
   - Geodesically optimal updates
   - Provable convergence on Riemannian spaces

2. **Parameter Efficiency**
   - Fixed-rank constraints enforced by geometry
   - No rank drift during training
   - Theoretically motivated initialization

3. **Stability**
   - Orthogonal constraints prevent numerical issues
   - Structured updates via retraction
   - Cleaner mathematical framework

4. **Simplicity**
   - Standard training loop
   - No data selection complexity
   - Fewer moving parts

### 8.4 Riemannian Limitations

1. **Memory Overhead**
   - Doubled LoRA rank due to manifold representation
   - More parameters than standard LoRA
   - ~2× memory for LoRA weights

2. **Computational Cost**
   - QR decompositions per optimization step
   - SVD computation per layer
   - Additional manifold operations

3. **Implementation Complexity**
   - Requires custom optimizers
   - Complex mathematical machinery
   - Fewer practitioners familiar with Riemannian optimization

4. **Limited Evidence**
   - Smaller codebase/fewer experiments
   - Less community adoption
   - Theoretical benefits may not always translate empirically

### 8.5 Comparison: When to Use What

#### Use CoLM When:
- Training on large, heterogeneous datasets
- Dataset contains redundant or noisy samples
- Multi-source data with fairness requirements
- Can afford batch-wise selection computation
- Prefer empirically validated approach

#### Use Riemannian When:
- Want mathematically principled optimization
- Have tight memory constraints (relative to rank doubling)
- Trust Riemannian geometry theory
- Can afford manifold operations overhead
- Prefer cleaning implementation

---

## 9. Practical Considerations

### 9.1 Setup and Requirements

#### CoLM Setup
```bash
# Prerequisites
torch==2.2.1
transformers==4.43.2
peft (for LoRA)
vllm (for eval)
traker (for coreset selection)
submodlib (for facility location, LESS integration)

# GPU Memory: ~24GB for Phi-2 fine-tuning
# Training time: see README (overhead ~0.1s/step selection)
```

#### Riemannian Setup
```bash
# Prerequisites
torch (recent)
transformers
trl (SFTTrainer)
peft
omegaconf (config management)

# GPU Memory: ~32GB+ (due to doubled LoRA rank)
# Training time: standard + ~1-3% manifold overhead
```

### 9.2 Configuration Strategies

#### CoLM Configuration
```python
# Key hyperparameters affecting performance:
small_batch_ratio: 0.5      # Selection ratio
gradient_accumulation_steps: 128
data_selection_method: "greats"  # or "fairot", "facility_location"
data_selection_unit: "masked_grad"  # or "rep", "mezo"

# Tuning advice:
- Smaller small_batch_ratio → more aggressive selection → faster but less accurate
- Larger gradient_accumulation_steps → more samples for selection → less efficient
```

#### Riemannian Configuration
```yaml
# Key hyperparameters:
adapter_config:
  ft_strategy: 'LoRA'
  LoRA_config:
    r: 64              # Base rank
    lora_alpha: 128
    
optimizer_config:
  optim: 'LoRA_Riemanian'  # or 'SGD_Riemannian'
  lr: 5e-4
  betas: [0.9, 0.999]

trainer_config:
  num_train_epochs: 3
  per_device_train_batch_size: 32
  max_steps: 10000

# Tuning advice:
- Larger r → more expressive → more manifold operations
- Choose between RiemannianLora (adaptive) vs SGD (non-adaptive)
```

---

## 10. Conclusion

### 10.1 Summary of Approaches

**CoLM (Data-Centric):**
- Solves: which samples to train on
- Method: intelligent coreset selection per batch
- Innovation: fair multi-source selection with minimal overhead
- Best for: large, mixed-quality datasets

**Riemannian (Geometry-Centric):**
- Solves: how to optimize LoRA parameters
- Method: manifold-aware optimization on fixed-rank spaces
- Innovation: theoretically principled Riemannian updates
- Best for: mathematically consistent optimization

### 10.2 Complementarity

These approaches are **orthogonal** and potentially **combinable**:
```
CoLM selects WHICH samples → Data selection
Riemannian optimizes HOW → Parameter optimization

Potential future work: CoLM + Riemannian
- Select samples via CoLM
- Update selected samples via Riemannian optimizer
- Best of both worlds?
```

### 10.3 Research Directions

**For CoLM:**
- Faster selection algorithms (neural network approximations)
- Online/streaming selection without gathering
- Adaptive selection ratios based on convergence signals

**For Riemannian:**
- Scalable manifold operations (hierarchical or approximate retraction)
- Theoretical convergence analysis for LLM fine-tuning
- Combination with other constraints (sparsity, compression)
