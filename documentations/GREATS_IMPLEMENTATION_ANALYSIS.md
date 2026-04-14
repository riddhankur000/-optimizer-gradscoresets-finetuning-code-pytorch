# GREATS Method Implementation Analysis
## Gradient Matching-Based Subset Selection

---

## Executive Summary

The GREATS method ("Gradient Regularized Training with Subset Selection") is implemented across three main components in the codebase:

1. **Gradient Computation** (Lines 1171-1250 in `subset_trainer_distributed.py`): `save_select()` method
2. **Similarity Computation** (Lines 226-271 in `utils.py`): `compute_cost_matrix()` function  
3. **Greedy Selection** (Lines 1-36 in `greats.py`): `greedy_selection()` function

---

## 1. Component 1: Gradient Computation - `save_select()` Method

**File**: `llm-experiments/colm/train/subset_trainer_distributed.py`  
**Lines**: 1171-1250 (for MEZO mode)

### Purpose
Computes gradient representations for each training sample using Zero-Order Optimization (MEZO).

### Implementation Flow

#### MEZO Gradient Computation (Lines 1188-1210)
```python
def save_select(self, model, inputs):
    elif self.args.data_selection_unit == "mezo":
        # Step 1: Prepare parameters to optimize
        if len(self.named_parameters_to_optim) == 0:
            for name, param in model.named_parameters():
                if any(substring in name for substring in self.last_layers):
                    self.named_parameters_to_optim.append((name, param))

        # Step 2: Compute finite difference gradient (Zero-Order)
        # Forward: +ε perturbation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        
        # Backward: -ε perturbation
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        
        # Compute directional derivative: (f(x+ε) - f(x-ε)) / 2ε
        projected_grad = ((loss1 - loss2) / (2 * (self.args.mezo_eps))).item()
        
        # Restore parameters
        self.zo_perturb_parameters(scaling_factor=1)
```

**UTILITY FUNCTION FOR GRADIENT MATCHING**:
```python
        # Step 3: Generate random direction and compute projected gradient
        torch.manual_seed(self.zo_random_seed)  
        
        res_list = []
        for _, (name, param) in enumerate(self.named_parameters_to_optim):
            # Sample random direction z ~ N(0, I)
            z = torch.normal(mean=0, std=1, size=param.data.size(), 
                           device=param.data.device, dtype=param.data.dtype)
            
            # Compute gradient vector: projected_grad × z
            # This is the UTILITY FUNCTION: g_sample = gradient_direction × random_vector
            grad_update = projected_grad * z
            
            # Optional: Apply weight-based scaling for importance weighting
            if self.args.mezo_selection == "weight_grad" and not torch.all(param.data == 0):
                grad_update = grad_update * param.data  # Element-wise multiplication
            
            flattened_res = grad_update.flatten()
            res_list.append(flattened_res)
        
        # Concatenate all layer gradients into single vector
        res = torch.cat(res_list, dim=0).flatten()
```

**Key Points**:
- `projected_grad`: Scalar directional derivative computed via finite differences
- `z`: Random Gaussian noise (one sample per batch item)
- `grad_update = projected_grad × z`: **This is the gradient representation per sample**
- `res`: Final gradient vector representation for similarity computation

---

## 2. Component 2: Similarity Computation - `compute_cost_matrix()`

**File**: `llm-experiments/colm/train/utils.py`  
**Lines**: 226-271

### Purpose
Computes pairwise similarity/distance matrices between gradient representations.

### Implementation

```python
def compute_cost_matrix(X_source: torch.Tensor, X_target: torch.Tensor, 
                       metric: str = "euclidean", return_sims=False) -> torch.Tensor:
    """
    X_source: (n, d)        # Training samples' gradient representations
    X_target: (m, d)        # Validation/reference samples' gradient representations
    metric: similarity metric for computing similarity matrix
    return_sims: if True, returns both cost matrix D and similarity matrix S
    """
    
    if metric == "cosine":
        # Normalize vectors to unit length
        X1 = F.normalize(X_source, dim=1)  # (n, d) normalized
        X2 = F.normalize(X_target, dim=1)  # (m, d) normalized
        
        # Compute cosine similarity: S[i,j] = <X1[i], X2[j]>
        cosine_sim = X1 @ X2.T  # (n, m)
        
        # Cost matrix: D[i,j] = 1 - similarity (distance)
        D = 1 - cosine_sim
        S = cosine_sim
    
    # Handle NaN values
    if torch.isnan(S).sum() > 0:
        S = torch.nan_to_num(S, nan=-0.95)
    
    if return_sims:
        return D, S  # Returns both distance and similarity matrices
    return D
```

**Matrices Returned**:
- `S`: (n, m) similarity matrix - used in GREATS for greedy selection
- `D`: (n, m) distance matrix - alternative representation

---

## 3. Component 3: Greedy Selection - `greedy_selection()`

**File**: `llm-experiments/colm/train/greats.py`  
**Lines**: 1-36

### Purpose
Selects K most useful samples using greedy algorithm with gradient matching.

### Implementation

```python
def greedy_selection(scores, interaction_matrix, K):
    """
    The UTILITY FUNCTION for greedy subset selection based on:
    
    GREATS Objective:
    max_{S} [∑_{i∈S} scores[i] - λ * ∑_{i,j∈S} interaction_matrix[i,j]]
    
    Parameters:
    -----------
    scores: (n,)
        - Initial utility scores (gradient relevance per sample)
        - Computed as: mean(sims_cross[i,:]) 
        - Where sims_cross[i,j] = similarity(train_sample_i, validation_sample_j)
        - Represents how well sample i's gradient matches validation gradient
    
    interaction_matrix: (n, n)
        - Pairwise gradient similarity between training samples
        - sims[i,j] = cosine_similarity(grad_i, grad_j)
        - High similarity = high interaction cost (redundancy penalty)
    
    K: int
        - Number of samples to select
    
    Returns:
    --------
    selected_indices: list
        - Indices of K selected samples
    """
    
    # Make copies to avoid in-place modification issues
    scores = scores.copy()
    interaction_matrix = interaction_matrix.copy()
    selected_indices = []
    K = min(K, scores.shape[0])
    
    # GREEDY ALGORITHM WITH GRADIENT-BASED UTILITY FUNCTION
    for iteration in range(K):
        # STEP 1: Find sample with highest current utility score
        idx_max = np.argmax(scores)  # argmax_i scores[i]
        selected_indices.append(idx_max)
        
        # STEP 2: UPDATE UTILITY SCORES (Gradient Interaction Penalty)
        # For each remaining sample j, reduce its score by its similarity
        # with the newly selected sample:
        # scores[j] -= interaction_matrix[idx_max, j]
        # 
        # INTUITION: If sample j is very similar (in gradient space) to 
        # the already-selected sample idx_max, we penalize it to avoid 
        # redundancy in the subset.
        scores -= interaction_matrix[idx_max, :]  # Subtract interaction costs
        
        # STEP 3: ENSURE SELECTED SAMPLE NOT CHOSEN AGAIN
        # Mark selected sample as "used" by setting its score to -∞
        scores[idx_max] = -np.inf
    
    return selected_indices
```

### Algorithm Pseudocode

```
GREATS_Greedy_Selection(scores, interactions, K):
    selected = []
    S = scores.copy()                    // Utility scores per sample
    I = interactions.copy()              // Pairwise gradient similarities
    
    for iteration = 1 to K:
        i* = argmax_i S[i]               // Find best sample based on current score
        selected ← selected ∪ {i*}       // Add to selection
        
        // Penalize scores based on gradient interaction with newly selected sample
        for j in remaining samples:
            S[j] ← S[j] - I[i*, j]       // Reduce score by similarity penalty
        
        S[i*] ← -∞                       // Ensure i* not selected again
    
    return selected
```

---

## 4. Data Flow Integration into Training

**Location**: Lines 800-850 in `subset_trainer_distributed.py`

### Training Loop Integration

```python
# STEP 1: Collect gradient representations for all training samples
for batch in train_dataloader:
    reps = self.save_select(model, batch)  # Compute gradient representation
    all_reps.append(reps)                  # Accumulate

# STEP 2: Compute similarity matrices
_, sims = utils.compute_cost_matrix(all_reps, all_reps, 
                                    metric="cosine", return_sims=True)
# sims[i,j] = cosine_similarity(gradient_i, gradient_j)

# STEP 3: Compute cross-sample scores (gradient matching to validation set)
eval_reps = inputs  # Validation/reference gradient set
_, sims_cross = utils.compute_cost_matrix(all_reps, eval_reps, 
                                          metric="cosine", return_sims=True)
# sims_cross[i,j] = similarity(train_grad_i, val_grad_j)

# STEP 4: Call GREATS selection
scores = sims_cross.mean(1)  # Average validation matching per train sample
selected_idx = greats.greedy_selection(scores, sims, max_samples)
# Inputs to greedy_selection:
# - scores: how well each training gradient matches validation gradient
# - sims: pairwise similarity between training gradients (interaction penalty)
```

---

## 5. Mathematical Formulation

### GREATS Utility Function

**Objective**: 
$$\max_S \left[ \sum_{i \in S} \text{score}_i - \lambda \sum_{i,j \in S, i \neq j} \text{sim}(g_i, g_j) \right]$$

Where:
- $\text{score}_i = \frac{1}{|V|} \sum_{j \in V} \text{sim}(g_i^{\text{train}}, g_j^{\text{val}})$
  - Average gradient matching to validation set
  - **GRADIENT MATCHING TERM**: How well training sample's gradient aligns with validation
  
- $\text{sim}(g_i, g_j) = \frac{g_i \cdot g_j}{||g_i|| \cdot ||g_j||}$
  - Cosine similarity between gradients
  - **REDUNDANCY PENALTY**: Penalizes similar gradients in selected subset

### Gradient Representation Per Sample

For MEZO mode:
$$g_i = \text{projected\_grad} \times z$$

Where:
- $\text{projected\_grad} = \frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}$ (directional derivative)
- $z \sim \mathcal{N}(0, I)$ (random Gaussian direction)
- Both multiplied element-wise across selected parameter layers

---

## 6. Code Location Summary Table

| Component | File | Lines | Function | Purpose |
|-----------|------|-------|----------|---------|
| **Gradient Computation** | `subset_trainer_distributed.py` | 1171-1210 | `save_select()` | Compute gradient representation (MEZO) |
| **Similarity Matrix** | `utils.py` | 226-271 | `compute_cost_matrix()` | Compute cosine similarity matrices |
| **Greedy Selection Algorithm** | `greats.py` | 1-36 | `greedy_selection()` | Select K samples using greedy algorithm |
| **Selection Integration** | `subset_trainer_distributed.py` | 1370-1382 | `select_data()` (GREATS branch) | Orchestrate GREATS selection |
| **Training Loop Call** | `subset_trainer_distributed.py` | 823-825 | `_inner_training_loop()` | Call select_data during training |

---

## 7. Key Parameters

| Parameter | Where Set | Default | Purpose |
|-----------|-----------|---------|---------|
| `data_selection_method` | TrainingArguments | "none" | Set to "greats" to enable GREATS |
| `data_selection_unit` | TrainingArguments | "rep" | Set to "mezo" for gradient-based selection |
| `mezo_eps` | TrainingArguments | 1e-3 | Perturbation size for finite differences |
| `mezo_selection` | TrainingArguments | "grad" | "grad" or "weight_grad" for weighting |
| `last_layers` | TrainingArguments | [] | Which layers to compute gradients from |
| `zo_dim` | TrainingArguments | -1 | Dimension for random projection (-1 = full) |

---

## 8. Execution Flow Diagram

```
Training Loop (_inner_training_loop)
    ↓
For each batch:
    ├─→ save_select(model, batch)
    │   ├─→ zo_perturb_parameters(+ε)
    │   ├─→ loss1 = zo_forward(model, inputs)
    │   ├─→ zo_perturb_parameters(-2ε)
    │   ├─→ loss2 = zo_forward(model, inputs)
    │   ├─→ projected_grad = (loss1 - loss2) / 2ε
    │   ├─→ z ~ N(0, 1)
    │   ├─→ grad_update = projected_grad × z
    │   └─→ RETURN: gradient_representation
    │
    └─→ Accumulate all_reps (gradient representations)

After collecting all samples:
    ├─→ compute_cost_matrix(all_reps, all_reps, metric="cosine")
    │   └─→ RETURN: sims[i,j] = gradient_similarity(i, j)
    │
    ├─→ compute_cost_matrix(all_reps, val_reps, metric="cosine")
    │   └─→ RETURN: sims_cross[i,j] = gradient_match(train_i, val_j)
    │
    ├─→ select_data(all_reps) [GREATS branch]
    │   ├─→ scores = sims_cross.mean(1)  # Gradient matching scores
    │   ├─→ greedy_selection(scores, sims, max_samples)
    │   │   ├─→ FOR k=1 to K:
    │   │   │   ├─→ i* = argmax(scores)  # Best gradient match
    │   │   │   ├─→ scores -= sims[i*, :]  # Redundancy penalty
    │   │   │   └─→ scores[i*] = -∞
    │   │   └─→ RETURN: selected_indices
    │   └─→ RETURN: selected indices
    │
    └─→ Filter batch to selected samples
         └─→ Continue training with subset
```

---

## 9. Key Insights

### Gradient Matching Mechanism

1. **What is matched**: Gradients of training samples vs. validation samples
   - Training gradient: Directional derivative via finite differences (MEZO)
   - Validation gradient: Gradients on validation set
   - Matching: Cosine similarity in gradient space

2. **Why gradient matching**: 
   - Samples whose gradients align well with validation set are more informative
   - Reduces overfitting by selecting samples that help on held-out data

3. **Redundancy penalty**:
   - After selecting a sample, reduce scores of similar samples
   - Avoids selecting multiple samples with redundant gradient information
   - Enforces diversity in gradient space

### MEZO Gradient Representation

Instead of full backprop, GREATS uses:
- **Finite difference approximation**: $(f(x+\epsilon) - f(x-\epsilon)) / 2\epsilon$
- **Random projection**: Scale the gradient by random Gaussian vector
- **Efficiency**: ~1 forward + 2 backward passes per sample vs. full backprop

---

## 10. Related Concepts

### Why Gradient Matching for Selection

Papers cited in the method:
- **GREATS**: Uses gradient alignment with validation set
- **CoreSet**: Uses representativeness (coverage in feature space)
- **BALD**: Uses uncertainty (entropy of gradient predictions)
- **Facility Location**: Uses diversity in feature space

GREATS is unique because:
- ✅ Directly optimizes for validation performance
- ✅ Considers gradient space (parameter importance)
- ✅ Handles redundancy via interaction penalty
- ✅ Computationally efficient (MEZO approximation)

---

## 11. Theor Validation

The greedy algorithm achieves approximately $(1 - 1/e)$ of the optimal objective value for submodular functions because:

- **Diminishing returns property**: Selecting sample i reduces future utility of similar samples
- **Greedy guarantee**: Each iteration selects highest marginal gain
- **Submodular approximation**: The objective is approximately submodular in gradient space

Formula:
$$\text{OPT} - \text{GREEDY} \leq e^{-1} \cdot \max_i \text{score}_i$$

---

## Summary

**The GREATS utility function for gradient matching is implemented across three layers:**

1. **Gradient Computation Layer** (`save_select()`): 
   - Computes `projected_grad = (loss1 - loss2) / 2ε`
   - Samples random direction `z ~ N(0,I)`
   - **Utility vector**: `grad_update = projected_grad × z`

2. **Similarity Layer** (`compute_cost_matrix()`):
   - Computes cosine similarity: `sims[i,j] = cosine_sim(grad_i, grad_j)`
   - Computes cross-similarity: `sims_cross[i,j] = cosine_sim(train_i, val_j)`

3. **Selection Layer** (`greedy_selection()`):
   - **Utility function**: `scores[i] = mean(sims_cross[i, :])`
   - **Greedy selection**: Iteratively pick highest score, penalize by interaction
   - **Penalty function**: `scores -= interaction_matrix`

The combination creates a gradient-space aware data selection method that maximizes validation performance while ensuring diversity.
