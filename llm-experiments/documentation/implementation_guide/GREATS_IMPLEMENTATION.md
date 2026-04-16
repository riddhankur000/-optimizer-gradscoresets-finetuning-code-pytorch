# GREATS: Implementation Details

> **Paper**: "GREATS: Online Selection of High-Quality Data for LLM Training in Every Iteration" (NeurIPS 2024)

---

## What GREATS Does (One Line)

At each training step, instead of training on a random mini-batch, GREATS selects the K examples from a larger candidate batch that will maximally reduce validation loss — using a greedy algorithm with Taylor approximation.

---

## Core Mathematical Idea

The utility of a subset S for model update at step t is:

```
U(S; z_val) = ℓ(w_t, z_val) - ℓ(w_t - η * Σ_{z∈S} ∇ℓ(w_t, z), z_val)
```

Directly evaluating this requires updating the model for every candidate subset — exponentially expensive. GREATS approximates it with two Taylor expansions:

**First-order approximation of marginal gain of adding z_i to already-selected set B̂:**

```
U(z_i | B̂) ≈ η * g(z_i) · g(z_val)              ← TracIN score
             - η² * g(z_i) · H(z_val) · Σ_{z∈B̂} g(z)  ← Hessian correction
```

Where:
- `g(z_i) = ∇ℓ(w_t, z_i)` — gradient of training example
- `g(z_val) = ∇ℓ(w_t, z_val)` — gradient of validation example
- `H(z_val) = ∇²ℓ(w_t, z_val)` — Hessian at validation point
- `H ≈ I` (identity approximation) simplifies to `g(z_i) · g(z*)`

---

## Algorithm (from `greats.py`)

```python
def greedy_selection(scores, interaction_matrix, K):
    # scores:             [train_bs, val_bs]  — initial TracIN scores
    # interaction_matrix: [train_bs, train_bs] — pairwise gradient inner products
    # K:                  number to select
    
    scores = scores.copy()
    selected = []

    for _ in range(K):
        idx_max = np.argmax(scores)        # pick highest TracIN score
        selected.append(idx_max)
        scores -= interaction_matrix[idx_max, :]  # subtract interactions (Hessian correction)
        scores[idx_max] = -np.inf          # never select again

    return selected
```

**The key insight**: After selecting `z*`, subtracting `interaction_matrix[idx_max, :]` from all remaining scores is equivalent to computing `U(z_i | {z*}) = ϕ_i - η² g(z_i) · g(z*)` — the corrected marginal gain for each remaining candidate.

---

## How It's Called in `subset_trainer_distributed.py`

```python
# Inside select_data(), when method == "greats":
_, sims = compute_cost_matrix(inputs, inputs, metric="cosine", return_sims=True)
eval_reps = inputs   # using training reps as proxy for validation
_, sims_cross = compute_cost_matrix(inputs, eval_reps, metric="cosine", return_sims=True)

idx = greats.greedy_selection(
    tocpu(sims_cross.mean(1)),  # scores:             [N] — mean sim to "validation"
    tocpu(sims),                # interaction_matrix: [N, N] — pairwise similarities
    max_samples
)
```

**Note**: In the actual codebase, the cross-similarity to sampled validation examples is used as the TracIN score proxy, and pairwise training similarities as the interaction/Hessian term.

---

## Ghost Inner-Product (from Paper — Efficiency Technique)

The paper describes a "ghost inner-product" technique to compute all pairwise gradient inner products in one backpropagation pass. For a linear layer `s = aW`:

```
∂ℓ(i)/∂W · ∂ℓ(j)/∂W = (b(i) · b(j)) * (a(i) · a(j))
```

Where `a(i)` is the layer input and `b(i) = ∂ℓ/∂s(i)` is the output gradient — both available during a single backward pass.

**In the codebase**: The `data_selection_unit = "masked_grad"` path in `save_select` uses actual backprop gradients of the last layer. The ghost inner-product is not explicitly coded as a separate module — instead, the framework computes per-sample gradient vectors and then `compute_cost_matrix` handles the pairwise similarities.

---

## Full Data Flow for GREATS

```
Large batch Bt (B examples)
    ↓
save_select (mezo or masked_grad) → all_reps [B, full_dim]
    ↓
select_masking → sparsify to [B, zo_dim]  (keep top-zo_dim dims by magnitude)
    ↓
compute_cost_matrix(reps, reps, "cosine") → sims [B, B]  (interaction_matrix)
compute_cost_matrix(reps, eval_reps, "cosine") → sims_cross [B, B]  (TracIN scores)
    ↓
greedy_selection(sims_cross.mean(1), sims, K)
    → iteratively argmax + subtract interactions
    → returns K selected indices
    ↓
train on selected K examples
```

---

## Score Initialization vs. Update

| Step | Formula | Meaning |
|------|---------|---------|
| Initialize | `ϕ_z = η * g(z) · g(z_val)` | How much does training on z reduce validation loss |
| After selecting z* | `ϕ_z -= η² * g(z) · g(z*)` | Penalize z for being similar to already-selected z* |
| z* score | `ϕ_{z*} = -∞` | Never re-select |

The subtraction step is what makes GREATS different from simple top-K selection — it accounts for **redundancy** among selected examples.

---

## Hessian Approximations in Algorithm 1

The paper offers two variants (from `Algorithm 1` in the paper):

```
HessianApprox = "exact":
    ϕ_z -= η² * g(z) · H(z_val) · g(z*)   # actual Hessian-vector product

HessianApprox = "identity":
    ϕ_z -= η² * g(z) · g(z*)               # H ≈ I (what the codebase uses via cosine sim)
```

The identity approximation is used in practice because computing the Hessian is expensive and the approximation works well empirically (confirmed by Figure 1 in the paper showing Pearson correlation ≈ 0.84 with Hessian vs ≈ 0.76 without).

---

## Runtime: Why It's Fast

| Method | Per-iteration cost |
|--------|-------------------|
| Direct greedy | O(k * |Bt|) model updates + validation loss evals |
| GREATS (naive) | O(k * |Bt|) backprops (per-sample gradients) |
| GREATS (ghost) | ~1 backprop (ghost inner-product) + O(k * |Bt|) score updates |

The ghost inner-product makes GREATS run at nearly the same speed as regular training (Table 3 in paper: 71.3 vs 76.2 throughput).

---

## What Validation Data Is Used

From `training_arguments.py` and the training scripts — GREATS requires a small validation set (`≤ 16` examples in paper experiments). In the codebase:

```python
# In SubsetTrainer.sample_k_random_items:
def sample_k_random_items(self, k, seed=None):
    indices = random.sample(range(len(self.eval_dataset)), k)
    samples = [self.eval_dataset[i] for i in indices]
    return default_data_collator(samples)
```

The eval_dataset passed to the trainer serves as the validation set for GREATS scoring.

---

## Experiments in Paper vs. Codebase

| Paper Setting | Codebase Equivalent |
|---------------|-------------------|
| LLAMA-2-7B + LESS → MMLU | `lora_train_math.sh` with appropriate model/data args |
| GPT-SMALL pretraining | `training_arguments.py` with `data_selection_method="greats"` |
| 5-16 validation points | `sample_k_random_items(k=5)` or similar |
| Identity Hessian approx | `sims` matrix (cosine similarities) as interaction term |
