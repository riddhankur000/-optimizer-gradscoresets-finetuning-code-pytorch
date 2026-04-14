# CoLM: Implementation Details

> **Paper**: "Mini-Batch Coresets for Memory-Efficient Language Model Training on Data Mixtures" (ICLR 2025)

---

## What CoLM Does (One Line)

CoLM finds small mini-batch coresets that match the gradient of larger random batches — handling imbalanced data sources, Adam optimizer, and high-dimensional LLM gradients simultaneously.

---

## Three Core Challenges & Solutions

| Challenge | Problem | CoLM's Solution |
|-----------|---------|----------------|
| Imbalanced data | Small sources get no medoids | Always include all small-source examples |
| Adam optimizer | Vanilla gradient matching suboptimal | Normalize by historical exponential averages |
| High-dim gradients | Distances vacuous in high dim | Zeroth-order gradient + sparsification to top-h dims |

---

## Challenge 1: Imbalanced Sources

### Theory (Theorem 4.1 & 4.2 in paper)
For a source with too few examples in each random batch, medoids found by gradient matching are NOT representative of that source with high probability.

### Implementation in `subset_trainer_distributed.py`

**Identifying small sources** — done in `train.py` at startup:
```python
# From training_arguments.py:
keep_sources = "0_1_3_5_7_8_9_10_11_13"  # small source indices for MathInstruct

# In train.py:
training_args.keep_sources = [int(idx) for idx in training_args.keep_sources.split('_')]
```

**Separating small-source examples** — in `_inner_training_loop` on rank 0:
```python
list_idx_keep = []
include_in_selection = []

for idx in sampling_indices:
    if complete_input_list[idx]["sources"][0] in self.args.keep_sources:
        include_in_selection.append(False)
        list_idx_keep.append(idx)      # always include these
    else:
        include_in_selection.append(True)  # go through selection

max_samples -= len(list_idx_keep)      # reduce budget for big-source selection
all_reps = all_reps[include_in_selection]     # only big-source reps go to selector
sampling_indices = sampling_indices[include_in_selection]

# After selection:
selected_idx = list_idx_keep + sampling_indices[selected_big_source_idx].tolist()
```

**Uniform weighting** — all selected examples get weight 1:
```python
selected_weights = torch.tensor(
    [1 for _ in range(len(list_idx_keep))] + selected_weights.tolist()
)
```

---

## Challenge 2: Adam Optimizer Normalization

### Theory
Adam scales gradient dimensions by `m / (sqrt(v) + ε)`. Matching vanilla gradients is suboptimal — we should match normalized gradients.

### Implementation in `_inner_training_loop`

```python
if self.args.mezo_optim == "adam":
    all_reps_squared = torch.square(all_reps)

    # Get or initialize historical terms
    if 'exp_avg' in self.optimizer.state[param]:
        # If using backprop gradients, get directly from optimizer state
        prev_m_t = torch.cat([self.optimizer.state[p]['exp_avg'].flatten() ...])
        prev_v_t = torch.cat([self.optimizer.state[p]['exp_avg_sq'].flatten() ...])
    else:
        # For MeZO, track manually
        prev_m_t = self.prev_m_t  # zeros on first step
        prev_v_t = self.prev_v_t

    # Adam update equations
    m_t = args.adam_beta1 * prev_m_t + (1 - args.adam_beta1) * all_reps
    v_t = args.adam_beta2 * prev_v_t + (1 - args.adam_beta2) * all_reps_squared
    m_hat = m_t / (1 - args.adam_beta1 ** (global_step + 1))   # bias correction
    v_hat = v_t / (1 - args.adam_beta2 ** (global_step + 1))
    
    adam_updates = m_hat / (torch.sqrt(v_hat) + args.adam_epsilon)
    all_reps = adam_updates   # use normalized grads for subset selection

    # After selection — update historical terms with selected subset's mean
    if "grad" not in self.args.data_selection_unit:
        self.prev_m_t = m_t[selected_idx].mean(dim=0).detach()
        self.prev_v_t = v_t[selected_idx].mean(dim=0).detach()
```

**Key detail**: Historical terms `m, v` are computed only from **big-source** examples (since small sources are excluded from `all_reps` before this computation).

---

## Challenge 3: High-Dimensional Gradients

### Zeroth-Order Gradient Estimation (MeZO)

Instead of backpropagation, CoLM uses SPSA (Simultaneous Perturbation Stochastic Approximation):

```
ĝ = [L(θ + ε·z) - L(θ - ε·z)] / (2ε) * z  ≈ z·z^T·g
```

This gives a **rank-1 reconstruction** of the gradient that is smoother than the actual backprop gradient.

**In `save_select` with `data_selection_unit="mezo"`:**

```python
# Set up target layer parameters
for name, param in model.named_parameters():
    if any(substring in name for substring in self.last_layers):
        self.named_parameters_to_optim.append((name, param))

# +ε perturbation
self.zo_perturb_parameters(scaling_factor=1)     # θ += ε·z
loss1 = self.zo_forward(model, inputs)            # L(θ + ε·z)

# -ε perturbation
self.zo_perturb_parameters(scaling_factor=-2)    # θ -= 2ε·z (reverses first, goes opposite)
loss2 = self.zo_forward(model, inputs)            # L(θ - ε·z)

# Projected gradient (scalar)
projected_grad = ((loss1 - loss2) / (2 * self.args.mezo_eps)).item()

# Reset
self.zo_perturb_parameters(scaling_factor=1)     # θ += ε·z (back to original)

# Reconstruct gradient vector (deterministic from seed)
torch.manual_seed(self.zo_random_seed)
for name, param in self.named_parameters_to_optim:
    z = torch.normal(mean=0, std=1, size=param.data.size(), ...)
    grad_update = projected_grad * z              # scalar × random vector
    res_list.append(grad_update.flatten())

res = torch.cat(res_list)   # shape [num_params_in_last_layer]
```

**`zo_perturb_parameters`:**
```python
def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
    torch.manual_seed(random_seed or self.zo_random_seed)
    for name, param in self.named_parameters_to_optim:
        z = torch.normal(mean=0, std=1, size=param.data.size(), ...)
        param.data = param.data + scaling_factor * z * self.args.mezo_eps
```

The **same seed** is used for perturbation and gradient reconstruction, so z is identical in both cases.

---

### Sparsification (`select_masking`)

After getting raw gradient vectors `all_reps [N, full_dim]`, reduce to `[N, zo_dim]`:

```python
def select_masking(self, all_reps, source_list, per_source=True):
    masked_reps = torch.zeros((N, self.args.zo_dim), ...)

    for source in unique_sources:
        source_indices = where(source_list == source)
        source_reps = all_reps[source_indices]          # [n_source, full_dim]

        # Compute mean absolute gradient across source examples
        mean_reps = torch.abs(torch.mean(source_reps, dim=0))  # [full_dim]

        # Keep top zo_dim by magnitude (CoLM's mask M_q)
        if mezo_topk == "largest":
            ranked_indices = torch.argsort(mean_reps, descending=True)[:zo_dim]
        elif mezo_topk == "smallest":
            ranked_indices = torch.argsort(mean_reps)[:zo_dim]
        elif mezo_topk == "random":
            ranked_indices = torch.randperm(len(mean_reps))[:zo_dim]

        masked_reps[source_indices] = source_reps[:, ranked_indices]

    return masked_reps   # [N, zo_dim]
```

This implements the source-wise mask `M_q` from the paper — dimensions with largest mean gradient magnitude are most informative for distinguishing examples within that source.

---

## Facility Location Subset Selection (`facility_location.py`)

After sparsification, CoLM finds medoids via Facility Location:

**Objective** (submodular):
```
S* = argmax_{|S|≤b} Σ_{i∈V} max_{s∈S} [C - ||g_i - g_s||]
```

### `get_orders_and_weights`

```python
def get_orders_and_weights(B, X, metric, y=None, strategy="proportional", optim=None):
    # y: source labels per example
    # strategy: how to allocate budget B across sources

    # 1. Compute per-source budget
    if strategy == 'proportional':
        num_per_class = floor(count_per_class / N * B)  # proportional to source size

    # 2. Per source: find medoids
    for c in classes:
        class_X = X[class_indices]
        S, D = compute_cost_matrix(class_X, class_X, metric, return_sims=True)

        if optim is None:
            # Standard submodlib facility location
            flf = FacilityLocationFunction(n=len(class_indices), sijs=S, mode="dense")
            greedy_indices = flf.maximize(budget=num_per_class[c], optimizer="LazyGreedy")
            orders = [x[0] for x in greedy_indices]
        else:
            # Custom optimizer (e.g., FairOT)
            orders = optim(S, num_per_class[c], dist=D)

        # 3. Assign weights = cluster sizes
        weights = zeros(num_per_class[c])
        for i in range(len(class_indices)):
            if i in orders:
                weights[where(orders == i)] += 1
            else:
                weights[argmax(S[i, orders])] += 1  # assign to nearest medoid

        orders_all.append(class_indices[orders])
        weights_all.append(weights)

    return orders_all, weights_all
```

**Weight assignment**: Each non-selected example adds 1 to the weight of its nearest medoid. So medoids representing large clusters get high weights.

### `similarity` function
```python
def similarity(X, metric):
    if metric == 'cosine':
        S = pairwise_cosine_similarity(X, X)
    elif metric == 'l1':
        dists = torch.cdist(X, X, p=1)
        S = max(dists) - dists    # convert distance to similarity
    return S.numpy()
```

---

## Efficient MeZO: `SubsetTrainerEfficient`

For large batches, computing MeZO one sample at a time (2 forward passes × N samples) is slow. `SubsetTrainerEfficient` batches this using `DecomposedPhiCausalLM`.

### `custom_phi.py` — `DecomposedPhiCausalLM`

Splits Phi-2 into two parts:

```python
class DecomposedPhiCausalLM:
    def forward_till_penultimate(self, input_ids, attention_mask, ...):
        # Runs layers 0 to 30 (all except last)
        hidden_states = embed(input_ids)
        for layer in self.layers[:31]:   # first 31 layers
            hidden_states = layer(hidden_states, ...)
        return {'hidden_states': hidden_states, 'past_key_values': ..., ...}

    def forward_final_layer(self, intermediate_outputs, labels, ...):
        # Runs only layer 31 + layernorm + lm_head + loss
        hidden_states = intermediate_outputs['hidden_states']
        hidden_states = self.layers[31](hidden_states, ...)
        hidden_states = self.final_layernorm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = cross_entropy(logits, labels, reduction='none')  # per-sample
        return loss
```

### `SubsetTrainerEfficient.save_select`

```python
def save_select(self, model, inputs):
    # 1. Single shared forward pass through layers 0-30
    zo_intermediate = self.zo_forward_till_penultimate(model, inputs)
    zo_past_key_values = deepcopy(zo_intermediate["past_key_values"])

    # 2. +ε perturbation on last layer only — run layer 31 for entire batch
    self.zo_perturb_parameters(scaling_factor=1)
    loss1_batch = self.zo_forward_final_layer(model, inputs["labels"], zo_intermediate)
    # loss1_batch shape: [batch_size]

    # 3. -ε perturbation — restore past_key_values, run layer 31 again
    zo_intermediate["past_key_values"] = zo_past_key_values
    self.zo_perturb_parameters(scaling_factor=-2)
    loss2_batch = self.zo_forward_final_layer(model, inputs["labels"], zo_intermediate)

    # 4. Per-sample projected gradients (vectorized)
    projected_grads = (loss1_batch - loss2_batch) / (2 * self.args.mezo_eps)
    # projected_grads shape: [batch_size]

    # 5. Reconstruct gradient vectors for all samples at once
    torch.manual_seed(self.zo_random_seed)
    param = self.named_parameters_to_optim[0][1]
    z = torch.normal(mean=0, std=1, size=param.data.size(), ...)

    # Outer product: [batch_size, 1, 1, ...] * [param_shape] = [batch_size, *param_shape]
    grad_updates = projected_grads.view(-1, *([1]*len(param.shape))) * z.unsqueeze(0)

    res = grad_updates.view(batch_size, -1)  # [batch_size, num_params_in_layer]
    return res
```

**Cost comparison**:
- Sequential MeZO: `2 * N` forward passes (N = batch size)
- Efficient MeZO: `1 shared forward pass + 2 last-layer passes` ≈ `1 + 2/32` effective forward passes

---

## Full CoLM Pipeline (End-to-End)

```
Large random batch (e.g., B=128 examples from MathInstruct)
    │
    ├─ Separate small sources (sources 0,1,3,5,7,8,9,10,11,13 in MathInstruct)
    │   └─ list_idx_keep: always included, not processed further
    │
    ├─ For big sources (sources 2,4,6,12 in MathInstruct):
    │   │
    │   ├─ save_select() → per-example gradient [n_big, full_dim]
    │   │   ├─ MeZO: 2 forward passes, projected grad * z
    │   │   └─ Efficient MeZO: 1 shared pass + 2 last-layer passes (vectorized)
    │   │
    │   ├─ Adam normalization (if mezo_optim="adam"):
    │   │   └─ normalize by m_t / (sqrt(v_t) + ε)
    │   │
    │   ├─ select_masking():
    │   │   └─ per source: top zo_dim=2560 dims by mean |grad| → [n_big, 2560]
    │   │
    │   └─ select_data() via get_orders_and_weights():
    │       ├─ compute pairwise L1 distances between sparsified grads
    │       ├─ per big-source: run FacilityLocationFunction.maximize(LazyGreedy)
    │       └─ assign weights = cluster sizes
    │
    ├─ Combine: list_idx_keep + selected_big_source_idx
    │
    └─ Train on combined selected set (K examples total)
        └─ uniform weights for all (small + big source medoids)
```

---

## Variance Reduction Guarantee (Theorem 4.3)

The paper proves that mini-batch coresets have smaller variance than random mini-batches by up to:

```
(κ/m) * (α_u - α*) * (2α* + (κ/m)(α_u - α*))
```

Where:
- `κ` = number of outliers not in any dense cluster
- `m` = number of mini-batches
- `α_u` = max distance from outlier to centroid
- `α*` = neighborhood radius of medoids

In practice: Figure 3(b) in paper shows CoLM (bs=64) has lower gradient variance than random (bs=64), though higher than random (bs=128) — but the more uniform learning across sources makes up for this.

---

## Key Hyperparameters and Their Effect

| Parameter | Default | Effect |
|-----------|---------|--------|
| `zo_dim` | 2560 | Sparsified gradient dim; too small = poor distance metric; too large = curse of dimensionality |
| `small_batch_ratio` | 0.5 | Select 50% of large batch; 0.5 means train on half the examples |
| `mezo_eps` | 1e-3 | MeZO perturbation scale; too large = inaccurate gradient estimate |
| `mezo_optim` | "adam" | "adam" normalizes by historical avg; "sgd" uses raw gradients |
| `facility_similarity` | "cosine" | Distance metric for facility location; "l1" used for high-dim |
| `keep_sources` | source indices | Which sources are "small" and always included |
| `source_wise_selection` | "proportional" | How to split selection budget across big sources |
| `mezo_topk` | "largest" | "largest" keeps dims with largest mean gradient = most discriminative |

---

## MathInstruct Source Distribution

```
Sources 10-13 (large): ~100K+ examples each  → go through facility location selection
Sources 0-9  (small):  < average count        → always included via keep_sources

Average count: 18,717
Ratio largest/smallest: ~300x
```

This extreme imbalance is exactly why the `list_idx_keep` mechanism is critical — without it, small sources would never appear in the selected mini-batch.

---

## SuperGLUE Without Source Labels

When source information is not available (e.g., SST-2, CB, MultiRC):

```python
# From subset_trainer_distributed.py and train.py:
# 1. Warm up for 20 iterations with random mini-batches
# 2. Cluster model's hidden states to find sources:

from sklearn.cluster import KMeans
hidden_states = model.forward(inputs, output_hidden_states=True).hidden_states[-1]
kmeans = KMeans(n_clusters=c).fit(hidden_states.cpu().numpy())
source_labels = kmeans.labels_

# 3. Use cluster assignments as source labels going forward
# 4. Update clusters periodically during training (every ~25% of steps)
```

Table 7 in paper shows this improves over random fine-tuning by 4.5% average on SuperGLUE.
