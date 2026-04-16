# Muon Optimizer: Visual Code Comparison

## Side-by-Side Code Implementation Details

---

## 1. Orthogonalization Method Comparison

### NemoMuon: Newton-Schulz Orthogonalization
```python
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Computes matrix square root inverse via Newton-Schulz iterations.
    Used for orthogonalization: X → X / ||X|| → converges to U*V^T via iteration
    """
    assert G.ndim == 2
    
    # Newton-Schulz coefficients (tuned for stability)
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Ensure tall matrix (more numerically stable)
    if G.size(0) > G.size(1):
        G = G.t()
        transposed = True
    else:
        transposed = False
    
    # Normalize
    norm = G.norm() + eps
    X = G / norm
    X = X.bfloat16()  # Cast to lower precision for speed
    
    # Newton-Schulz Iteration: X_{k+1} = X_k @ (a*I + B_k)
    # where B_k = b*X_k^T*X_k + c*X_k^T*X_k^T*X_k
    for _ in range(steps):
        A = X.t() @ X              # A = X^T @ X (2x2 matrix)
        B = b * A + c * A @ A      # B = b*A + c*A^2 (polynomial update)
        X = X @ (a * torch.eye(X.size(1), device=X.device, dtype=X.dtype) + B)
    
    if transposed:
        X = X.t()
    
    return X.float()  # Cast back to float32

# Complexity: O(k * m * n^2) where k=5 iterations, m × n = matrix size
# For large matrices: O(n^2) iterations but each iteration is matrix-multiply heavy
```

### GREATS_COLM: SVD-Based Orthogonalization
```python
def _apply_muon_step(self, p, grad, group, state):
    """Apply Muon with SVD-based orthogonalization"""
    
    # ... (momentum setup) ...
    
    # SVD-based orthogonalization for 2D weight matrices
    if p.dim() == 2:
        # Compute SVD: A = U @ Σ @ V^T
        u, s, v = torch.linalg.svd(
            update / (update.norm() + group['eps']),  # Normalize first
            full_matrices=False  # Only compute thin SVD
        )
        # u: m × r, s: r, v: r × n (where r = rank)
        
        # Reconstruct orthogonal matrix: U @ V^T
        # This is the closest orthogonal matrix to the scaled gradient
        update_ortho = u @ v  # Result: m × n
    else:
        update_ortho = update
    
    # Apply update (no scaling)
    p.add_(update_ortho, alpha=-group['lr'])

# Complexity: O(m * n^2) for thin SVD via LAPACK
# For large matrices: Much slower than Newton-Schulz for m, n > 1000
```

### Complexity Comparison
```
Matrix Size: 1000 x 1000

Newton-Schulz (5 iterations):
  Each iteration: 1000 * 1000^2 = 1 billion ops
  5 iterations: 5B × 5 = ~25B operations
  Time: ~50ms on GPU

SVD:
  Thin SVD: 1000 * 1000^2 = ~1 billion ops
  Time: ~200-300ms on GPU (much slower!)

Winner for large matrices: Newton-Schulz by 4-6x
```

---

## 2. Parameter Classification: Line-by-Line

### NemoMuon: Smart Classification
```python
def _classify_param(self, p):
    """
    Classify parameter as Muon-eligible (2D weight) or AdamW-eligible (bias/embedding).
    
    Muon works best on 2D matrices but can diverge on huge embeddings.
    Strategy: Use Muon on medium-sized 2D matrices only.
    """
    is_embedding = (p.ndim == 2 and p.size(0) > 10000)      # EMBEDDING TEST
    # If 2D matrix has >10,000 rows: likely embedding table
    # Example: Embedding(vocab=50000, dim=512) → 50000×512 matrix
    # Such large matrices can cause numerical instability with Muon
    
    is_norm_or_bias = (p.ndim < 2)                          # BIAS/NORM TEST
    # 1D parameters: Always biases, layer norms, etc.
    # Example: Linear(1000, 512) has bias of shape (512,)
    
    is_linear_weight = (p.ndim == 2 and not is_embedding)   # WEIGHT TEST
    # 2D matrices that aren't embeddings
    # Example: Linear(1000, 512) has weight shape (512, 1000)
    
    return is_linear_weight                                  # ← Only these get Muon!

# Result: Muon applied to query/key/value projections, MLPs only
# AdamW applied to embeddings, layer norms, biases
```

### GREATS_COLM: Simple Classification
```python
def _is_2d_parameter(param: torch.nn.Parameter) -> bool:
    """Classify: 2D → Muon, else → AdamW"""
    return param.dim() == 2  # ← That's it!

# In create_muon_optimizer:
for p in params_list:
    if _is_2d_parameter(p):      # ALL 2D matrices
        params_2d.append(p)      # Even large embeddings!
    else:
        params_non_2d.append(p)

# Result: Muon applied to ALL 2D matrices including embeddings
# AdamW applied to all non-2D parameters only
```

### Parameter Classification in Action
```
Model: GPT-2 (1.5B params)

Parameter Distribution:
─────────────────────────────────────────

Embeddings:
  token_embeddings: (50257, 768)      2D → GREATS: Muon! NemoMuon: AdamW
  position_embeddings: (1024, 768)    2D → GREATS: Muon! NemoMuon: Muon

Transformer Blocks (12 layers × 768 hidden):
  attention.q_proj: (768, 768)        2D → Both: Muon ✓
  attention.k_proj: (768, 768)        2D → Both: Muon ✓
  attention.v_proj: (768, 768)        2D → Both: Muon ✓
  attention.out_proj: (768, 768)      2D → Both: Muon ✓
  
  mlp.dense_h_to_4h: (768, 3072)      2D → Both: Muon ✓
  mlp.dense_4h_to_h: (3072, 768)      2D → Both: Muon ✓

Biases:
  attention.q_proj.bias: (768,)       1D → Both: AdamW ✓
  mlp.dense_h_to_4h.bias: (3072,)     1D → Both: AdamW ✓

Layer Norms:
  ln_1.weight: (768,)                 1D → Both: AdamW ✓
  ln_1.bias: (768,)                   1D → Both: AdamW ✓

Output Head:
  lm_head: (50257, 768)               2D → GREATS: Muon! NemoMuon: AdamW

─────────────────────────────────────────

DIFFERENCE SUMMARY:
  NemoMuon approach:
    - Muon: medium 2D matrices (projections, MLPs) ✓ ✓ ✓
    - AdamW: embeddings, biases, norms ✓ ✓ ✓
    → More conservative, tuned for stability
  
  GREATS_COLM approach:
    - Muon: ALL 2D (including embeddings) ✗ ✗ ✗ (risky!)
    - AdamW: biases, norms only ✓ ✓ ✓
    → More aggressive, might diverge on embeddings
```

---

## 3. Main Optimization Step Loop

### NemoMuon: Full Implementation
```python
@torch.no_grad()
def step(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        lr = group['lr']
        momentum = group['momentum']
        nesterov = group['nesterov']
        ns_steps = group['ns_steps']
        adam_w_lr = group['adam_w_lr']
        adam_beta1, adam_beta2 = group['adam_w_betas']
        weight_decay = group['weight_decay']
        eps = group['eps']

        for p in group['params']:
            # ============================================================
            # GRADIENT RETRIEVAL (Megatron Support!)
            # ============================================================
            grad = p.grad
            if grad is None and hasattr(p, 'main_grad'):  # ← Megatron check!
                grad = p.main_grad  # Use Megatron's gradient buffer
            if grad is None:
                continue
            
            state = self.state[p]
            
            # Initialize state
            if len(state) == 0:
                state['step'] = 0
                state['use_muon'] = self._classify_param(p)  # Smart classification
                state['momentum_buffer'] = torch.zeros_like(p)
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            
            state['step'] += 1
            use_muon = state['use_muon']
            
            if use_muon:
                # ====================================================
                # MUON UPDATE FOR 2D WEIGHT MATRICES
                # ====================================================
                
                # 1. Weight Decay (optional, applied to parameters)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                
                # 2. Update Momentum Buffer
                # M_t = β * M_{t-1} + (1-β) * ∇L
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                
                # 3. Nesterov Acceleration (optional)
                # If Nesterov: use "lookahead" gradient
                if nesterov:
                    g = buf.clone().add_(grad, alpha=1 - momentum)
                else:
                    g = buf
                
                # 4. Orthogonalize via Newton-Schulz
                # Compute: g_ortho = U @ V^T (orthogonal matrix)
                g_ortho = zeropower_via_newtonschulz5(g, steps=ns_steps)
                
                # 5. ASPECT RATIO SCALING (CRITICAL!)
                # Scale by √(rows/cols) to handle rectangular matrices
                rows, cols = g.size()
                scale = max(1, rows / cols) ** 0.5  # ← E.g., 10×5 → scale by √2
                g_ortho *= scale
                
                # 6. Apply Parameter Update
                # W_{t+1} = W_t - lr * g_ortho
                p.add_(g_ortho, alpha=-lr)
            
            else:
                # ====================================================
                # ADAMW UPDATE FOR BIASES/EMBEDDINGS
                # ====================================================
                
                # Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1 - adam_w_lr * weight_decay)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                # Update moments
                exp_avg.mul_(adam_beta1).add_(grad, alpha=1 - adam_beta1)
                exp_avg_sq.mul_(adam_beta2).addcmul_(grad, grad, value=1 - adam_beta2)
                
                # Bias correction
                bias_correction1 = 1 - adam_beta1 ** state['step']
                bias_correction2 = 1 - adam_beta2 ** state['step']
                
                step_size = adam_w_lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                
                # Adaptive update
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)
    
    return loss
```

### GREATS_COLM: Cleaner Implementation
```python
def step(self, closure=None):
    """Simpler version without Megatron support"""
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()
    
    for group in self.param_groups:
        for p in group['params']:
            # ============================================================
            # GRADIENT RETRIEVAL (PyTorch only)
            # ============================================================
            if p.grad is None:
                continue  # Skip if no gradient
            
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError('Muon does not support sparse gradients')
            
            state = self.state[p]
            
            # Initialize state
            if len(state) == 0:
                state['step'] = 0
                state['momentum_buffer'] = None
            
            state['step'] += 1
            
            # Weight decay (incorporated into gradient)
            if group['weight_decay'] != 0:
                grad = grad.add(p, alpha=group['weight_decay'])
            
            # Dispatch to appropriate handler
            if p.dim() == 2:
                self._apply_muon_step(p, grad, group, state)
            else:
                self._apply_adamw_step(p, grad, group, state)
    
    return loss

def _apply_muon_step(self, p, grad, group, state):
    """MUON UPDATE"""
    
    # Momentum update
    buf = state['momentum_buffer']
    if buf is None:
        buf = state['momentum_buffer'] = torch.clone(grad).detach()
    else:
        buf.mul_(group['momentum']).add_(grad)
    
    # Nesterov
    if group['nesterov']:
        update = buf.add(grad, alpha=group['momentum'])
    else:
        update = buf
    
    # SVD-based orthogonalization
    u, s, v = torch.linalg.svd(
        update / (update.norm() + group['eps']),
        full_matrices=False
    )
    update_ortho = u @ v
    
    # Apply update (NO SCALING!)
    p.add_(update_ortho, alpha=-group['lr'])

def _apply_adamw_step(self, p, grad, group, state):
    """ADAMW UPDATE (simplified)"""
    
    # Simple momentum-based approach
    momentum = group['momentum']
    
    buf = state['momentum_buffer']
    if buf is None:
        buf = state['momentum_buffer'] = torch.clone(grad).detach()
    else:
        buf.mul_(momentum).add_(grad, alpha=1)
    
    if group['nesterov']:
        update = buf.add(grad, alpha=momentum)
    else:
        update = buf
    
    p.add_(update, alpha=-group['lr'])
```

### Code Comparison Summary
```
NemoMuon Characteristics:
✓ 544 lines - comprehensive
✓ Megatron support (main_grad fallback)
✓ Smart parameter classification
✓ Aspect ratio scaling
✓ Newton-Schulz orthogonalization
✓ Separate AdamW implementation
✓ Extensive diagnostics/logging
✗ Higher complexity
✗ NeMo/Megatron only

GREATS_COLM Characteristics:
✓ 372 lines - concise
✗ PyTorch only (no Megatron)
✓ Simple parameter classification
✗ No aspect ratio scaling
✓ SVD orthogonalization
✓ Simplified AdamW
✗ Minimal diagnostics
✓ Lower complexity
✓ Universal compatibility
```

---

## 4. Learning Rate Tuning: Default vs Recommended

### NemoMuon Defaults
```python
lr = 0.02                        # Primary LR for Muon (weight matrices)
adam_w_lr = 0.003               # Secondary LR for AdamW (biases/embeddings)
momentum = 0.95                 # Very high momentum (nearly integrator)
weight_decay = 0.1              # Strength weight decay
eps = 1e-8                       # Numerical stability
ns_steps = 5                     # Newton-Schulz iterations

Typical Usage:
  Weight matrices: 0.02 (Muon)
  Embeddings: 0.003 (AdamW) → ~6.7x lower!
  Biases: 0.003 (AdamW)
  Layer norms: 0.003 (AdamW)
```

### GREATS_COLM Defaults
```python
lr = 0.001                       # Single LR for all (must balance)
momentum = 0.95                  # Same as NemoMuon
weight_decay = 0.0               # Lower default
eps = 1e-7                       # Same magnitude
ns_steps = 5                     # (ignored, uses SVD)

Typical Usage:
  ALL 2D (weight matrices): 0.001 (Muon)
  ALL 2D (embeddings): 0.001 (Muon) ← Same LR!
  Non-2D: Falls back to simplified update

Issue: Embeddings get same LR as projections
→ Might need manual tuning: lower LR for embeddings
```

### Recommended Tuning
```
For models with embeddings (GPT, T5, etc.):

NemoMuon (built-in handling):
  Just use defaults:
    lr=0.02, adam_w_lr=0.003, momentum=0.95
  ✓ Automatically handles embedding scaling

GREATS_COLM (needs manual adjustment):
  Option 1: Create parameter groups
    param_groups = [
        {'params': weights_2d, 'lr': 0.02},
        {'params': embeddings, 'lr': 0.003},
    ]
  
  Option 2: Use lower global LR
    lr=0.01  # Compromise between embedding and weight needs
```

---

## 5. GPU Memory and Speed Trade-off

### Newton-Schulz (NemoMuon)
```
Memory: Minimal extra (only stores normalized matrix during iteration)
Speed:  Fast (matrix operations, GPU-friendly)
Stability: Good (coefficients tuned for numerical stability)

GPU Operations (inner loop, 1000×1000 matrix):
  1. Matrix multiply: X.t() @ X = 1B FLOPs
  2. Matrix multiply: A @ A = 1B FLOPs (optional, for B term)
  3. Matrix multiply: X @ (a*I + B) = 1B FLOPs
→ ~3B FLOPs per iteration × 5 iterations = 15B FLOPs
→ On A100: ~15ms per step

GPU Memory: ~5MB for intermediate matrices
```

### SVD (GREATS_COLM)
```
Memory: Higher (SVD stores U, Σ, V^T matrices)
Speed:  Slower (LAPACK-based, CPU-calling overhead)
Stability: Excellent (inherent numerical stability)

GPU Operations (1000×1000 matrix):
  1. SVD decomposition: ~6B FLOPs (LAPACK optimized)
  2. Reconstruct U @ V^T = 1B FLOPs
→ ~7B FLOPs total
→ On A100: ~100-200ms per step (overhead from LAPACK)

GPU Memory: ~20MB for U, Σ, V matrices
```

---

## 6. Debugging & Diagnostics

### NemoMuon: Comprehensive Logging
```python
class OptimizerDiagnosticCallback(Callback):
    """Logs detailed optimizer statistics every N steps"""
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 10 == 0:
            muon_updates = count_muon_updates()
            adam_updates = count_adam_updates()
            print(f"[Step {batch_idx}]")
            print(f"  Muon updates (2D): {muon_updates}")
            print(f"  Adam updates (1D): {adam_updates}")
            print(f"  Grad norms by layer...")
            # Full diagnostics available
```

### GREATS_COLM: Minimal Logging
```python
# No built-in diagnostics
# Users must manually add debugging:
if step % 100 == 0:
    print(f"Loss: {loss.item()}")
    # That's typically it
```

---

## Conclusion: Which One to Use?

```
CHOOSE NemoMuon IF:
├─ Using Megatron/NeMo for large-scale training
├─ Training on distributed clusters (GPU farms)
├─ Need comprehensive monitoring
├─ Dealing with very large matrices (10K×10K+)
├─ Want optimized performance for MoE models
└─ Have Newton-Schulz iteration tuning expertise

CHOOSE GREATS_COLM IF:
├─ Standard PyTorch project
├─ Single GPU or small multi-GPU setup
├─ Value simplicity and maintainability
├─ Want maximum numerical stability (SVD)
├─ Integrating with existing PyTorch infrastructure
└─ Prefer less specialized dependencies
```

