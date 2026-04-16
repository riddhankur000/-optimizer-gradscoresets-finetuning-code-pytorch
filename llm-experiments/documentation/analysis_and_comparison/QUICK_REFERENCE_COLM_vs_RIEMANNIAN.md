# Quick Reference: CoLM vs Riemannian Fine-Tuning

## At a Glance

| Feature | CoLM | Riemannian |
|---------|------|-----------|
| **Focus** | Data coreset selection | Manifold-aware parameter optimization |
| **Key Innovation** | Train only on informative samples | Riemannian geometry for LoRA updates |
| **Data Selection** | 5 algorithms (GREATS, FairOT, Facility Location) | None (static) |
| **LoRA Approach** | Standard PEFT LoRA | Fixed-Rank Riemannian LoRA (doubled rank) |
| **Optimizer** | AdamW (standard) | Custom (RiemannianLora, RiemannianSGD, etc.) |
| **Update Rule** | Euclidean: θ ← θ - lr∇L | Riemannian: θ ← Retract(θ - lr∇L) |
| **Training Loop** | Collect reps → Select → Train selected | Standard training loop |
| **Memory** | Lower (selective training) | Higher (2× LoRA) |
| **Complexity** | Medium (selection logic) | High (manifold operations) |
| **Inference** | vLLM (optimized, 17 benchmarks) | HF Pipeline (simpler, custom) |

## Training Time Overhead

| Operation | CoLM | Riemannian |
|-----------|------|-----------|
| Representation extraction | Per-batch (~0.1s disclosed) | N/A |
| Selection algorithm | 500 iters for FairOT, greedy for others | N/A |
| Manifold operations | N/A | ~1-3% per step (QR+SVD) |
| **Total** | ~0.15-0.2s per batch | ~1-3% overhead |

## Memory Footprint (Phi-2 2.7B example)

| Item | CoLM | Riemannian |
|------|------|-----------|
| Base model | ~10GB | ~10GB |
| LoRA weights (r=32) | ~0.4GB | ~0.8GB (doubled r) |
| Activation cache | Higher (full batch for selection) | Standard |
| Gradient buffer | Lower (select only) | Standard |
| **Total** | ~24GB | ~32GB+ |

## Mathematical Core

### CoLM: Coreset Selection Problem
```
minimize Σᵢ wᵢ · Lᵢ(θ)
subject to: Σᵢ wᵢ = 1, wᵢ ≥ 0
            select top-K indices greedily
            fairness constraints (optional)
```

### Riemannian: Manifold Optimization Problem
```
minimize L(θ)
subject to: rank(θ_LoRA) = r  [hard constraint]
            θ ∈ St(m,r) × ℝ(n,r)  [on fixed-rank manifold]
            
Update rule: θₜ₊₁ = Retract_θₜ(-αₜ∇L(θₜ))
```

## When Each Excels

### CoLM Advantages
✅ Reduces samples to train on (effective batch < actual batch)
✅ Handles heterogeneous multi-source data
✅ Empirically validated (ICLR 2025 paper)
✅ Integrates with standard infrastructure
✅ Better for redundant/noisy datasets

### Riemannian Advantages
✅ Theoretically principled updates
✅ No rank drift (geometry enforced)
✅ Cleaner convergence properties
✅ Structured, interpretable updates
✅ Fewer hyperparameters to tune

## Architecture Differences

### CoLM Data Flow
```
Batch → Extract Reps → Select K/B samples → Train on K → Optimizer.step()
```
- **Key**: Selection BEFORE training (saves compute)
- **Cost**: Representation extraction + selection algorithm
- **Benefit**: Only train on important samples

### Riemannian Data Flow
```
Batch → Forward → Backward → [Manifold ops] → Optimizer.step()
```
- **Key**: Manifold operations DURING optimization
- **Cost**: Extra QR/SVD per layer per step
- **Benefit**: Guaranteed manifold constraint satisfaction

## Selection Algorithms (CoLM Only)

1. **Facility Location** (~O(n²))
   - Selects K representative centers
   - Assigns remaining to nearest center
   - Smooth diversity-importance tradeoff

2. **GREATS** (Fast greedy)
   - Selects high-gradient samples
   - Penalizes duplicate selections
   - Best for simple cases

3. **FairOT** (Slow but principled)
   - Balances similarity with fairness
   - Handles multi-source data
   - Regularized Wasserstein distance

4. **SPOT** (Streaming)
   - For online/streaming scenarios
   - Less used in current implementation

5. **FairOT Multi-source**
   - Per-source variant of FairOT
   - Ensures diversity across sources

## Configuration Checklist

### For CoLM
- [ ] Dataset has redundant or low-quality samples
- [ ] Multi-source data with fairness needs
- [ ] Can afford ~0.1-0.2s selection overhead
- [ ] Have 24GB+ GPU memory available
- [ ] Want empirically proven method

### For Riemannian
- [ ] Want theoretical guarantees
- [ ] Have 32GB+ GPU memory (2× LoRA)
- [ ] Comfortable with custom optimizers
- [ ] Manifold optimization sounds appealing
- [ ] Small model high-quality dataset

## Integration Points

### CoLM with standard training:
```python
# Easy to modify:
from transformers import Trainer
trainer = SubsetTrainer(...)  # drop-in replacement
# Uses standard LoRA, standard optimizers
```

### Riemannian with standard training:
```python
# Requires custom setup:
from src.finetune import get_trainer  # custom trainer
# Must use Riemannian optimizers explicitly
# YAML configuration required
```

## Hyperparameter Tuning Priority

### CoLM (in importance order)
1. `data_selection_method` (algorithm choice)
2. `small_batch_ratio` (selection aggressiveness)
3. `gradient_accumulation_steps` (batch for selection)
4. `data_selection_unit` ("masked_grad" vs "mezo" vs "rep")

### Riemannian (in importance order)
1. `optimizer_config.optim` (algorithm choice)
2. LoRA `r` (rank)
3. `lr` (learning rate)
4. `betas` (momentum parameters)

## Evaluation Coverage

### CoLM
- 6 Math benchmarks (MATH, GSM8K, SVAMP, etc.)
- 11 SuperGLUE tasks
- **Total: 17 tasks**
- **Framework: vLLM** (optimized inference)

### Riemannian
- Custom dataset per config
- Task-based evaluation (configurable)
- **Framework: HF Pipeline** (simpler)

## Paper Citations & References

### CoLM
```bibtex
@article{nguyen2025mini,
  title={Mini-batch Coresets for Memory-efficient Language Model Training on Data Mixtures},
  author={Nguyen, Dang and Yang, Wenhan and Anand, Rathul and Yang, Yu and Mirzasoleiman, Baharan},
  journal={ICLR},
  year={2025}
}
```

### Riemannian (Inferred from code)
- Fixed-Rank manifold optimization for LoRA
- Combines Riemannian geometry with LoRA fine-tuning
- Custom optimizers: RiemannianLora, RiemannianSGD
- LoRA-QR factorization callback for stability

## Debugging Tips

### CoLM
- Check selection indices saved in `args.output_dir/indices/`
- Monitor `small_batch_ratio` impact on convergence
- Use `data_selection_unit="masked_grad"` for clearer signals
- Watch for synchronization delays in DDP

### Riemannian
- Verify orthogonality of A matrices (should be ~ identity)
- Check rank conservation in B matrices
- Monitor manifold distance violations (should be ~0)
- Enable `lora_qr` callback for numerical stability

## Quick Start Commands

### CoLM Training
```bash
cd /data/riddhankur/PROJECTS/GREATS_COLM_pytorch/local/llm-experiments
bash scripts/run_math_efficient.sh

# Or custom:
python colm/train/train.py \
  --model_name_or_path phi-2 \
  --lora \
  --data_selection_method greats \
  --output_dir ./output
```

### Riemannian Training
```bash
cd /data/riddhankur/PROJECTS/REIMANIAN_FINETUNE/RiemanianFinetune
python run.py config.yaml

# Modify config.yaml for different tasks/settings
```

---

**Full Document**: See `/data/riddhankur/LLM_FINETUNING_COMPARISON.md` for comprehensive analysis with code examples and diagrams.
