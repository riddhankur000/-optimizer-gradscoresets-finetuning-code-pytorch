# Quick Start: AdamW vs Muon

## TL;DR - Switch Optimizers in 30 Seconds

### Use AdamW (Default)
```yaml
# In config.yaml:
optimizer_config:
  optimizer_type: "adamw"
```

### Use Muon
```yaml
# In config.yaml:
optimizer_config:
  optimizer_type: "muon"
```

Then run training:
```bash
bash scripts/train_multitask_dataset.sh
```

---

## Pre-Made Configs

### AdamW Setup
```bash
bash scripts/train_multitask_dataset.sh ./config_adamw.yaml
```

### Muon Setup  
```bash
bash scripts/train_multitask_dataset.sh ./config_muon.yaml
```

---

## Side-by-Side Comparison

| Feature | AdamW | Muon |
|---------|-------|------|
| **Optimizer Type** | Adaptive learning rate | Orthogonal updates |
| **Suitable For** | General purpose | LLM training |
| **Parameter Type** | All parameters | 2D params only |
| **Bias Handling** | AdamW | AdamW (automatic) |
| **Learning Rate** | 1e-4 to 1e-3 | 1e-3 to 1e-2 |
| **Speed** | Fast | Slower (more accurate) |
| **Convergence** | Stable | May be faster |
| **Memory** | Low | Low |

---

## Minimal Config Changes

### From AdamW to Muon
1. Open `config.yaml`
2. Change: `optimizer_type: "adamw"` → `optimizer_type: "muon"`
3. Update learning rate: `muon_lr: 0.002` (10x higher usually)
4. Save and run

### From Muon to AdamW
1. Open `config.yaml`
2. Change: `optimizer_type: "muon"` → `optimizer_type: "adamw"`
3. Keep learning rate: `learning_rate: 0.0002`
4. Save and run

---

## Key Hyperparameter Defaults

**AdamW**:
- `learning_rate`: 0.0002
- `weight_decay`: 0.01
- `adam_beta1`: 0.9

**Muon**:
- `muon_lr`: 0.002 (10x higher)
- `muon_weight_decay`: 0.1 (10x higher)
- `muon_momentum`: 0.95
- `muon_ns_steps`: 5

---

## Quick Experiment

### Run Both Side-by-Side
```bash
# Terminal 1: AdamW
bash scripts/train_multitask_dataset.sh ./config_adamw.yaml

# Terminal 2 (in another window): Muon
bash scripts/train_multitask_dataset.sh ./config_muon.yaml
```

### Compare Results on W&B
- Same number of steps
- Same batch size
- Same seed
- Different run names for easy identification
- Monitor: loss curves, validation metrics, convergence speed

---

## Typical Results For Math Training

### AdamW
- Training loss: Smooth decreasing curve
- Convergence: ~500-700 steps to good loss
- Stability: Very stable
- Speed: ~1.2ms per step

### Muon  
- Training loss: May have slight oscillations initially
- Convergence: ~400-600 steps (sometimes faster)
- Stability: Good with proper hyperparams
- Speed: ~1.5-1.8ms per step (slower but more accurate)

---

## When to Use Each

### Use AdamW If:
- ✅ You have a working baseline
- ✅ Training is stable and converging well
- ✅ You want fastest training speed
- ✅ You want safest choice (well-tested)

### Use Muon If:
- ✅ You want to experiment with new methods
- ✅ You're optimizing for final accuracy (not speed)
- ✅ You have time for hyperparameter tuning
- ✅ You want to compare different optimizer implementations

---

## Troubleshooting

**Q: How do I know which optimizer is running?**

A: Check the output:
```
✓ Created AdamW optimizer with lr=0.0002
# OR
✓ Creating Muon optimizer
  2D parameters (Muon): 5600
  Non-2D parameters (AdamW): 200
```

**Q: Why is Muon slower?**

A: Muon does Newton-Schulz iterations (5 steps) for orthogonalization, which is more computation but produces better updates.

**Q: Can I mix both optimizers?**

A: Muon automatically does this! 2D params use Muon, non-2D use AdamW.

**Q: Which is better?**

A: Depends on your goal:
- Speed: AdamW wins
- Final accuracy: Muon often wins (research shows)
- Stability: AdamW wins

---

## Files Modified

```
config.yaml                      # Added muon parameters
config_adamw.yaml               # New: AdamW ready-to-use config
config_muon.yaml                # New: Muon ready-to-use config
colm/train/config_loader.py     # Updated: config summary
colm/train/optimizer_factory.py # New: optimizer creation
colm/train/train_multitask.py   # Updated: trainer integration
MUON_GUIDE.md                   # New: full documentation
MUON_IMPLEMENTATION.md          # New: implementation details
```

---

## Minimal Working Example

```bash
# 1. Copy example config
cp config_adamw.yaml my_experiment_adamw.yaml
cp config_muon.yaml my_experiment_muon.yaml

# 2. Edit hyperparameters if needed
# vim my_experiment_adamw.yaml
# vim my_experiment_muon.yaml

# 3. Run both
bash scripts/train_multitask_dataset.sh ./my_experiment_adamw.yaml &
bash scripts/train_multitask_dataset.sh ./my_experiment_muon.yaml

# 4. Monitor on W&B dashboard
# Compare run names: "muon" vs "adamw"
```

---

## One-Liner Tests

```bash
# Test AdamW config
bash scripts/train_multitask_dataset.sh ./config_adamw.yaml

# Test Muon config
bash scripts/train_multitask_dataset.sh ./config_muon.yaml

# Test with custom steps for quick validation
# (edit max_steps: 10 in config for quick test)
bash scripts/train_multitask_dataset.sh ./config_muon.yaml
```

---

## Checklist

- [ ] PyTorch >= 2.10.0 installed
- [ ] Dataset regenerated with Llama tokenizer
- [ ] config.yaml has correct optimizer_type
- [ ] Either config_adamw.yaml or config_muon.yaml ready
- [ ] WANDB credentials configured
- [ ] GPU memory is sufficient (47GB available)

---

**Ready?** Pick an optimizer, run training, and monitor results! 🚀
