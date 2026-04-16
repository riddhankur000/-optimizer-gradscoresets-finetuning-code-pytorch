# Configuration Profile System - Quick Reference

## Single Config File Structure

```
config.yaml (UNIFIED)
├── active_profiles              ← SELECT HERE
│   ├── optimizer: "adamw"       (or "muon")
│   └── gpu: "gpu_0"             (or "gpu_1", "gpu_multi", "cpu")
│
├── optimizer_profiles           ← PROFILE DEFINITIONS
│   ├── adamw:
│   │   ├── optimizer_type: "adamw"
│   │   ├── adam_beta1: 0.9
│   │   └── ...
│   └── muon:
│       ├── optimizer_type: "muon"
│       ├── muon_lr: 0.002
│       └── ...
│
├── gpu_profiles                 ← PROFILE DEFINITIONS
│   ├── gpu_0:
│   │   ├── device_ids: "0"
│   │   └── ...
│   ├── gpu_1:
│   │   ├── device_ids: "1"
│   │   └── ...
│   ├── gpu_multi:
│   │   ├── device_ids: "0,1"
│   │   └── ...
│   └── cpu:
│       ├── device_ids: null
│       └── ...
│
├── model_config
├── tokenizer_config
├── lora_config
├── dataset_config
├── training_config
├── tokenizer_args
├── training_behavior
├── multitask_config
├── wandb_config
└── system_config
```

---

## Profile Selection Flow

```
User edits config.yaml:
┌─────────────────────────────┐
│ active_profiles:            │
│   optimizer: "muon"    ──┐  │
│   gpu: "gpu_1"         ┌┘  │
└────────────────────────┼───┘
                         │
                         ▼
              Load config.yaml
                         │
                         ▼
         apply_active_profiles()
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
Get "muon"         Get "gpu_1"        Merge other configs
optimizer profile  gpu profile
    │                    │                    │
    ▼                    ▼                    ▼
optimizer_config   gpu_config          training_config
(filled with muon  (filled with     (unchanged)
 settings)         gpu_1 settings)
    
    └────────────────────┬────────────────────┘
                         │
                         ▼
              Complete Config Object
                         │
                         ▼
          Use in training code
```

---

## How to Switch Configurations

### From AdamW → Muon
```yaml
# Before:
active_profiles:
  optimizer: "adamw"   ← Change this
  gpu: "gpu_0"

# After:
active_profiles:
  optimizer: "muon"    ← To muon
  gpu: "gpu_0"
```

### From GPU 0 → GPU 1
```yaml
# Before:
active_profiles:
  optimizer: "adamw"
  gpu: "gpu_0"         ← Change this

# After:
active_profiles:
  optimizer: "adamw"
  gpu: "gpu_1"         ← To gpu_1
```

### From Single GPU → Multi-GPU
```yaml
# Before:
active_profiles:
  optimizer: "adamw"
  gpu: "gpu_0"         ← Change this

# After:
active_profiles:
  optimizer: "adamw"
  gpu: "gpu_multi"     ← To multi-GPU (0,1)
```

---

## Command Examples

```bash
# Train with default (adamw + gpu_0)
bash scripts/train_multitask_dataset.sh

# Custom config (after editing config.yaml)
bash scripts/train_multitask_dataset.sh ./config.yaml

# Different config file (if you want to keep multiple versions)
# (Not needed anymore, but possible)
bash scripts/train_multitask_dataset.sh ./config-backup.yaml
```

---

## Profile Names

### Optimizer Profiles
```
"adamw"   → Adam with Momentum optimizer
"muon"    → Muon orthogonal optimizer
```

### GPU Profiles
```
"gpu_0"      → Single GPU (GPU 0)
"gpu_1"      → Single GPU (GPU 1)
"gpu_multi"  → Multiple GPUs (0,1) with DDP
"cpu"        → CPU-only (for testing)
```

---

## What Each Profile Contains

### Optimizer Profile
```yaml
optimizer_profiles:
  adamw:
    optimizer_type: "adamw"
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_epsilon: 1e-8
```

### GPU Profile
```yaml
gpu_profiles:
  gpu_0:
    device_ids: "0"
    use_distributed: false
    device_map: "auto"
    use_tf32: true
```

---

## Benefits of Unified Config

| Before | After |
|--------|-------|
| Multiple config files (config.yaml, config_adamw.yaml, config_muon.yaml, config_gpu_0.yaml, config_gpu_1.yaml) | **Single config.yaml** |
| Maintain consistency across files manually | **Profiles enforce consistency** |
| Hard to see what's different | **Change one line to switch profiles** |
| Duplicate configuration sections | **DRY - profiles reuse common sections** |
| Confusing which file to use | **Always use config.yaml** |

---

## Summary Table

| Task | Before | After |
|------|--------|-------|
| Switch optimizer | Edit config_muon.yaml | Edit line in config.yaml |
| Switch GPU | Edit config_gpu_1.yaml | Edit line in config.yaml |
| Add new combination | Create new config file | Add profile in config.yaml |
| Files to maintain | 5+ separate files | 1 config.yaml |

**Result**: Simpler, cleaner, more maintainable! ✨
