# Unified Configuration with Profiles

**All optimizer and GPU configurations are now in a single `config.yaml` file with profiles!**

No need to maintain separate config files anymore. Just edit `config.yaml` and select which profile to use.

---

## Quick Start

### Train with AdamW on GPU 0 (Default)
```bash
bash scripts/train_multitask_dataset.sh
# Uses ./config.yaml with optimizer: "adamw" and gpu: "gpu_0"
```

### Train with Muon on GPU 1
Edit `config.yaml`:
```yaml
active_profiles:
  optimizer: "muon"    # ← Change to muon
  gpu: "gpu_1"         # ← Change to gpu_1
```

Then run:
```bash
bash scripts/train_multitask_dataset.sh
```

---

## How It Works

### Structure of config.yaml

The unified `config.yaml` has 3 new sections:

#### 1. Active Profiles (SELECT HERE)
```yaml
active_profiles:
  optimizer: "adamw"  # Select optimizer profile
  gpu: "gpu_0"        # Select GPU profile
```

#### 2. Optimizer Profiles
```yaml
optimizer_profiles:
  adamw:
    optimizer_type: "adamw"
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_epsilon: 1e-8
    
  muon:
    optimizer_type: "muon"
    muon_lr: 0.002
    muon_weight_decay: 0.1
    muon_momentum: 0.95
    # ... more muon settings
```

#### 3. GPU Profiles
```yaml
gpu_profiles:
  gpu_0:
    device_ids: "0"
    use_distributed: false
    device_map: "auto"
    use_tf32: true
    
  gpu_1:
    device_ids: "1"
    use_distributed: false
    device_map: "auto"
    use_tf32: true
    
  gpu_multi:
    device_ids: "0,1"
    use_distributed: true
    device_map: "auto"
    use_tf32: true
    
  cpu:
    device_ids: null
    use_distributed: false
    device_map: "cpu"
    use_tf32: false
```

---

## Available Profiles

### Optimizer Profiles
| Profile | Use Case |
|---------|----------|
| **adamw** | Standard training (default, stable) |
| **muon** | Fast orthogonal updates (best for 2D params) |

### GPU Profiles
| Profile | Use Case |
|---------|----------|
| **gpu_0** | Single GPU training (GPU 0) |
| **gpu_1** | Single GPU training (GPU 1) |
| **gpu_multi** | Multi-GPU distributed training (GPUs 0,1) |
| **cpu** | CPU-only training (for testing) |

---

## Common Use Cases

### Use Case 1: AdamW on GPU 0 (Default)
```yaml
active_profiles:
  optimizer: "adamw"
  gpu: "gpu_0"
```

### Use Case 2: Muon on GPU 0
```yaml
active_profiles:
  optimizer: "muon"
  gpu: "gpu_0"
```

### Use Case 3: AdamW on GPU 1
```yaml
active_profiles:
  optimizer: "adamw"
  gpu: "gpu_1"
```

### Use Case 4: Muon on multiple GPUs
```yaml
active_profiles:
  optimizer: "muon"
  gpu: "gpu_multi"
```

---

## How Profiles Are Applied

1. **Load config.yaml**
   - `load_config_yaml()` reads the file

2. **Apply profiles** 
   - `apply_active_profiles()` function:
     - Reads `active_profiles.optimizer` name
     - Reads `active_profiles.gpu` name
     - Merges selected profiles into `optimizer_config` and `gpu_config`
     - Logs which profiles were applied

3. **Use merged config**
   - Training code uses the merged `optimizer_config`
   - Training code uses the merged `gpu_config`

Example flow:
```
config.yaml with:
  active_profiles:
    optimizer: "muon"
    gpu: "gpu_1"
  optimizer_profiles:
    muon: {...}
  gpu_profiles:
    gpu_1: {...}
  
↓ (apply_active_profiles called)

Resulting config_object has:
  optimizer_config: {
    optimizer_type: "muon",
    muon_lr: 0.002,
    ...
  }
  gpu_config: {
    device_ids: "1",
    use_distributed: false,
    ...
  }
```

---

## Creating Custom Profiles

You can add custom profiles to `config.yaml`:

### Add custom optimizer profile
```yaml
optimizer_profiles:
  adamw:
    # existing adamw config
  muon:
    # existing muon config
  adamw_custom:           # ← NEW
    optimizer_type: "adamw"
    adam_beta1: 0.95       # ← Custom value
    adam_beta2: 0.9999
    adam_epsilon: 1e-9
```

Then use it:
```yaml
active_profiles:
  optimizer: "adamw_custom"  # ← Use custom profile
  gpu: "gpu_0"
```

### Add custom GPU profile
```yaml
gpu_profiles:
  gpu_0:
    # existing
  gpu_1:
    # existing
  gpu_multi_4:            # ← NEW
    device_ids: "0,1,2,3" # ← Use 4 GPUs
    use_distributed: true
    device_map: "auto"
    use_tf32: true
```

Then use it:
```yaml
active_profiles:
  optimizer: "adamw"
  gpu: "gpu_multi_4"      # ← Use custom GPU profile
```

---

## Command Line

### Train with default profiles (adamw + gpu_0)
```bash
bash scripts/train_multitask_dataset.sh
```

### Train with custom config
```bash
# First, edit config.yaml to select profiles, then:
bash scripts/train_multitask_dataset.sh
# or explicitly:
bash scripts/train_multitask_dataset.sh ./config.yaml
```

---

## Backward Compatibility

The code automatically handles missing profiles:
- If `optimizer` profile not found → uses defaults
- If `gpu` profile not found → uses defaults

Always falls back gracefully! ✓

---

## What Changed

### Files Updated
- **config.yaml**: Now has unified structure with profiles
- **colm/train/config_loader.py**: Added `apply_active_profiles()` function
- **scripts/train_multitask_dataset.sh**: Shows which profiles are active

### Files Removed/Archived (No longer needed!)
- ~~config_adamw.yaml~~ → Profiles in config.yaml
- ~~config_muon.yaml~~ → Profiles in config.yaml
- ~~config_gpu_0.yaml~~ → Profiles in config.yaml
- ~~config_gpu_1.yaml~~ → Profiles in config.yaml

**Only `config.yaml` is needed!** 🎉

---

## Troubleshooting

### Problem: "Optimizer profile not found"
**Solution**: Check `active_profiles.optimizer` matches a profile in `optimizer_profiles`

```yaml
active_profiles:
  optimizer: "adamw"   # ← Make sure this exists in optimizer_profiles
```

### Problem: "GPU profile not found"
**Solution**: Check `active_profiles.gpu` matches a profile in `gpu_profiles`

```yaml
active_profiles:
  gpu: "gpu_0"        # ← Make sure this exists in gpu_profiles
```

### Problem: Wrong GPU being used
**Solution**: 
1. Check `active_profiles.gpu` is set correctly
2. Verify selected GPU profile has correct `device_ids`
3. Run: `echo "Selected profiles: $(grep -A2 'active_profiles:' config.yaml)"`

### Problem: Can't find config file
**Solution** Run from project root:
```bash
cd /path/to/llm-experiments
bash scripts/train_multitask_dataset.sh
```

---

## Summary

✅ **Single `config.yaml` file** - No separate config files needed  
✅ **Select profiles** - Just change `active_profiles` section  
✅ **Profiles are merged** - Code handles all profile setup automatically  
✅ **Easy to extend** - Add custom profiles as needed  
✅ **Graceful fallbacks** - Missing profiles use defaults  

**To train with different settings:**
```bash
# Edit config.yaml:
active_profiles:
  optimizer: "muon"   # or "adamw"
  gpu: "gpu_1"        # or "gpu_0", "gpu_multi"

# Run training:
bash scripts/train_multitask_dataset.sh
```

Done! 🚀
