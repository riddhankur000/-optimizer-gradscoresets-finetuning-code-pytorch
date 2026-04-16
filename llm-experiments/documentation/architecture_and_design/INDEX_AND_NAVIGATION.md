# 📑 Complete Index: Config-Based Training Setup

## 🎯 Navigation Guide

### For Different Use Cases

#### 👤 "I want to START TRAINING NOW"
1. Make script executable: `chmod +x train.sh`
2. Read: **Quick Summary** (below)
3. Run: `bash train.sh`
4. Monitor: WandB dashboard

---

#### 👤 "I want to UNDERSTAND HOW TO USE THIS"
1. Read: `TRAINING_WITH_CONFIG_YAML.md` (600+ lines, user guide)
2. Edit: `config.yaml` for your setup
3. Run: `bash train.sh`
4. Try overrides: `bash train.sh --optimizer muon --gpu gpu_multi`

---

#### 👤 "I want to UNDERSTAND THE IMPLEMENTATION"
1. Read: `CONFIG_YAML_IMPLEMENTATION_SUMMARY.md` (400+ lines, technical)
2. Review: `colm/train/config_parser.py` (config loading)
3. Review: `colm/train/train_sequential_from_config.py` (training logic)
4. Study: Inline docstrings in Python files

---

#### 👤 "I want to UNDERSTAND THE ARCHITECTURE"
1. Read: `CHECKPOINT_AND_TRAINING_FLOW.md` (checkpoint flow)
2. Read: `CONFIG_YAML_IMPLEMENTATION_SUMMARY.md` (Implementation section)
3. Trace: Model persistence through lines 375-390 in training script

---

## 📂 Complete File Listing

### NEW Python Modules (colm/train/)

| File | Lines | Purpose | Key Classes |
|------|-------|---------|------------|
| `config_parser.py` | 400+ | YAML config loading | `ConfigLoader`, `OptimizerConfig`, `GPUConfig` |
| `train_sequential_from_config.py` | 600+ | Main training with config | `MonitoringCallbackSeq` |

### NEW Shell Scripts

| File | Lines | Purpose | Usage |
|------|-------|---------|-------|
| `train.sh` | 180+ | Easy training launcher | `bash train.sh [--optimizer OPT] [--gpu PROFILE]` |

### NEW Documentation

| File | Lines | Purpose | Audience |
|------|-------|---------|----------|
| `TRAINING_WITH_CONFIG_YAML.md` | 600+ | User guide with examples | End users |
| `CONFIG_YAML_IMPLEMENTATION_SUMMARY.md` | 400+ | Technical implementation | Developers |
| `IMPLEMENTATION_COMPLETE.md` | 400+ | Final report and summary | Everyone |
| **This file** | - | Navigation index | Everyone |

### EXISTING Configuration (Updated)

| File | Purpose | Status |
|------|---------|--------|
| `config.yaml` | Main configuration file | PRIMARY (updated to use with new scripts) |
| `configs/sequential_riemannian_config.json` | Legacy JSON config | REFERENCE (still works) |

### EXISTING Documentation (Reference)

| File | Purpose | Status |
|------|---------|--------|
| `CHECKPOINT_AND_TRAINING_FLOW.md` | Architecture explanation | REFERENCE |
| `SEQUENTIAL_RIEMANNIAN_TRAINING_GUIDE.md` | Legacy training guide | REFERENCE |
| `CODEBASE_CHANGES_SEQUENTIAL_TRAINING.md` | Code changes log | REFERENCE |
| `QUICK_REFERENCE_SEQUENTIAL_CHANGES.md` | Quick lookup | REFERENCE |
| `FILE_STRUCTURE_AND_LOCATIONS.md` | File mapping | REFERENCE |

### EXISTING Scripts (Still Working)

| File | Purpose | Status |
|------|---------|--------|
| `colm/train/train_sequential_riemannian.py` | Legacy training script | WORKING (old approach) |
| `colm/train/train.py` | Original CoLM trainer | WORKING (original) |
| All other CoLM files | Core functionality | WORKING (unchanged) |

---

## 🚀 Quick Start Summary

### Step 1: Setup
```bash
cd /data/riddhankur/PROJECTS/GREATS_COLM_pytorch/local/llm-experiments
chmod +x train.sh
```

### Step 2: Configure
```bash
# Edit config.yaml to customize:
vim config.yaml

# Key settings:
# - active_profiles.optimizer: "adamw" or "muon"
# - active_profiles.gpu: "gpu_0", "gpu_1", "gpu_multi", or "cpu"
# - model_config.model_id: Your model
# - dataset_config.dataset_path: Your dataset
```

### Step 3: Run
```bash
# With config settings
bash train.sh

# Or with quick override (no config edit)
bash train.sh --optimizer muon --gpu gpu_multi
```

### Step 4: Monitor
- Open WandB dashboard
- Watch training curves
- Check metrics in real-time

---

## 📊 Feature Matrix

```
┌─────────────────────────────────────────────────────────┐
│            Config-Based Training Features               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ✅ AdamW Optimizer (default, stable)                  │
│  ✅ Muon Optimizer (optional, efficient)               │
│  ✅ Automatic fallback if Muon unavailable             │
│                                                         │
│  ✅ Single GPU training (GPU 0 or 1)                   │
│  ✅ Multi-GPU distributed training (DDP)               │
│  ✅ CPU training (testing/debugging)                   │
│                                                         │
│  ✅ Sequential multi-task training                     │
│  ✅ Model persistence (cumulative learning)            │
│  ✅ Eval loss tracking (overfitting detection)         │
│                                                         │
│  ✅ 20+ metrics to WandB                               │
│  ✅ Command-line overrides                             │
│  ✅ Easy configuration file                            │
│                                                         │
│  ✅ Fully backward compatible                          │
│  ✅ Zero breaking changes                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔄 Usage Patterns

### Pattern 1: Use Config File
```bash
# Edit config.yaml once
bash train.sh

# Training starts with configured settings
```

### Pattern 2: Quick Override
```bash
# No config edit needed
bash train.sh --optimizer muon --gpu gpu_multi

# Overrides config.yaml settings
```

### Pattern 3: Custom Config
```bash
# Multiple configs for different experiments
bash train.sh --config config_adamw_gpu0.yaml
bash train.sh --config config_muon_gpu_multi.yaml
```

### Pattern 4: Direct Python Script
```bash
# For advanced users or scripting
python colm/train/train_sequential_from_config.py ./config.yaml
```

---

## 💡 Common Questions

### Q: Which optimizer should I use?
**A:** 
- **AdamW**: Better stability, proven, always works
- **Muon**: Better efficiency, newer, requires installation

Start with AdamW, then try Muon if you want higher throughput.

---

### Q: Single GPU or Multi-GPU?
**A**:
- **gpu_0 or gpu_1**: For prototyping, single GPU available
- **gpu_multi**: For production, have 2+ GPUs

Multi-GPU gives ~2x speedup on 2 GPUs.

---

### Q: How do I compare optimizer performance?
**A**:
```bash
# Run 1: AdamW
bash train.sh --optimizer adamw --config ./config.yaml

# Run 2: Muon (same everything else)
bash train.sh --optimizer muon --config ./config.yaml

# Compare in WandB: Same run name except optimizer
```

---

### Q: How do I track progress?
**A**:
- WandB dashboard (real-time, cloud)
- Console logs (real-time, local)
- Checkpoints saved (step-based recovery)

---

## 🎓 Architecture Quick Ref

### Config Loading Flow
```
config.yaml
    ↓
ConfigLoader.parse_yaml()
    ↓
get_model_config()
get_optimizer_config()  ← Select AdamW or Muon
get_gpu_config()        ← Select single or multi-GPU
    ↓
train_sequential_from_config.py
    ↓
Setup model, optimizer, GPU
    ↓
Sequential task loop (model persists!)
    ↓
WandB logging
```

### Model Persistence
```
Load model ONCE (outside loop)
    ↓
Task 0: train (update weights in-place)
    ↓
Task 1: train (continue from task 0 weights)
    ↓
Task 2: train (continue from task 0+1 weights)
    ↓
Result: Cumulative learning across tasks
```

### Optimizer Selection
```
config.yaml: optimizer: "adamw"
    ↓
ConfigLoader: optimizer_config.optimizer_type = "adamw"
    ↓
training script: HF Trainer handles AdamW
    ↓
Result: Adam with configured beta1, beta2, etc.

------

config.yaml: optimizer: "muon"
    ↓
ConfigLoader: optimizer_config.optimizer_type = "muon"
    ↓
training script: Try import Muon
    ✓ If available: Create Muon optimizer
    ✗ If not: Fall back to AdamW (warning shown)
    ↓
Result: Muon or AdamW+warning
```

---

## 📈 Metrics For Analysis

### Always Logged
- Training loss (per step)
- Eval loss (per eval)
- Perplexity (both)
- Gradient norms
- GPU/CPU usage

### Auto-Calculated  
- `overfit_ratio` = eval_loss / train_loss
- Task-specific summaries
- Per-GPU memory stats

### For Comparison
- Run name includes: model, optimizer, lr, lora rank
- WandB allows easy comparison
- Results saved per experiment

---

## 🔧 Troubleshooting Quick Guide

| Problem | Solution | Docs |
|---------|----------|------|
| Muon not found | `pip install muon` or just use AdamW | TRAINING_WITH_CONFIG_YAML.md |
| Multi-GPU not working | Check `gpu: gpu_multi` and `nvidia-smi` | TRAINING_WITH_CONFIG_YAML.md |
| OOM error | Reduce batch size, increase gradual accum | TRAINING_WITH_CONFIG_YAML.md |
| WandB not logging | `pip install wandb` + `wandb login` | TRAINING_WITH_CONFIG_YAML.md |
| Training slow | Increase batch size, reduce eval frequency | TRAINING_WITH_CONFIG_YAML.md |

---

## 📚 Reading Order (Recommended)

### For Immediate Use (15 min)
1. This file (INDEX) - 5 min
2. Quick Start Summary (above) - 2 min
3. Run `bash train.sh` - 3 min
4. Check WandB dashboard - 5 min

### For Full Understanding (1 hour)
1. Quick Start Summary - 5 min
2. `TRAINING_WITH_CONFIG_YAML.md` - 35 min
3. Guide examples - 10 min
4. Run your own experiment - 10 min

### For Deep Dive (3 hours)
1. `CONFIG_YAML_IMPLEMENTATION_SUMMARY.md` - 40 min
2. `config_parser.py` code + docstrings - 40 min
3. `train_sequential_from_config.py` code - 40 min
4. `CHECKPOINT_AND_TRAINING_FLOW.md` - 30 min
5. Hands-on with debugger - 30 min

---

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] Script executable: `ls -la train.sh` shows `x` permission
- [ ] Config loads: `python -c "from colm.train.config_parser import ConfigLoader; ConfigLoader('./config.yaml')"` succeeds
- [ ] Help shows: `bash train.sh --help` displays options
- [ ] Dataset exists: `ls -la {dataset_path}` shows files
- [ ] GPU available: `nvidia-smi` shows GPU(s)
- [ ] Dependencies: `pip list | grep -E "torch|wandb|peft"` shows packages
- [ ] WandB login: `wandb login` succeeds (if using WandB)

---

## 🚀 Launch Command Cheat Sheet

```bash
# Most basic
bash train.sh

# Standard experiment: AdamW + single GPU
bash train.sh --optimizer adamw --gpu gpu_0

# Efficient experiment: Muon + multi-GPU
bash train.sh --optimizer muon --gpu gpu_multi

# Custom config
bash train.sh --config ./experiments/exp1.yaml

# Multiple experiments (loop)
for opt in adamw muon; do
  bash train.sh --optimizer $opt
done

# With logging to file
bash train.sh 2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log
```

---

## 📞 Support Resources

### Documentation
- **User Guide**: `TRAINING_WITH_CONFIG_YAML.md`
- **Technical**: `CONFIG_YAML_IMPLEMENTATION_SUMMARY.md`
- **Architecture**: `CHECKPOINT_AND_TRAINING_FLOW.md`
- **Index/Nav**: THIS FILE

### Code Reference
- **Config Loader**: `colm/train/config_parser.py`
- **Training Script**: `colm/train/train_sequential_from_config.py`
- **Wrapper**: `train.sh`

### Examples
- **Config**: `config.yaml` (in repo)
- **User Guide**: Extended examples in `TRAINING_WITH_CONFIG_YAML.md`

---

## 🎯 Summary

✅ **Everything is set up and ready to use**

**Files Created**: 2 Python + 1 Shell + 2 Docs = 5 total
**Time to Setup**: < 5 minutes
**Time to First Training**: < 15 minutes
**Complexity**: Simple - One command (`bash train.sh`)

**Next Action**: `bash train.sh`

