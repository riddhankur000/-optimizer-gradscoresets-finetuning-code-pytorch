# YAML Configuration System Guide

## Overview

The training pipeline now uses a centralized YAML configuration file `config.yaml` that contains all parameters for model loading, tokenization, LoRA adaptation, data handling, and training. This replaces inline bash parameters and enables easier experimentation and reproducibility.

## File Structure

- **`config.yaml`** - Centralized configuration with all parameters organized by section
- **`colm/train/config_loader.py`** - Python utility module to load and parse YAML config
- **`colm/train/train_multitask.py`** - Main training script updated to load config
- **`scripts/train_multitask_dataset.sh`** - Bash wrapper that passes config file to training script

## Configuration Sections

### 1. model_config
- `model_id`: HuggingFace model identifier (e.g., "meta-llama/Llama-3.1-8B")
- `torch_dtype`: PyTorch data type (bfloat16 for Llama-3.1-8B)
- `device_map`: Device mapping strategy (auto)

### 2. tokenizer_config
- `tokenizer_path`: Path to tokenizer (usually same as model_id)
- `model_max_length`: Max sequence length (512)
- `padding_side`: Padding side (right)

### 3. lora_config
- `lora_rank`: LoRA rank (128)
- `lora_alpha`: LoRA alpha scaling (512)
- `lora_dropout`: LoRA dropout (0.05)
- `target_modules`: List of layer names to apply LoRA (q_proj, v_proj, k_proj, o_proj, up_proj, down_proj for Llama)

### 4. dataset_config
- `dataset_path`: Path to combined dataset directory
- `dataset_names`: List of datasets (MetaMathQA, GSM8K)
- `max_seq_length`: Max sequence length for tokenization

### 5. training_config
- `output_dir`: Output directory for checkpoints
- `max_steps`: Total training steps (1024)
- `per_device_train_batch_size`: Train batch size per device (2)
- `per_device_eval_batch_size`: Eval batch size per device (4)
- `gradient_accumulation_steps`: Gradient accumulation steps (8)
- `effective_batch_size`: Computed as batch_size × gradient_accumulation
- `learning_rate`: Learning rate (0.0002)
- `lr_scheduler_type`: LR scheduler type (linear)
- `warmup_ratio`: Warmup ratio (0.03 = 3%)
- `weight_decay`: Weight decay (0.01)
- `max_grad_norm`: Max gradient norm (1.0)
- `bf16`: Use bfloat16 precision (true)
- `fp16`: Use float16 precision (false)
- `save_steps`: Save checkpoint every N steps (256)
- `eval_steps`: Evaluate every N steps (256)
- `logging_steps`: Log every N steps (10)
- `seed`: Random seed (0)
- `report_to`: Report metrics to wandb

### 6. optimizer_config
- `adam_beta1`: AdamW beta1/momentum (0.9)
- `adam_beta2`: AdamW beta2 (0.999)
- `adam_epsilon`: AdamW epsilon (1e-8)

### 7. multitask_config
- `track_per_task_loss`: Track loss per task (true)
- `eval_per_task`: Evaluate per task (true)

### 8. wandb_config
- `project`: W&B project name
- `entity`: W&B entity
- `tags`: List of tags for experiment tracking
- `notes`: Experiment notes/description

### 9. system_config
- `num_workers`: Number of dataloader workers (4)
- `pin_memory`: Pin memory for faster data loading (true)

## Usage

### Regenerate Dataset with Llama Tokenizer

Before first training run, regenerate the dataset with Llama-3.1-8B tokenizer:

```bash
cd /home1/riddhankur/adamw_vs_muon_2/llm-experiments
rm -rf ./colm_math_combined_dataset
python colm/data/load_math_datasets.py
```

### Run Training with Default Config

Uses `./config.yaml` by default:

```bash
bash scripts/train_multitask_dataset.sh
```

### Run Training with Custom Config

Pass custom config file path:

```bash
bash scripts/train_multitask_dataset.sh /path/to/custom_config.yaml
```

### Example: Run with Different Learning Rate

1. Copy and modify config:
   ```bash
   cp config.yaml config_high_lr.yaml
   ```

2. Edit `config_high_lr.yaml`:
   ```yaml
   training_config:
       learning_rate: 0.0005  # Changed from 0.0002
   ```

3. Run training:
   ```bash
   bash scripts/train_multitask_dataset.sh ./config_high_lr.yaml
   ```

## How It Works

### Flow Diagram

```
config.yaml (YAML parameters)
    ↓
scripts/train_multitask_dataset.sh (passes config file path as argument)
    ↓
colm/train/train_multitask.py (receives config file from sys.argv[1])
    ↓
colm/train/config_loader.py (loads YAML and parses it)
    ↓
Training components (model, tokenizer, LoRA, training)
```

### Python Integration

In `train_multitask.py`:

```python
# Check if config.yaml is provided as first argument
if config_file:
    config = load_config_yaml(config_file)
    training_args = config_dict_to_hf_training_args(config)
    model_config = get_model_config(config)
    # ... etc
else:
    # Fall back to command-line arguments
    # Uses HfArgumentParser for traditional CLI args
```

## Config Loader API

### Functions in `colm/train/config_loader.py`

```python
# Load YAML file
config = load_config_yaml("./config.yaml")

# Extract specific sections
model_config = get_model_config(config)
lora_config = get_lora_config(config)
dataset_config = get_dataset_config(config)
training_config = get_training_config(config)
optimizer_config = get_optimizer_config(config)

# Convert to HF TrainingArguments
training_args = config_dict_to_hf_training_args(config)

# Print formatted summary
print_config_summary(config)
```

## Advantages

1. **Centralized Parameters**: All hyperparameters in one YAML file
2. **Easy Experimentation**: Copy config and modify for different experiments
3. **Reproducibility**: Config file documents exact parameters used
4. **Version Control**: Track config evolution with git
5. **Documentation**: YAML is self-documenting
6. **Backward Compatible**: Still supports command-line arguments

## Best Practices

1. **Commit configs to git**:
   ```bash
   git add config.yaml config_*.yaml
   git commit -m "Add training configs for experiments"
   ```

2. **Name config files descriptively**:
   - `config.yaml` - Default/baseline
   - `config_high_lr.yaml` - Higher learning rate
   - `config_small_rank.yaml` - Smaller LoRA rank

3. **Document changes** in commit messages:
   ```
   Add config_large_rank.yaml with rank=256 for larger capacity
   ```

4. **Use config summary output**:
   - Training logs show which config was loaded
   - Verify config values before starting training

## Troubleshooting

### Config File Not Found
```
ERROR: Config file not found: ./config.yaml
```
**Solution**: Ensure config file exists in current directory or provide full path

### Config Key Missing
```
KeyError: 'model_id'
```
**Solution**: Check config.yaml has all required sections (see Configuration Sections above)

### Type Mismatch
```
TypeError: 'int' expected
```
**Solution**: Verify YAML types match expected Python types (e.g., learning_rate should be float, not string)

## Next Steps

1. ✅ Regenerate dataset with Llama tokenizer
2. ✅ Run training: `bash scripts/train_multitask_dataset.sh`
3. ✅ Monitor training on W&B dashboard
4. ✅ Experiment with parameter variations using different config files

---

**Current Config Values**:
- Model: Llama-3.1-8B
- Precision: BF16
- Learning Rate: 0.0002
- Optimizer Beta1: 0.9 (momentum)
- LoRA Rank: 128
- Training Steps: 1024
- Dataset: MetaMathQA + GSM8K (362K train examples)
