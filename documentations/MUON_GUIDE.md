# Muon Optimizer Integration Guide

## Overview

The training pipeline now supports both **AdamW** and **Muon** optimizers. You can select which optimizer to use by setting the `optimizer_type` in `config.yaml`.

## Key Differences

### AdamW
- Standard Adam optimizer with weight decay
- Works well with learning rates tuned for adaptive methods
- Default choice for most tasks
- All parameters are optimized with the same algorithm

### Muon (Moment Up Normalization)
- Optimizes 2D parameters (weight matrices) with Newton-Schulz orthogonalization
- Non-2D parameters (bias, embeddings) are automatically handled with AdamW
- Better for LLM training according to recent research
- Learning rate and weight decay tuned separately
- **Important**: Muon parameters are automatically separated from non-2D parameters

## Configuration

### Using AdamW (Default)

```yaml
optimizer_config:
  optimizer_type: "adamw"  # or just use default
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1e-8

training_config:
  learning_rate: 0.0002
  weight_decay: 0.01
```

### Using Muon

```yaml
optimizer_config:
  optimizer_type: "muon"
  
  # Muon-specific hyperparameters
  muon_lr: 0.002           # Learning rate for 2D parameters
  muon_weight_decay: 0.1   # Weight decay for Muon
  muon_momentum: 0.95      # Momentum factor
  muon_nesterov: true      # Enable Nesterov momentum
  muon_ns_coefficients: [3.4445, -4.775, 2.0315]  # Newton-Schulz coefficients
  muon_eps: 1e-7          # Epsilon for numerical stability
  muon_ns_steps: 5         # Number of Newton-Schulz iterations
  muon_adjust_lr_fn: "match_rms_adamw"  # LR adjustment: "original" or "match_rms_adamw"

training_config:
  learning_rate: 0.0002    # Fallback LR for non-2D params (AdamW)
  weight_decay: 0.01       # Fallback weight decay for non-2D params
```

## How Muon Works with LLM Parameters

When using Muon with LLaMA or other LLMs:

1. **2D Parameters (optimized by Muon)**:
   - Linear layer weights (q_proj, v_proj, k_proj, o_proj, up_proj, down_proj)
   - These are weight matrices that benefit from orthogonal updates

2. **Non-2D Parameters (optimized by AdamW)**:
   - Biases (bias)
   - Embeddings (embed_tokens)
   - Layer norms (norm.weight)
   - Any 1D parameters

The optimizer factory automatically detects parameter dimensions and applies the appropriate optimization algorithm.

## Training Commands

### Default (AdamW)
```bash
# Uses AdamW from config
bash scripts/train_multitask_dataset.sh
```

### With Muon
```bash
# First, update config.yaml to set optimizer_type: "muon"
# Then run:
bash scripts/train_multitask_dataset.sh
```

### With Custom Config
```bash
# Create a new config file (e.g., config_muon.yaml)
cp config.yaml config_muon.yaml

# Edit config_muon.yaml and set:
# optimizer_type: "muon"
# and adjust muon_* parameters as needed

# Run with custom config:
bash scripts/train_multitask_dataset.sh ./config_muon.yaml
```

## Muon Hyperparameters Explained

### muon_lr (Learning Rate)
- Typical range: 0.001 to 0.01
- For LLMs: Often 10x larger than AdamW's learning rate
- Default: 0.002

### muon_weight_decay
- Typical range: 0.05 to 0.2
- Controls L2 regularization strength
- Default: 0.1

### muon_momentum
- Typical range: 0.9 to 0.99
- Controls acceleration of updates
- Default: 0.95 (recommended for stability)

### muon_nesterov
- Boolean flag for Nesterov momentum
- Improves convergence
- Default: true

### muon_ns_coefficients
- Coefficients for Newton-Schulz orthogonalization
- Default: (3.4445, -4.775, 2.0315) - these are optimized values
- Usually don't need to be changed

### muon_ns_steps
- Number of Newton-Schulz iterations
- More iterations = more accurate orthogonalization, slower computation
- Typical range: 3 to 7
- Default: 5

### muon_adjust_lr_fn
- Learning rate adjustment strategy:
  - `"original"`: Keller's original implementation (scales by max(A,B))
  - `"match_rms_adamw"`: Moonshot implementation (matches AdamW's RMS)
- For tuned AdamW learning rates, use `"match_rms_adamw"`
- Default: "match_rms_adamw"

## Performance Tips

### For Math Training (Your Use Case)
```yaml
optimizer_config:
  optimizer_type: "muon"
  muon_lr: 0.003           # Slightly higher for math tasks
  muon_weight_decay: 0.1
  muon_momentum: 0.95
  muon_ns_steps: 5
  muon_adjust_lr_fn: "match_rms_adamw"
```

### For Stability (Conservative)
```yaml
optimizer_config:
  optimizer_type: "muon"
  muon_lr: 0.0015          # Lower learning rate
  muon_weight_decay: 0.15  # Higher regularization
  muon_momentum: 0.97      # Higher momentum
  muon_ns_steps: 7         # More iterations
```

### For Speed (Faster but Less Stable)
```yaml
optimizer_config:
  optimizer_type: "muon"
  muon_lr: 0.005           # Higher learning rate
  muon_weight_decay: 0.05  # Lower regularization
  muon_momentum: 0.9       # Standard momentum
  muon_ns_steps: 3         # Fewer iterations
```

## Comparison Experiments

To compare AdamW vs Muon:

1. **Config for AdamW**:
   ```yaml
   optimizer_config:
     optimizer_type: "adamw"
     adam_beta1: 0.9
   
   training_config:
     run_name: "llama-adamw-bs2-gas8"
   ```

2. **Config for Muon**:
   ```yaml
   optimizer_config:
     optimizer_type: "muon"
     muon_lr: 0.002
     muon_adjust_lr_fn: "match_rms_adamw"
   
   training_config:
     run_name: "llama-muon-bs2-gas8"
   ```

3. **Run both**:
   ```bash
   # AdamW
   cp config.yaml config_adamw.yaml
   # Edit config_adamw.yaml
   bash scripts/train_multitask_dataset.sh ./config_adamw.yaml
   
   # Muon
   cp config.yaml config_muon.yaml
   # Edit config_muon.yaml
   bash scripts/train_multitask_dataset.sh ./config_muon.yaml
   ```

4. **Compare on W&B Dashboard**:
   - Monitor loss curves
   - Check validation metrics
   - Compare final performance
   - Track convergence speed

## PyTorch Version Requirement

Muon optimizer requires **PyTorch 2.10.0 or newer** (added in PyTorch 2.10).

Check your version:
```bash
python -c "import torch; print(torch.__version__)"
```

If you have an older version, update PyTorch:
```bash
pip install --upgrade torch torchvision torchaudio
```

## Implementation Details

### Optimizer Factory (`colm/train/optimizer_factory.py`)

The optimizer factory handles:
- Parameter dimension detection (2D vs non-2D)
- Separate Muon and AdamW updates
- Parameter group configuration
- Proper logging of optimizer setup

### Integration in Training (`colm/train/train_multitask.py`)

The `MultiTaskTrainer` class:
- Accepts config dictionary
- Calls optimizer factory in `create_optimizer()` method
- Supports both AdamW and Muon seamlessly
- Falls back to default AdamW if no config provided

### Configuration (`config.yaml`)

New fields in `optimizer_config`:
- `optimizer_type`: Select "adamw" or "muon"
- Complete set of Muon hyperparameters
- AdamW hyperparameters still available

## Troubleshooting

### Issue: "RuntimeError: Muon is not available"
**Solution**: Update PyTorch to version 2.10.0 or later

### Issue: "Unknown optimizer type"
**Solution**: Check that `optimizer_type` in config.yaml is either "adamw" or "muon"

### Issue: "2D parameters (Muon): 0"
**Solution**: The model has no 2D parameters, which can happen with:
- Certain architecture changes
- Some quantization methods
- Check model structure with `model.print_trainable_parameters()`

### Issue: Training is slower with Muon
**Possible causes**:
- `muon_ns_steps` too high - try reducing to 3-5
- Using `adjust_lr_fn: "original"` which does extra computation
- GPU memory bandwidth limited

**Solutions**:
- Reduce `muon_ns_steps` for faster computation
- Ensure GPU is fully utilized
- Increase batch size if possible

## References

- **Paper**: [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)
- **Scalability**: [Muon is Scalable for LLM Training](https://arxiv.org/pdf/2502.16982)
- **PyTorch Docs**: [torch.optim.Muon](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)

## Next Steps

1. ✅ Update `config.yaml` with `optimizer_type: "adamw"` or `"muon"`
2. ✅ Regenerate dataset: `python colm/data/load_math_datasets.py`
3. ✅ Run training: `bash scripts/train_multitask_dataset.sh`
4. ✅ Monitor on W&B: Check convergence and performance
5. ✅ Experiment: Try different hyperparameters

---

**Questions?** Check the configuration values match the official PyTorch documentation.
