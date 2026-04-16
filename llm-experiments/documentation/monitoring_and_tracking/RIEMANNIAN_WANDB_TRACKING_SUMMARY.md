# Riemannian Codebase WandB Tracking Implementation

## 1. WandB Initialization Strategy

The Riemannian codebase **does NOT explicitly call `wandb.init()`**. Instead, it relies on the Hugging Face Transformers library's automatic wandb integration via the `WandbCallback`.

### Key Pattern:
- Uses **`report_to: wandb`** in the trainer configuration (YAML)
- Transformers library automatically handles initialization when `report_to` includes "wandb"
- WandB login is done via command line (commented out in notebooks):
  ```python
  # os.system(f'wandb login {api_keys.get_secret("wandb")}')
  ```

---

## 2. WandB Initialization Configuration (Implicit)

The transformers library's `SFTTrainer` with `report_to: wandb` automatically initializes wandb with:

### From `trainer_config` in config.yaml:
```yaml
trainer_config:
  run_name: ${exp_name}/${run_name}-cfg${cfg_no}
  output_dir: bogachevv/${exp_name}-${run_name}-cfg${cfg_no}
  report_to: wandb
  # ... other training args
```

### Typical Initialization Arguments (implicit in SFTConfig):
- **`run_name`**: e.g., `"Llama-32-1b-optimizers/Riemanian-cfg6"` (hierarchical naming with custom run name)
- **`project`**: Defaults to HF Transformers settings (not explicitly set)
- **`entity`**: Uses default wandb account entity
- **`output_dir`**: `"bogachevv/${exp_name}-${run_name}-cfg${cfg_no}"` (also used for checkpoint saving)
- **`tags`**: Not explicitly configured in this codebase

### Example Configurations Used:

**Optimization/Fine-tuning Config:**
```yaml
exp_name: Llama-32-1b-optimizers
run_name: Riemanian
cfg_no: 6
trainer_config:
  run_name: ${exp_name}/${run_name}-cfg${cfg_no}  # Becomes: "Llama-32-1b-optimizers/Riemanian-cfg6"
  output_dir: bogachevv/${exp_name}-${run_name}-cfg${cfg_no}
  report_to: wandb
```

**Initialization Config:**
```yaml
run_name: Optimal
trainer_config:
  run_name: ${run_name}-init  # Becomes: "Optimal-init"
  report_to: none  # ← Note: Initialization does NOT use wandb!
```

---

## 3. Metrics Logged to WandB During Training

### A. Default Metrics (from Transformers SFTTrainer)

The transformers library automatically logs:
- **`loss`**: Training loss (logged every `logging_steps`)
- **`eval_loss`**: Evaluation loss (logged during `eval_strategy: steps`)
- **`learning_rate`**: Current learning rate
- **`epoch`**: Current training epoch
- **`global_step`**: Training step counter

### B. Custom Gradient/Weight Logging

The Riemannian codebase implements **`LoRAWandbLogger`** (custom callback) for detailed Layer-wise logging:

**File**: `src/loggers.py`

**Activation**: Enabled when `detailed_lora_logs: true` in config

**Logged Metrics** (per LoRA layer):
```python
# For each layer in LoRA model:
class LoRAWandbLogger(WandbCallback):
    def on_optimizer_step(self, args, state, control, **kwargs):
        # Parse layer information
        layer_no, module_name, factor = self._parse_layer_name(name)
        log_name = f'{factor}/layer_{layer_no}.{module_name}'
        
        # Log metrics
        wandb.log({
            f'grad_{log_name}': grad_norm,        # L2 norm of gradients
            f'weight_{log_name}': weight_norm     # L2 norm of weights
        }, global_step)
```

**Example logged metrics**:
```
grad_lora_A/layer_0.q_proj: 0.152
weight_lora_A/layer_0.q_proj: 2.341
grad_lora_B/layer_0.q_proj: 0.087
weight_lora_B/layer_0.q_proj: 1.234
grad_lora_A/layer_1.k_proj: 0.201
weight_lora_A/layer_1.k_proj: 2.156
... (for all target_modules and layers)
```

### C. When Detailed Logging is Enabled

```yaml
detailed_lora_logs: true  # Enable detailed LoRA gradient/weight tracking
target_modules:  # These modules are tracked
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - up_proj
  - down_proj
  - gate_proj
```

---

## 4. Evaluation/Validation Metrics

### Configuration in trainer_config:
```yaml
trainer_config:
  eval_strategy: steps        # Evaluate every N steps
  eval_steps: 128             # Evaluation frequency
  logging_steps: 16           # Logging frequency  
  load_best_model_at_end: false  # Don't load best model
```

### Metrics Logged During Evaluation:
- **`eval_loss`**: Evaluation loss computed on validation dataset
- Logged at interval specified by `eval_steps`

### How Evaluation Dataset is Used:
```python
def get_trainer(config, model, tokenizer, train_dataset, val_dataset):
    # Val dataset is filtered to specific size if configured
    if config.get('val_ds_size', None):
        if config.get('val_ds_seed', None) is not None:
            val_dataset = val_dataset.shuffle(config.val_ds_seed)
        val_dataset = val_dataset.select(range(config.val_ds_size))
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Passed here
        # ...
    )
```

---

## 5. Run Name and Tags Configuration

### A. Run Name Format

Uses **template variables** for hierarchical organization:

```yaml
# Optimization template:
run_name: ${exp_name}/${run_name}-cfg${cfg_no}
# Example: "Llama-32-1b-optimizers/Riemanian-cfg6"

# Initialization template:
run_name: ${run_name}-init
# Example: "Optimal-init"
```

### B. Output Directory Format

Also templated (used for checkpoint saving):
```yaml
# Optimization
output_dir: bogachevv/${exp_name}-${run_name}-cfg${cfg_no}
# Example: "bogachevv/Llama-32-1b-optimizers-Riemanian-cfg6"

# Initialization
output_dir: bogachevv/${run_name}-init
# Example: "bogachevv/Optimal-init"
```

### C. Tags

**No explicit tags are set in this codebase.** Tags could be added to the trainer config if needed:
```yaml
# Not used in current implementations, but could be added:
# wandb_tags:
#   - optimizer: "Riemannian"
#   - model: "Llama-3.2-1B"
```

---

## 6. Complete Code Patterns

### Pattern 1: Basic WandB Integration (No Explicit Init)

```python
# File: src/run_experimet.py
def run_finetune(config, model, tokenizer, train_dataset, val_dataset):
    # Config has trainer_config with report_to: wandb
    trainer = finetune.get_trainer(config, model, tokenizer, train_dataset, val_dataset)
    trainer.train()  # WandB autostart here via SFTTrainer
```

### Pattern 2: Trainer Setup with WandB Config

```python
# File: src/finetune.py
def get_trainer(config, model, tokenizer, train_dataset, val_dataset):
    # trainer_config is converted to SFTConfig
    training_args = SFTConfig(
        **OmegaConf.to_object(config.trainer_config),
    )
    # This automatically enables wandb if report_to: wandb is in config
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # ...
    )
    
    # Add custom LoRA logging callback
    if config.get('detailed_lora_logs', False):
        lora_callback = LoRAWandbLogger(model, config)
        trainer.add_callback(lora_callback)
    
    return trainer
```

### Pattern 3: Custom WandB Logging Callback

```python
# File: src/loggers.py
from transformers.integrations import WandbCallback
import wandb

class LoRAWandbLogger(WandbCallback):
    def __init__(self, model: PeftModel, config):
        super().__init__()
        self.model = model
        target_modules = config.adapter_config.target_modules
        self.module_re = re.compile('|'.join(re.escape(e) for e in target_modules))
        self.layer_re = re.compile('\.\d+')
        self.factor_re = re.compile('lora_[A,B]')

    @torch.no_grad()
    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        global_step = state.global_step
        if global_step % state.logging_steps != 0:
            return
        
        gradients = {}
        weights = {}
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad: 
                continue
            
            layer_no, module_name, factor = self._parse_layer_name(name)
            log_name = f'{factor}/layer_{layer_no}.{module_name}'
            
            grad_norm = torch.linalg.norm(param.grad).item()
            weight_norm = torch.linalg.norm(param).item()
            
            gradients[f'grad_{log_name}'] = grad_norm
            weights[f'weight_{log_name}'] = weight_norm
        
        wandb.log(gradients, global_step)
        wandb.log(weights, global_step)
```

---

## 7. Key Configuration Parameters

### trainer_config YAML Parameters Relevant to WandB:

```yaml
trainer_config:
  run_name: str                    # WandB run name (supports template vars)
  output_dir: str                  # Checkpoint directory (also used by WandB)
  report_to: "wandb" | "none"      # Enable/disable WandB reporting
  
  # Logging frequency
  logging_steps: int               # Log metrics every N steps
  eval_steps: int                  # Evaluate every N steps  
  eval_strategy: "steps" | "epoch" # When to evaluate
  
  # Training hyperparameters (logged to WandB)
  learning_rate: float
  per_device_train_batch_size: int
  per_device_eval_batch_size: int
  num_train_epochs: int
  lr_scheduler_type: str
  warmup_ratio: float
  seed: int
```

### Config Flags for Detailed Logging:

```yaml
detailed_lora_logs: true | false   # Enable/disable detailed LoRA logging
lora_qr: true | false              # Enable QR decomposition callback
```

---

## 8. Summary of Key Implementation Details

| Aspect | Implementation |
|--------|----------------|
| **Explicit `wandb.init()` call** | ❌ Not used - relies on transformers integration |
| **WandB enablement** | Via `report_to: wandb` in trainer config (YAML) |
| **Default metrics logged** | loss, eval_loss, learning_rate, epoch (automatic) |
| **Custom logging** | `LoRAWandbLogger` callback for layer-wise gradients/weights |
| **Run naming** | Templated hierarchical names: `${exp_name}/${run_name}-cfg${cfg_no}` |
| **Tags** | Not explicitly used (could be added to config if needed) |
| **Entity/Project** | Uses WandB defaults (no explicit configuration) |
| **Eval metrics** | eval_loss tracked at `eval_steps` intervals |
| **Logging frequency** | Configurable via `logging_steps` (default: 16) |

