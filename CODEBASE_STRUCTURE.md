# CoLM & GREATS Codebase Structure

## Repository Root: `llm-experiments/`

```
llm-experiments/
├── config.yaml                          # YAML config for multi-task training (model, LoRA, optimizer, GPU profiles)
├── env.sh                               # Environment setup script (conda, pip, submodlib, flash-attn)
│
├── colm/                                # Main package
│   ├── __init__.py                      # Package init
│   │
│   ├── data/                            # Data loading & preprocessing
│   │   ├── get_training_dataset.py      # Loads MathInstruct/Alpaca/SuperGLUE; returns SupervisedDataset with source labels
│   │   ├── tasks.py                     # SuperGLUE task definitions, Sample dataclass, label mappings
│   │   └── utils.py                     # Collators, forward wrappers for option-only training (SuperGLUE classification)
│   │
│   └── train/                           # Training logic
│       ├── __init__.py
│       ├── train.py                     # MAIN ENTRY POINT — parses args, loads model+LoRA, picks trainer, runs training
│       ├── train_multitask.py           # Multi-task training script for MetaMathQA+GSM8K; uses YAML config; supports Muon/AdamW
│       │
│       ├── training_arguments.py        # Custom TrainingArguments extending HF — adds subset selection, MeZO, LoRA, wandb args
│       ├── model_arguments.py           # ModelArguments dataclass — model path, LoRA config, dtype, dropout
│       ├── data_arguments.py            # DataArguments dataclass — train files, seq length, sampling percentage
│       ├── config_loader.py             # Loads config.yaml, applies optimizer/GPU profiles, converts to HF TrainingArguments
│       │
│       ├── subset_trainer_distributed.py # CORE — SubsetTrainer + SubsetTrainerEfficient; implements the full online batch selection loop
│       ├── huggingface_trainer.py       # CustomTrainer — vanilla HF Trainer with minor overrides (NaN grad handling, signature columns)
│       │
│       ├── facility_location.py         # Facility location subset selection (CoLM's medoid finding via submodlib)
│       ├── greats.py                    # GREATS greedy selection — TracIN scores + Hessian interaction matrix update
│       ├── SPOTgreedy.py                # SPOT greedy prototype selection using OT cost matrix
│       ├── fairot.py                    # FairOT v1 — entropic partial OT with KKT-based optimal alpha (older version)
│       ├── fairot2.py                   # FairOT v2 — refactored with exact/approx gain, vectorized, uses POT library
│       ├── sinkhorn.py                  # Sinkhorn OT solvers — pot_partial_extended (dummy row trick) + pot_partial_library (POT)
│       │
│       ├── custom_phi.py                # DecomposedPhiCausalLM — splits Phi-2 into penultimate + final layer for efficient MeZO
│       ├── utils.py                     # Shared utilities — cost matrix, similarity, collate_fn, stable_entropy, trak projectors
│       ├── optimizer_factory.py         # Creates AdamW or custom Muon optimizer from config; separates 2D/non-2D params for Muon
│       ├── buffsub.py                   # Standalone compute_loss snippet (reference/buffer code)
│       ├── buffhf.py                    # Empty buffer file
│       └── plot.py                      # Plots EMA-smoothed loss curves from multiple log files using matplotlib
│
├── math_eval/                           # Math benchmark evaluation
│   ├── __init__.py
│   ├── run_open.py                      # Main eval script — loads model (full or LoRA), runs PoT/CoT on math datasets
│   ├── data_loader.py                   # Reads gsm8k/math/svamp/numglue/simuleq/deepmind JSON files; BatchDatasetLoader
│   ├── prompt_utils.py                  # Few-shot prompt templates (alpaca, vicuna, tulu, etc.) + example sets per dataset
│   ├── utils.py                         # Answer cleaning, code execution, number comparison, string normalization
│   ├── eval_finetuned.sh                # Shell script — evaluates multiple checkpoints in parallel across GPUs
│   └── eval_pretrained.sh               # Shell script — evaluates a pretrained (non-finetuned) model
│
└── scripts/ (inside colm/scripts/)
    └── train/
        ├── base_training_args.sh        # Shared bash variables for training (LoRA rank, batch size, optimizer, etc.)
        ├── lora_train_math.sh           # Launches training on MathInstruct with all configurable args passed as positional params
        └── lora_train_superglue.sh      # Loops over SST2/CB/MultiRC and launches training for each SuperGLUE task
```

---

## Script-by-Script Summary

### Entry Points

| Script | Role |
|--------|------|
| `colm/train/train.py` | Main training entry — loads everything, picks trainer, runs `trainer.train()` |
| `colm/train/train_multitask.py` | Alternative entry for multi-task YAML-configured training with per-task metrics |
| `math_eval/run_open.py` | Evaluation — generates answers with vLLM or HF, scores against ground truth |

### Training Core

| Script | Role |
|--------|------|
| `subset_trainer_distributed.py` | Replaces HF's `_inner_training_loop` — accumulates large batch, gathers across GPUs, runs subset selection, trains on selected subset |
| `huggingface_trainer.py` | Standard HF training with no subset selection; adds NaN gradient handling |
| `custom_phi.py` | Splits Phi-2 forward pass at second-to-last layer so efficient batched MeZO can share activations |

### Subset Selection Algorithms

| Script | Algorithm | Paper |
|--------|-----------|-------|
| `facility_location.py` | Facility Location (submodular) medoid finding | CoLM |
| `greats.py` | Greedy TracIN + Hessian correction | GREATS (NeurIPS 2024) |
| `SPOTgreedy.py` | OT-based prototype selection | SPOT |
| `fairot.py` / `fairot2.py` | Entropic partial OT prototype selection | FairOT |
| `sinkhorn.py` | Sinkhorn solvers used by FairOT | — |

### Configuration & Arguments

| Script | Role |
|--------|------|
| `training_arguments.py` | All custom args: `small_batch_ratio`, `data_selection_method`, `zo_dim`, `keep_sources`, `mezo_optim`, etc. |
| `model_arguments.py` | `model_name_or_path`, `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules` |
| `data_arguments.py` | `train_files`, `max_seq_length`, `percentage`, `subset_index_files` |
| `config_loader.py` | Reads `config.yaml`, applies active optimizer/GPU profiles, converts to HF `TrainingArguments` |
| `optimizer_factory.py` | Factory for AdamW or Muon; Muon applies Newton-Schulz orthogonalization to 2D weight matrices |

### Data

| Script | Role |
|--------|------|
| `data/get_training_dataset.py` | Loads JSONL datasets; creates `SupervisedDataset` with `source` labels per example; handles MathInstruct source mapping |
| `data/tasks.py` | SuperGLUE task wrappers, label spaces, prompt formatting |
| `data/utils.py` | `DataCollatorWithPaddingAndNesting`, `forward_wrap_with_option_len` for classification-style training |

### Evaluation

| Script | Role |
|--------|------|
| `math_eval/run_open.py` | Batched generation with PoT (program-of-thought) primary + CoT fallback; handles LoRA adapter loading |
| `math_eval/data_loader.py` | Reads benchmark JSON files for gsm8k, math, svamp, numglue, simuleq, deepmind |
| `math_eval/prompt_utils.py` | All few-shot prompt templates and per-dataset example sets |
| `math_eval/utils.py` | `answer_clean`, `execute_with_timeout` (code execution for PoT), `compare_both_string_and_number_format` |

### Utilities

| Script | Role |
|--------|------|
| `utils.py` | `compute_cost_matrix` (cosine/L1/L2/dot), `collate_fn`, `stable_entropy`, `convert_to_ordered_range`, trak projector loading |
| `plot.py` | Reads training log files, extracts loss values, plots EMA-smoothed curves |

---

## Key Configuration Flags and Their Effect

```
data_selection_method = "none"          → CustomTrainer (vanilla fine-tuning)
data_selection_method = "submodlib"     → SubsetTrainer + facility_location
data_selection_method = "greats"        → SubsetTrainer + greats.greedy_selection
data_selection_method = "spot"          → SubsetTrainer + SPOTgreedy
data_selection_method = "fairot"        → SubsetTrainer + fairot2.greedy_fairot
data_selection_method = "fairot_multisource" → SubsetTrainer + facloc with fairot optimizer

data_selection_unit = "mezo"            → zeroth-order gradient as representation
data_selection_unit = "rep"             → last hidden state as representation
data_selection_unit = "masked_grad"     → backprop gradient of last layer

efficient_mezo = True                   → SubsetTrainerEfficient (batched MeZO via DecomposedPhiCausalLM)
efficient_mezo = False                  → SubsetTrainer (sequential per-sample MeZO)

mezo_optim = "adam"                     → normalize gradients by exponential moving averages
mezo_optim = "sgd"                      → use raw gradient vectors

lora = True                             → wrap model with PEFT LoRA before training
lora = False                            → full parameter fine-tuning
```
