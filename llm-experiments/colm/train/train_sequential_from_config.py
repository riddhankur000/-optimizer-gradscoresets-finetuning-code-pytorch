#!/usr/bin/env python
# coding=utf-8
"""
Sequential Multi-Task Training with Config-based Setup

Uses config.yaml for all configuration:
- Supports both AdamW and Muon optimizers (selectable in config)
- Supports single GPU or distributed multi-GPU training
- Automatic WandB tracking with comprehensive metrics
- Eval loss tracking and overfitting detection
"""

import logging
import os
import sys
import json
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Suppress bitsandbytes warnings (not needed for regular LoRA)
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")

# Monkeypatch to fix bitsandbytes compatibility issues with newer PyTorch versions
import sys
try:
    import torch._C
    if not hasattr(torch._C, '_has_xpu'):
        torch._C._has_xpu = False
except Exception:
    pass

try:
    import torch.compiler
    if not hasattr(torch.compiler, 'is_compiling'):
        torch.compiler.is_compiling = lambda: False
except Exception:
    pass

import datasets
import torch
import torch.nn as nn
import torch.distributed as dist
import transformers
import psutil
import GPUtil
import wandb
from transformers import (
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    default_data_collator,
    TrainerCallback,
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import random_split

# Import config parser and task loader
from colm.train.config_parser import ConfigLoader, OptimizerConfig, GPUConfig
from colm.data.sequential_task_loader import SequentialTaskLoader, convert_task_samples_to_hf_dataset

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MonitoringCallbackSeq(TrainerCallback):
    """Enhanced monitoring callback with GPU/system stats and overfitting detection"""
    
    def __init__(self, task_id: int = 0, config_loader: Optional[ConfigLoader] = None):
        self.trainer = None
        self.last_grad_norm = None
        self.task_id = task_id
        self.config_loader = config_loader
    
    def on_backward_end(self, args, state, control, **kwargs):
        """Compute gradient norms after backward pass."""
        if not self.trainer:
            return
        
        try:
            model = self.trainer.model
            if model:
                self.last_grad_norm = self._get_grad_norm(model)
        except Exception as e:
            logger.debug(f"Error computing grad norm: {e}")
            self.last_grad_norm = None
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Add enhanced metrics to logs."""
        if logs is None:
            return
        
        logs['task_id'] = self.task_id
        
        if 'loss' in logs and logs['loss'] is not None:
            try:
                logs['train_perplexity'] = math.exp(min(logs['loss'], 100))
            except:
                pass
        
        if self.last_grad_norm:
            logs.update(self.last_grad_norm)
        
        try:
            gpu_stats = self._get_gpu_stats()
            logs.update(gpu_stats)
        except Exception as e:
            logger.debug(f"Error getting GPU stats: {e}")
        
        try:
            sys_stats = self._get_system_stats()
            logs.update(sys_stats)
        except Exception as e:
            logger.debug(f"Error getting system stats: {e}")
        
        if 'eval_loss' in logs and logs['eval_loss'] is not None:
            try:
                logs['eval_perplexity'] = math.exp(min(logs['eval_loss'], 100))
            except:
                pass
            
            if 'loss' in logs:
                logs['overfit_ratio'] = logs['eval_loss'] / (logs['loss'] + 1e-6)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Handle evaluation metrics."""
        if metrics is None or not self.trainer:
            return
        
        metrics['task_id'] = self.task_id
        
        if 'eval_loss' in metrics and metrics['eval_loss'] is not None:
            try:
                metrics['eval_perplexity'] = math.exp(min(metrics['eval_loss'], 100))
            except:
                pass
        
        try:
            model = self.trainer.model
            if model:
                grad_stats = self._get_grad_norm(model)
                metrics.update({f'eval_{k}': v for k, v in grad_stats.items()})
        except Exception as e:
            logger.debug(f"Error computing eval grad norm: {e}")
        
        try:
            gpu_stats = self._get_gpu_stats()
            metrics.update({f'eval_{k}': v for k, v in gpu_stats.items()})
        except Exception as e:
            logger.debug(f"Error getting eval GPU stats: {e}")
        
        try:
            sys_stats = self._get_system_stats()
            metrics.update({f'eval_{k}': v for k, v in sys_stats.items()})
        except Exception as e:
            logger.debug(f"Error getting eval system stats: {e}")
        
        try:
            if self.trainer and hasattr(self.trainer, 'log_metrics'):
                self.trainer.log_metrics("eval", metrics, state.global_step)
            
            eval_loss = metrics.get('eval_loss', 'N/A')
            eval_perp = metrics.get('eval_perplexity', 'N/A')
            logger.info(f"✓ Task {self.task_id} Eval at step {state.global_step}: loss={eval_loss}, perplexity={eval_perp}")
        except Exception as e:
            logger.debug(f"Could not log eval metrics: {e}")
    
    def _get_grad_norm(self, model: nn.Module) -> Dict[str, float]:
        """Calculate gradient norm statistics."""
        total_norm = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2.)
        
        return {
            'grad_norm': total_norm,
            'grad_norm_avg': total_norm / max(param_count, 1),
        }
    
    def _get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU memory and utilization statistics."""
        try:
            gpus = GPUtil.getGPUs()
            stats = {}
            
            if gpus:
                total_mem_used = sum(g.memoryUsed for g in gpus)
                total_mem_total = sum(g.memoryTotal for g in gpus)
                avg_utilization = sum(g.memoryUtil for g in gpus) / len(gpus)
                avg_load = sum(g.load for g in gpus) / len(gpus)
                
                stats['gpu_memory_used_gb'] = total_mem_used / 1024
                stats['gpu_memory_total_gb'] = total_mem_total / 1024
                stats['gpu_memory_utilization_%'] = avg_utilization * 100
                stats['gpu_load_%'] = avg_load * 100
                
                for i, gpu in enumerate(gpus):
                    stats[f'gpu_{i}_mem_used_gb'] = gpu.memoryUsed / 1024
                    stats[f'gpu_{i}_mem_util_%'] = gpu.memoryUtil * 100
            
            return stats
        except Exception as e:
            logger.debug(f"Could not get GPU stats: {e}")
            return {}
    
    def _get_system_stats(self) -> Dict[str, float]:
        """Get CPU and system memory statistics."""
        try:
            stats = {
                'cpu_percent': psutil.cpu_percent(interval=0.01),
                'cpu_memory_percent': psutil.virtual_memory().percent,
                'cpu_memory_available_gb': psutil.virtual_memory().available / (1024**3),
            }
            return stats
        except Exception as e:
            logger.debug(f"Could not get system stats: {e}")
            return {}


def create_validation_split(dataset, val_split_ratio: float = 0.1):
    """Create validation split from training dataset."""
    dataset_size = len(dataset)
    val_size = max(1, int(dataset_size * val_split_ratio))
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Dataset split: {train_size} training, {val_size} validation")
    return dataset, val_dataset


def setup_optimizer(model: nn.Module, optimizer_config: OptimizerConfig, training_args: TrainingArguments) -> Optional[torch.optim.Optimizer]:
    """
    Setup optimizer based on configuration.
    
    For AdamW: Use HF Trainer's built-in optimizer
    For Muon: Create custom optimizer (if muon library available)
    """
    if optimizer_config.optimizer_type == 'muon':
        try:
            # Try to import muon optimizer
            from muon import Muon
            
            logger.info("🔧 Setting up Muon optimizer...")
            
            muon_args = {
                'lr': optimizer_config.muon_lr or training_args.learning_rate,
                'weight_decay': optimizer_config.muon_weight_decay or training_args.weight_decay,
                'momentum': optimizer_config.muon_momentum or 0.95,
                'nesterov': optimizer_config.muon_nesterov if optimizer_config.muon_nesterov is not None else True,
            }
            
            if optimizer_config.muon_ns_coefficients:
                muon_args['ns_coefficients'] = optimizer_config.muon_ns_coefficients
            
            if optimizer_config.muon_eps:
                muon_args['eps'] = optimizer_config.muon_eps
            
            if optimizer_config.muon_ns_steps:
                muon_args['ns_steps'] = optimizer_config.muon_ns_steps
            
            if optimizer_config.muon_adjust_lr_fn:
                muon_args['adjust_lr_fn'] = optimizer_config.muon_adjust_lr_fn
            
            logger.info(f"Muon optimizer config: {muon_args}")
            
            return Muon(model.parameters(), **muon_args)
        
        except ImportError:
            logger.warning("Muon optimizer not available, falling back to AdamW")
            return None
    
    else:  # AdamW
        logger.info("🔧 Setting up AdamW optimizer (via HF Trainer)...")
        # AdamW is handled by the trainer
        return None


def initialize_wandb(config_loader: ConfigLoader, model_name: str) -> str:
    """Initialize WandB with configuration from config.yaml"""
    
    wandb_config = config_loader.get_wandb_config()
    optimizer_name = config_loader.optimizer_name
    gpu_profile = config_loader.gpu_profile_name
    training_config = config_loader.get_training_config()
    optimizer_config = config_loader.get_optimizer_config()
    
    # Build run name
    model_short = model_name.split('/')[-1].lower()
    lora_config = config_loader.get_lora_config()
    use_lora = lora_config.get('enabled', True)
    lora_rank = lora_config.get('lora_rank', 0) if use_lora else 0
    
    lr = training_config.get('learning_rate', 2e-4)
    finetuning_type = f"lora_r{lora_rank}" if use_lora else "full"
    lr_str = f"{lr:.0e}".replace('e-0', 'e-')
    
    run_name = f"{model_short}_{optimizer_name}_{finetuning_type}_{lr_str}"
    
    if wandb_config.get('enabled', True):
        if wandb.run is None:
            wandb.init(
                project=wandb_config.get('project', 'colm-sequential-training'),
                name=run_name,
                config={
                    "model": model_name,
                    "optimizer": optimizer_name,
                    "gpu_profile": gpu_profile,
                    "use_lora": use_lora,
                    "lora_rank": lora_rank,
                    "learning_rate": lr,
                    **optimizer_config.__dict__,
                },
                tags=wandb_config.get('tags', ['sequential-training', 'multi-task']),
                notes=wandb_config.get('notes', 'Sequential training with config.yaml'),
            )
        logger.info(f"✅ WandB initialized: {run_name}")
    
    return run_name


def run_sequential_training_from_config(config_path: str):
    """
    Main function: Load config and run sequential training.
    
    Args:
        config_path: Path to config.yaml file
    """
    # Load configuration
    logger.info(f"📖 Loading configuration from: {config_path}")
    config_loader = ConfigLoader(config_path)
    config_loader.print_config_summary()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    
    # Setup distributed training if needed
    gpu_config = config_loader.get_gpu_config()
    config_loader.setup_distributed_training(gpu_config)
    
    logger.info(f"⚙️  GPU Configuration: {gpu_config.num_gpus} GPU(s), Distributed: {gpu_config.use_distributed}")
    
    # Get configurations
    model_config = config_loader.get_model_config()
    tokenizer_config = config_loader.get_tokenizer_config()
    lora_config_dict = config_loader.get_lora_config()
    dataset_config = config_loader.get_dataset_config()
    training_config_dict = config_loader.build_training_arguments()
    optimizer_config = config_loader.get_optimizer_config()
    multitask_config = config_loader.get_multitask_config()
    
    # Set seed
    set_seed(training_config_dict.get('seed', 0))
    
    # Load tokenizer
    logger.info("=" * 80)
    logger.info("STEP 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config.get('tokenizer_path'),
        model_max_length=tokenizer_config.get('model_max_length', 512),
        padding_side=tokenizer_config.get('padding_side', 'right'),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("✓ Tokenizer loaded")
    
    # Load model
    logger.info("=" * 80)
    logger.info("STEP 2: Loading base model ONCE...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.get('model_id'),
        torch_dtype=torch.bfloat16 if training_config_dict.get('bf16') else torch.float32,
        device_map=gpu_config.device_map,
        trust_remote_code=True,
    )
    logger.info(f"✓ Model loaded: {type(model).__name__}")
    logger.info(f"  Size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    # Explicitly disable gradient checkpointing (incompatible with LoRA gradients)
    model.config.gradient_checkpointing = False
    model.gradient_checkpointing = False
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    logger.info("✓ Gradient checkpointing disabled, input gradients enabled")
    
    # Add LoRA
    if lora_config_dict.get('enabled', True):
        logger.info("=" * 80)
        logger.info("STEP 3: Applying LoRA...")
        try:
            lora_config = LoraConfig(
                r=lora_config_dict.get('lora_rank', 128),
                lora_alpha=lora_config_dict.get('lora_alpha', 512),
                lora_dropout=lora_config_dict.get('lora_dropout', 0.05),
                bias=lora_config_dict.get('bias', 'none'),
                task_type=TaskType.CAUSAL_LM,
                target_modules=lora_config_dict.get('target_modules', ['q_proj', 'v_proj']),
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            logger.info("✓ LoRA applied successfully")
        except Exception as e:
            if 'torch._C' in str(e) or 'torch.compiler' in str(e) or 'bitsandbytes' in str(e):
                logger.warning(f"⚠ LoRA initialization warning (bitsandbytes compatibility): {e}")
                logger.info("  Attempting to apply LoRA with workaround...")
                try:
                    # Try applying LoRA with a clean import
                    model = get_peft_model(model, lora_config)
                    model.print_trainable_parameters()
                    logger.info("✓ LoRA applied successfully (on retry)")
                except Exception as retry_error:
                    logger.error(f"Failed to apply LoRA even with workaround: {retry_error}")
                    raise
            else:
                logger.error(f"Error applying LoRA: {e}")
                raise
    else:
        logger.info("STEP 3: Full fine-tuning (no LoRA)")
    
    # Initialize WandB
    logger.info("=" * 80)
    logger.info("STEP 4: Initializing WandB...")
    run_name = initialize_wandb(config_loader, model_config.get('model_id'))
    
    # Load dataset
    logger.info("=" * 80)
    logger.info("STEP 5: Loading dataset configuration...")
    
    dataset_config = config_loader.get_dataset_config()
    sequential_tasks_config = config_loader.get_sequential_tasks_config()
    
    # Check if sequential task loading is enabled
    use_sequential_tasks = sequential_tasks_config.get('enabled', False)
    
    if use_sequential_tasks:
        logger.info("🔄 Using SEQUENTIAL TASK LOADING (Riemannian Method)")
        task_loader = SequentialTaskLoader(sequential_tasks_config)
        task_loader.print_tasks_summary()
        num_tasks = task_loader.num_tasks
        tasks_dataset_per_id = {}  # Will load on-demand
        dataset = None  # Not used in sequential mode
    else:
        logger.info("📦 Using SINGLE DATASET LOADING")
        dataset_path = dataset_config.get('dataset_path')
        if os.path.isdir(dataset_path):
            dataset = load_from_disk(dataset_path)
        else:
            raise ValueError(f"Dataset path not found: {dataset_path}")
        logger.info(f"✓ Dataset loaded: {len(dataset)} examples")
        
        task_loader = None
        num_tasks = multitask_config.get('num_tasks', 1)
        tasks_dataset_per_id = {}
    
    val_split_ratio = dataset_config.get('val_split_ratio', 0.1)
    
    # Create training arguments
    training_args = TrainingArguments(**training_config_dict)
    
    # Note: Using standard transformers.Trainer instead of SubsetTrainerEfficient
    # We adopt Riemannian's dataset preprocessing but use standard AdamW training
    # This supports all LoRA parameters properly (lora_A and lora_B)
    
    # Sequential task training
    logger.info("=" * 80)
    logger.info(f"STARTING SEQUENTIAL TRAINING: {num_tasks} tasks")
    logger.info(f"Optimizer: {optimizer_config.optimizer_type}")
    logger.info(f"GPU Profile: {config_loader.gpu_profile_name}")
    logger.info("=" * 80)
    
    all_metrics = {}
    
    for task_id in range(num_tasks):
        task_name = f"Task_{task_id}"
        
        logger.info("\n" + "=" * 80)
        logger.info(f"TASK {task_id}: {task_name}")
        logger.info(f"Model state: Already trained on {task_id} previous task(s)")
        logger.info(f"Optimizer: {optimizer_config.optimizer_type}")
        logger.info("=" * 80)
        
        # Load task-specific dataset
        if use_sequential_tasks:
            logger.info(f"Loading task-specific dataset: {task_loader.tasks[task_id]}")
            train_subset_torch, val_subset_torch = task_loader.load_task(task_id)
            
            # Convert to HuggingFace dataset format
            train_subset = convert_task_samples_to_hf_dataset(train_subset_torch)
            val_subset = convert_task_samples_to_hf_dataset(val_subset_torch)
            
            task_name = task_loader.tasks[task_id]
            logger.info(f"✓ Loaded {len(train_subset)} training and {len(val_subset)} validation samples")
        else:
            # Use single dataset with random splits per task
            train_subset, val_subset = create_validation_split(dataset, val_split_ratio)
        
        # Setup trainer
        # Using standard transformers.Trainer for LoRA training (supports all lora_A and lora_B parameters)
        # We adopt Riemannian's dataset preprocessing but use standard training
        
        # Tokenize datasets for training (causal LM format)
        def tokenize_function(examples):
            """Tokenize text samples for causal LM training"""
            result = tokenizer(
                examples['text'],
                truncation=True,
                max_length=tokenizer_config.get('model_max_length', 512),
                padding='max_length',  # Pad all samples to same length
            )
            # For causal LM: labels = input_ids for next-token prediction
            # Masking of padding tokens will be handled by the attention_mask
            result['labels'] = result['input_ids'].copy()
            
            return result
        
        # Apply tokenization
        train_subset = train_subset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text', 'id', 'label'],  # Remove text, id, and original label field
            desc="Tokenizing training dataset"
        )
        val_subset = val_subset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text', 'id', 'label'],  # Remove text, id, and original label field
            desc="Tokenizing validation dataset"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            data_collator=default_data_collator,  # Simple collator for causal LM (no special label handling)
            callbacks=[MonitoringCallbackSeq(task_id=task_id, config_loader=config_loader)],
        )
        
        # Train
        logger.info(f"Training on {task_name}...")
        train_result = trainer.train()
        logger.info(f"✓ Task {task_id} training completed")
        logger.info(f"  Train loss: {train_result.training_loss:.4f}")
        
        # Evaluate
        logger.info(f"Evaluating on {task_name}...")
        eval_results = trainer.evaluate()
        logger.info(f"✓ Task {task_id} evaluation completed")
        logger.info(f"  Eval loss: {eval_results.get('eval_loss', 'N/A')}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(training_args.output_dir, f"task_{task_id}_checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        logger.info(f"✓ Checkpoint saved: {checkpoint_dir}")
        
        # Store metrics
        task_metrics_key = f"task_{task_id}_{task_name}"
        all_metrics[task_metrics_key] = {
            'train_loss': train_result.training_loss,
            'eval_loss': eval_results.get('eval_loss'),
            'optimizer': optimizer_config.optimizer_type,
            **eval_results
        }
        
        # Log to WandB
        task_log = {
            f"task_{task_id}/train_loss": train_result.training_loss,
            f"task_{task_id}/task_name": task_name,
            f"task_{task_id}/optimizer": optimizer_config.optimizer_type,
        }
        if 'eval_loss' in eval_results:
            task_log[f"task_{task_id}/eval_loss"] = eval_results['eval_loss']
        wandb.log(task_log)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("SEQUENTIAL TRAINING COMPLETED")
    logger.info("=" * 80)
    
    # Log final metrics
    summary_metrics = {
        "total_tasks_completed": num_tasks,
        "training_run_name": run_name,
        "optimizer_used": optimizer_config.optimizer_type,
        "gpu_profile_used": gpu_config.num_gpus,
    }
    
    for task_id, task_metrics in all_metrics.items():
        for metric_name, metric_value in task_metrics.items():
            if isinstance(metric_value, (int, float)):
                summary_metrics[f"final_{task_id}_{metric_name}"] = metric_value
    
    wandb.log(summary_metrics)
    
    logger.info(f"\nTraining Summary:")
    for task_name, metrics in all_metrics.items():
        logger.info(f"  {task_name}:")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                logger.info(f"    {metric_name}: {metric_value:.4f}")
    
    wandb.finish()
    
    logger.info("\n✅ Training completed successfully!")
    logger.info(f"Results saved to: {training_args.output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_sequential_from_config.py <config.yaml>")
        print("\nExample:")
        print("  python train_sequential_from_config.py ./config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    run_sequential_training_from_config(config_path)
