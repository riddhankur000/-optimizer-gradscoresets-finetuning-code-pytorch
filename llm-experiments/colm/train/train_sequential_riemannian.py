#!/usr/bin/env python
# coding=utf-8
"""
Sequential Multi-Task Training Script for CoLM
Implements Riemannian-style sequential training where:
- Model is loaded ONCE before task loop
- Each task starts from previous task's checkpoint weights
- Cumulative learning across tasks
- Single wandb run for all tasks
- Eval loss tracked during training for overfitting detection
"""

import logging
import os
import sys
import json
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

import datasets
import torch
import torch.nn as nn
import torch.distributed as dist
import transformers
import yaml
import psutil
import GPUtil
import wandb
from transformers import (
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments as HFTrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainerFeedback,
)
from transformers.utils import is_torch_available
from datasets import load_from_disk, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import random_split, Subset

# Import CoLM modules
from colm.train.config_loader import (
    load_config_yaml,
    config_dict_to_hf_training_args,
    get_model_config,
    get_lora_config,
    get_dataset_config,
    get_optimizer_config,
    get_gpu_config,
    print_config_summary,
)
from colm.train.optimizer_factory import create_optimizer_from_config
from colm.train.huggingface_trainer import CustomTrainer as Trainer
from colm.train.subset_trainer_distributed import SubsetTrainer, SubsetTrainerEfficient

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TrainingPhase(Enum):
    """Enum for training phases"""
    FINETUNE = "finetune"
    VALIDATE = "validate"


class MonitoringCallbackSeq(TrainerCallback):
    """
    Enhanced monitoring callback for sequential training:
    - Gradient norms tracking
    - GPU memory and utilization
    - System CPU/memory stats
    - Perplexity calculation
    - Per-task evaluation metrics
    - Overfitting detection (train vs eval loss)
    """
    
    def __init__(self, task_id: int = 0):
        self.trainer = None
        self.last_grad_norm = None
        self.task_id = task_id
    
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
        
        # Add task_id to all logs for tracking
        logs['task_id'] = self.task_id
        
        # Calculate perplexity for train loss
        if 'loss' in logs and logs['loss'] is not None:
            try:
                logs['train_perplexity'] = math.exp(min(logs['loss'], 100))
            except:
                pass
        
        # Add gradient norms
        if self.last_grad_norm:
            logs.update(self.last_grad_norm)
        
        # Add GPU stats
        try:
            gpu_stats = self._get_gpu_stats()
            logs.update(gpu_stats)
        except Exception as e:
            logger.debug(f"Error getting GPU stats: {e}")
        
        # Add system stats
        try:
            sys_stats = self._get_system_stats()
            logs.update(sys_stats)
        except Exception as e:
            logger.debug(f"Error getting system stats: {e}")
        
        # Eval loss perplexity
        if 'eval_loss' in logs and logs['eval_loss'] is not None:
            try:
                logs['eval_perplexity'] = math.exp(min(logs['eval_loss'], 100))
            except:
                pass
            
            # Overfitting detection
            if 'loss' in logs:
                logs['overfit_ratio'] = logs['eval_loss'] / (logs['loss'] + 1e-6)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Handle evaluation metrics."""
        if metrics is None or not self.trainer:
            return
        
        # Add task_id
        metrics['task_id'] = self.task_id
        
        # Calculate perplexity
        if 'eval_loss' in metrics and metrics['eval_loss'] is not None:
            try:
                metrics['eval_perplexity'] = math.exp(min(metrics['eval_loss'], 100))
            except:
                pass
        
        # Add gradient norms at eval time
        try:
            model = self.trainer.model
            if model:
                grad_stats = self._get_grad_norm(model)
                metrics.update({f'eval_{k}': v for k, v in grad_stats.items()})
        except Exception as e:
            logger.debug(f"Error computing eval grad norm: {e}")
        
        # GPU stats at eval time
        try:
            gpu_stats = self._get_gpu_stats()
            metrics.update({f'eval_{k}': v for k, v in gpu_stats.items()})
        except Exception as e:
            logger.debug(f"Error getting eval GPU stats: {e}")
        
        # System stats at eval time
        try:
            sys_stats = self._get_system_stats()
            metrics.update({f'eval_{k}': v for k, v in sys_stats.items()})
        except Exception as e:
            logger.debug(f"Error getting eval system stats: {e}")
        
        # Log to wandb if available
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
    """
    Create validation split from training dataset.
    
    Args:
        dataset: Training dataset
        val_split_ratio: Ratio of data to use for validation (default 10%)
    
    Returns:
        train_dataset, val_dataset
    """
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


def build_wandb_run_name(model_name: str, optimizer_name: str, use_lora: bool, 
                         learning_rate: float, lora_rank: Optional[int] = None) -> str:
    """
    Build wandb run name with all relevant parameters.
    
    Format: {model}_{optimizer}_{"lora" or "full"}_{lr}
    """
    model_short = model_name.split('/')[-1].lower()
    opt_short = optimizer_name.lower()
    finetuning_type = f"lora_r{lora_rank}" if use_lora and lora_rank else "full"
    lr_str = f"{learning_rate:.0e}".replace('e-0', 'e-')
    
    run_name = f"{model_short}_{opt_short}_{finetuning_type}_{lr_str}"
    return run_name


def initialize_wandb(
    model_name: str,
    optimizer_name: str,
    use_lora: bool,
    learning_rate: float,
    lora_rank: Optional[int] = None,
    project_name: str = "colm-sequential-training",
):
    """
    Initialize wandb with comprehensive tracking configuration.
    
    This uses a SINGLE run for all tasks (accumulated training).
    """
    run_name = build_wandb_run_name(
        model_name, optimizer_name, use_lora, learning_rate, lora_rank
    )
    
    wandb_config = {
        "model": model_name,
        "optimizer": optimizer_name,
        "use_lora": use_lora,
        "lora_rank": lora_rank if use_lora else None,
        "learning_rate": learning_rate,
    }
    
    # Initialize SINGLE run for all tasks
    if wandb.run is None:
        wandb.init(
            project=project_name,
            name=run_name,
            config=wandb_config,
            tags=["sequential-training", "multi-task", "cumulative"],
        )
    
    logger.info(f"WandB initialized with run name: {run_name}")
    return run_name


def run_sequential_training(
    model_args: 'ModelArguments',
    data_args: 'DataArguments',
    training_args: 'TrainingArguments',
    task_names: Optional[List[str]] = None,
    num_tasks: int = 1,
    val_split_ratio: float = 0.1,
):
    """
    Sequential multi-task training following Riemannian approach:
    1. Load model ONCE before task loop
    2. Loop through tasks, training on each
    3. Each task starts from previous task's weights
    4. Single wandb run for all tasks
    5. Eval loss tracked for overfitting detection
    
    Args:
        model_args: Model configuration
        data_args: Data configuration
        training_args: Training configuration
        task_names: Optional list of task names for logging
        num_tasks: Number of sequential tasks to train on
        val_split_ratio: Ratio of training data to use for validation
    """
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    
    # Log process info
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        f"distributed training: {bool(training_args.local_rank != -1)}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")
    
    # Set seed
    set_seed(training_args.seed)
    
    # STEP 1: Load tokenizer
    logger.info("=" * 80)
    logger.info("STEP 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=getattr(model_args, 'model_max_length', 512),
        padding_side="right",
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # STEP 2: Load model ONCE (outside task loop)
    logger.info("=" * 80)
    logger.info("STEP 2: Loading base model ONCE...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    
    logger.info(f"Model loaded: {type(model).__name__}")
    logger.info(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    # STEP 3: Add LoRA if specified
    if getattr(model_args, 'use_lora', True):
        logger.info("=" * 80)
        logger.info("STEP 3: Applying LoRA...")
        lora_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        logger.info("STEP 3: Full fine-tuning (no LoRA)")
    
    # STEP 4: Initialize wandb for SINGLE run across all tasks
    logger.info("=" * 80)
    logger.info("STEP 4: Initializing WandB...")
    
    optimizer_name = getattr(training_args, 'optimizer', 'adamw')
    use_lora = getattr(model_args, 'use_lora', True)
    
    run_name = initialize_wandb(
        model_name=model_args.model_name_or_path,
        optimizer_name=optimizer_name,
        use_lora=use_lora,
        learning_rate=training_args.learning_rate,
        lora_rank=getattr(model_args, 'lora_rank', None),
    )
    
    # STEP 5: Load dataset
    logger.info("=" * 80)
    logger.info("STEP 5: Loading dataset...")
    
    if os.path.isdir(data_args.dataset_path):
        dataset = load_from_disk(data_args.dataset_path)
    else:
        raise ValueError(f"Dataset path not found: {data_args.dataset_path}")
    
    logger.info(f"Dataset loaded: {len(dataset)} examples")
    
    # STEP 6: Sequential task training loop
    logger.info("=" * 80)
    logger.info(f"STARTING SEQUENTIAL TRAINING: {num_tasks} tasks")
    logger.info(f"Model will be LOADED ONCE and reused for all tasks (cumulative learning)")
    logger.info("=" * 80)
    
    all_metrics = {}
    
    for task_id in range(num_tasks):
        task_name = task_names[task_id] if task_names else f"Task_{task_id}"
        
        logger.info("\n" + "=" * 80)
        logger.info(f"TASK {task_id}: {task_name}")
        logger.info(f"Model state: Already trained on {task_id} previous task(s)")
        logger.info("=" * 80)
        
        # Create validation split for THIS task
        train_subset, val_subset = create_validation_split(dataset, val_split_ratio)
        
        # Setup training arguments for this task
        task_output_dir = os.path.join(
            training_args.output_dir,
            f"task_{task_id}_{task_name}"
        )
        os.makedirs(task_output_dir, exist_ok=True)
        
        # Create trainer with enhanced monitoring for THIS task
        trainer = SubsetTrainerEfficient(
            model=model,  # SAME model object from previous task
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer,
                pad_to_multiple_of=8 if training_args.fp16 else None,
                label_pad_token_id=-100,
            ),
            callbacks=[MonitoringCallbackSeq(task_id=task_id)],
        )
        
        # Train on this task
        logger.info(f"Training on {task_name}...")
        train_result = trainer.train()
        
        # Log task completion
        logger.info(f"✓ Task {task_id} training completed")
        logger.info(f"  Train loss: {train_result.training_loss:.4f}")
        
        # Evaluate on this task
        logger.info(f"Evaluating on {task_name}...")
        eval_results = trainer.evaluate()
        logger.info(f"✓ Task {task_id} evaluation completed")
        logger.info(f"  Eval loss: {eval_results.get('eval_loss', 'N/A')}")
        
        # Save checkpoint (optional)
        checkpoint_dir = os.path.join(task_output_dir, 'checkpoint')
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        logger.info(f"✓ Checkpoint saved: {checkpoint_dir}")
        
        # Store metrics for this task
        task_metrics_key = f"task_{task_id}_{task_name}"
        all_metrics[task_metrics_key] = {
            'train_loss': train_result.training_loss,
            'eval_loss': eval_results.get('eval_loss'),
            'eval_perplexity': eval_results.get('eval_perplexity'),
            **eval_results
        }
        
        # Log task metrics to wandb
        task_log = {f"task_{task_id}/train_loss": train_result.training_loss}
        if 'eval_loss' in eval_results:
            task_log[f"task_{task_id}/eval_loss"] = eval_results['eval_loss']
        wandb.log(task_log)
        
        logger.info(f"Task {task_id} metrics logged to WandB")
        
        # Model persists to next iteration with accumulated weights
        logger.info(f"Model weights accumulated - continuing to next task with modified weights")
    
    # STEP 7: Final summary
    logger.info("\n" + "=" * 80)
    logger.info("SEQUENTIAL TRAINING COMPLETED")
    logger.info("=" * 80)
    
    # Log final summary to wandb
    summary_metrics = {
        "total_tasks_completed": num_tasks,
        "training_run_name": run_name,
    }
    
    for task_id, task_metrics in all_metrics.items():
        for metric_name, metric_value in task_metrics.items():
            if isinstance(metric_value, (int, float)):
                summary_metrics[f"final_{task_id}_{metric_name}"] = metric_value
    
    wandb.log(summary_metrics)
    logger.info(f"Final metrics logged to WandB")
    
    logger.info(f"\nTraining Summary:")
    for task_name, metrics in all_metrics.items():
        logger.info(f"  {task_name}:")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                logger.info(f"    {metric_name}: {metric_value:.4f}")
    
    # Close wandb run
    wandb.finish()
    
    return all_metrics


@dataclass
class ModelArguments:
    """Model arguments"""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA fine-tuning"}
    )
    lora_rank: int = field(
        default=128,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=512,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )


@dataclass
class DataArguments:
    """Data arguments"""
    dataset_path: str = field(
        metadata={"help": "Path to dataset"}
    )
    val_split_ratio: float = field(
        default=0.1,
        metadata={"help": "Validation split ratio (default 10%)"}
    )
    num_tasks: int = field(
        default=1,
        metadata={"help": "Number of sequential tasks to train on"}
    )


@dataclass
class TrainingArguments(HFTrainingArguments):
    """Extended training arguments"""
    pass


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Run sequential training
    run_sequential_training(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        num_tasks=data_args.num_tasks,
        val_split_ratio=data_args.val_split_ratio,
    )
