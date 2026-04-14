#!/usr/bin/env python
# coding=utf-8
"""
Training script for multi-task dataset (MetaMathQA + GSM8K combined)
Loads from pre-created combined dataset and trains with per-task metrics
Supports both YAML config file and command-line arguments
Enhanced with comprehensive metrics, grad norm, and GPU monitoring
"""

import logging
import os
import sys
import json
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import datasets
import torch
import torch.nn as nn
import transformers
import yaml
import psutil
import GPUtil
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
)
from transformers.utils import is_torch_available
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType

# Import config loader
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

# Import optimizer factory
from colm.train.optimizer_factory import create_optimizer_from_config

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MonitoringCallback(TrainerCallback):
    """
    Callback to enhance logging with:
    - Gradient norms (L2 norm of gradients)
    - GPU memory and utilization
    - System CPU/memory stats
    - Perplexity calculation
    - Evaluation metrics with perplexity
    """
    
    def __init__(self):
        self.trainer = None  # Will be set after trainer creation
        self.last_grad_norm = None  # Store grad norm from on_step_end
    
    def on_backward_end(self, args, state, control, **kwargs):
        """
        Called after backward is complete - BEFORE gradients are zeroed.
        This is when we compute gradient norms.
        """
        if not self.trainer:
            return
        
        try:
            # Compute gradient norms before they're cleared by optimizer
            model = self.trainer.model
            if model:
                self.last_grad_norm = self._get_grad_norm(model)
        except Exception as e:
            logger.debug(f"Error computing grad norm in on_backward_end: {e}")
            self.last_grad_norm = None
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called when the trainer logs metrics.
        Add perplexity, grad norm, GPU and system stats to logs.
        """
        if logs is None:
            return
        
        # Add loss if not present (it should be from trainer)
        # Add perplexity calculation for train loss
        if 'loss' in logs and logs['loss'] is not None:
            try:
                logs['train_perplexity'] = math.exp(logs['loss'])
            except:
                pass
        
        # Add stored grad norm from on_backward_end
        if self.last_grad_norm:
            logs.update(self.last_grad_norm)
        
        # Add GPU stats to logs
        try:
            gpu_stats = self._get_gpu_stats()
            logs.update(gpu_stats)
        except Exception as e:
            logger.debug(f"Error getting GPU stats: {e}")
        
        # Add system stats to logs
        try:
            sys_stats = self._get_system_stats()
            logs.update(sys_stats)
        except Exception as e:
            logger.debug(f"Error getting system stats: {e}")
        
        # Add perplexity for eval loss
        if 'eval_loss' in logs and logs['eval_loss'] is not None:
            try:
                logs['eval_perplexity'] = math.exp(logs['eval_loss'])
            except:
                pass
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Called after evaluation is completed.
        Ensures eval metrics are properly logged to wandb.
        """
        if metrics is None or not self.trainer:
            return
        
        model = self.trainer.model
        
        # Add perplexity calculation
        if 'loss' in metrics and metrics['loss'] is not None:
            try:
                metrics['perplexity'] = math.exp(metrics['loss'])
            except:
                pass
        
        # Add gradient norms evaluated at eval time
        if model:
            try:
                grad_stats = self._get_grad_norm(model)
                metrics.update({f'eval_{k}': v for k, v in grad_stats.items()})
            except Exception as e:
                logger.debug(f"Error computing eval grad norm: {e}")
        
        # Add GPU stats at eval time
        try:
            gpu_stats = self._get_gpu_stats()
            metrics.update({f'eval_{k}': v for k, v in gpu_stats.items()})
        except Exception as e:
            logger.debug(f"Error getting eval GPU stats: {e}")
        
        # Add system stats at eval time
        try:
            sys_stats = self._get_system_stats()
            metrics.update({f'eval_{k}': v for k, v in sys_stats.items()})
        except Exception as e:
            logger.debug(f"Error getting eval system stats: {e}")
        
        # Log eval metrics to wandb via trainer
        try:
            self.trainer.log_metrics("eval", metrics, state.global_step)
            eval_loss = metrics.get('loss', 'N/A')
            eval_perplexity = metrics.get('perplexity', 'N/A')
            logger.info(f"✓ Evaluation at step {state.global_step}: loss={eval_loss}, perplexity={eval_perplexity}")
        except Exception as e:
            logger.debug(f"Could not log eval metrics to wandb: {e}")

            eval_loss = metrics.get('loss', 'N/A')
            eval_perplexity = metrics.get('perplexity', 'N/A')
            logger.info(f"✓ Evaluation at step {state.global_step}: loss={eval_loss}, perplexity={eval_perplexity}")
        except Exception as e:
            logger.debug(f"Could not log eval metrics to wandb: {e}")
    
    def _get_grad_norm(self, model: nn.Module) -> Dict[str, float]:
        """Calculate gradient norm statistics"""
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
        """Get GPU memory and utilization statistics"""
        try:
            gpus = GPUtil.getGPUs()
            stats = {}
            
            # Average stats across all GPUs
            if gpus:
                total_mem_used = sum(g.memoryUsed for g in gpus)
                total_mem_total = sum(g.memoryTotal for g in gpus)
                avg_utilization = sum(g.memoryUtil for g in gpus) / len(gpus)
                avg_load = sum(g.load for g in gpus) / len(gpus)
                
                stats['gpu_memory_used_gb'] = total_mem_used / 1024
                stats['gpu_memory_total_gb'] = total_mem_total / 1024
                stats['gpu_memory_utilization_%'] = avg_utilization * 100
                stats['gpu_load_%'] = avg_load * 100
                
                # Also log per-GPU for detailed tracking
                for i, gpu in enumerate(gpus):
                    stats[f'gpu_{i}_mem_used_gb'] = gpu.memoryUsed / 1024
                    stats[f'gpu_{i}_mem_util_%'] = gpu.memoryUtil * 100
            
            return stats
        except Exception as e:
            logger.debug(f"Could not get GPU stats: {e}")
            return {}
    
    def _get_system_stats(self) -> Dict[str, float]:
        """Get CPU and system memory statistics"""
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




@dataclass
class ModelArguments:
    """Model arguments"""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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
        metadata={"help": "Path to combined dataset"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )


@dataclass
class TrainingArguments(HFTrainingArguments):
    """Extended training arguments"""
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW optimizer beta1 (momentum)"}
    )


def tokenize_function(examples, tokenizer, max_seq_length):
    """Tokenize examples"""
    # Tokenize the text
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors=None,
    )
    
    # For causal LM, labels = input_ids but padding tokens have -100 (ignored in loss)
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    # Mask padding positions
    for i in range(len(tokenized['labels'])):
        tokenized['labels'][i] = [
            -100 if token_id == tokenizer.pad_token_id else token_id
            for token_id in tokenized['labels'][i]
        ]
    
    # Keep task and source info for per-task evaluation
    tokenized['task'] = examples['task']
    tokenized['source'] = examples['source']
    
    return tokenized


class MultiTaskTrainer(Trainer):
    """Custom trainer for multi-task dataset training with support for AdamW and Muon optimizers"""
    
    def __init__(self, *args, task_names=None, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_names = task_names or []
        self.config = config
    
    def create_optimizer(self):
        """Create optimizer (AdamW or Muon) based on config"""
        if self.config:
            # Use config-based optimizer creation
            logger.info("Creating optimizer from config...")
            self.optimizer = create_optimizer_from_config(
                self.model.parameters(),
                self.config,
                model=self.model,
            )
        else:
            # Fall back to HF's default optimizer creation
            logger.info("Creating default AdamW optimizer...")
            super().create_optimizer()
    
    def log_metrics(self, split, metrics, step=None):
        """Override to ensure metrics are properly logged"""
        super().log_metrics(split, metrics, step)
        # Also log to logger for visibility
        if split == "eval":
            logger.info(f"📊 Evaluation Metrics at step {step}:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not key.startswith('eval_'):
                    logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


def main():
    # Check if config.yaml path is provided
    config_file = None
    if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
        config_file = sys.argv[1]
        sys.argv = [sys.argv[0]]  # Remove config path from argv for argument parser
    
    # Load configuration
    if config_file:
        logger.info(f"Loading configuration from {config_file}")
        config = load_config_yaml(config_file)
        
        # Print config summary
        print_config_summary(config)
        
        # Set GPU devices from config
        gpu_config = get_gpu_config(config)
        device_ids = gpu_config.get('device_ids', '0')
        if device_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device_ids)
            logger.info(f"✓ Set CUDA_VISIBLE_DEVICES={device_ids}")
        
        # Configure TF32 if requested (faster but less precise)
        if gpu_config.get('use_tf32', True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("✓ Enabled TF32 for faster computation")
        
        # Convert config to training arguments
        training_args = config_dict_to_hf_training_args(config)
        
        # Extract model and data arguments from config
        model_config = get_model_config(config)
        dataset_config = get_dataset_config(config)
        lora_config_dict = get_lora_config(config)
        optimizer_config = get_optimizer_config(config)
        
        model_args = type('ModelArguments', (), {
            'model_name_or_path': model_config.get('model_id'),
            'lora_rank': lora_config_dict.get('lora_rank', 128),
            'lora_alpha': lora_config_dict.get('lora_alpha', 512),
            'lora_dropout': lora_config_dict.get('lora_dropout', 0.05),
        })()
        
        data_args = type('DataArguments', (), {
            'dataset_path': dataset_config.get('dataset_path'),
            'max_seq_length': dataset_config.get('max_seq_length', 512),
        })()
    else:
        # Parse command-line arguments
        parser = HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments)
        )
        
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            model_args, data_args, training_args = parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1])
            )
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
        config = None  # No config when using CLI args
        logger.info("Using command-line arguments (no config.yaml found)")
        
        # Set default GPU device (GPU 0) when using CLI args
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("✓ Set CUDA_VISIBLE_DEVICES=0 (CLI mode default)")
        logger.info("✓ Enabled TF32 for faster computation")
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    
    logger.info(f"Training parameters: {training_args}")
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Data parameters: {data_args}")
    
    # Log configuration source
    if config_file:
        logger.info(f"✓ Configuration loaded from YAML: {config_file}")
    else:
        logger.info("✓ Using command-line arguments")
    
    # Set seed
    set_seed(training_args.seed)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=data_args.max_seq_length,
        padding_side="right",
        use_fast=True,
    )
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    logger.info(f"Loading dataset from {data_args.dataset_path}")
    dataset_dict = load_from_disk(data_args.dataset_path)
    train_dataset = dataset_dict['train']
    eval_dataset = dataset_dict.get('validation', None)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Get unique tasks
    if 'task' in train_dataset.column_names:
        tasks = set(train_dataset['task'])
        logger.info(f"Found {len(tasks)} tasks: {tasks}")
    else:
        tasks = []
    
    # Tokenize datasets
    logger.info("Tokenizing train dataset...")
    tokenize_fn_partial = lambda examples: tokenize_function(
        examples, tokenizer, data_args.max_seq_length
    )
    
    train_dataset = train_dataset.map(
        tokenize_fn_partial,
        batched=True,
        num_proc=4,
        remove_columns=train_dataset.column_names,
    )
    
    if eval_dataset:
        logger.info("Tokenizing eval dataset...")
        eval_dataset = eval_dataset.map(
            tokenize_fn_partial,
            batched=True,
            num_proc=4,
            remove_columns=eval_dataset.column_names,
        )
        
        # Limit eval dataset to 100 samples for faster evaluation
        original_eval_size = len(eval_dataset)
        max_eval_samples = 100
        if original_eval_size > max_eval_samples:
            eval_dataset = eval_dataset.select(range(max_eval_samples))
            logger.info(f"✓ Limited eval dataset to {max_eval_samples} samples (was {original_eval_size} samples)")
        else:
            logger.info(f"✓ Eval dataset has {original_eval_size} samples (less than limit of {max_eval_samples})")
    
    # Load model
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    
    # Get device map from config if available
    if config:
        gpu_config = get_gpu_config(config)
        device_map = gpu_config.get('device_map', 'auto')
    else:
        device_map = 'auto'  # Default device map
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 instead of float16 to avoid AMP issues
        device_map=device_map,
    )
    logger.info(f"✓ Model loaded with device_map={device_map}")
    
    # Apply LoRA
    logger.info(f"Applying LoRA with rank={model_args.lora_rank}, alpha={model_args.lora_alpha}")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Setup training
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Create trainer with monitoring callback
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        task_names=list(tasks),
        config=config if config_file else None,
    )
    
    # Add monitoring callback with trainer reference
    monitoring_callback = MonitoringCallback()
    monitoring_callback.trainer = trainer  # Store reference for use in on_evaluate
    trainer.add_callback(monitoring_callback)
    
    logger.info("✓ Enhanced logging callback enabled:")
    logger.info("  - Train metrics: loss, perplexity, grad_norm, learning_rate")
    logger.info("  - GPU stats: memory used/total, utilization %, load %")
    logger.info("  - System stats: CPU %, system memory % and availability")
    logger.info("  - Eval metrics: loss, perplexity (every {training_args.eval_steps} steps)")
    logger.info("  - All metrics logged to wandb and tensorboard")
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    
    # Save training results
    with open(os.path.join(training_args.output_dir, "training_results.json"), "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"Results: {train_result.metrics}")


if __name__ == "__main__":
    main()
