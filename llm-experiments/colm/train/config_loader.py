#!/usr/bin/env python
# coding=utf-8
"""
Configuration loading utilities for multi-task training
"""

import yaml
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from transformers import TrainingArguments as HFTrainingArguments

logger = logging.getLogger(__name__)


def load_config_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file and apply active profiles.
    
    Profiles allow selecting predefined configurations without maintaining separate files:
    - active_profiles.optimizer: selects optimizer configuration ("adamw" or "muon")
    - active_profiles.gpu: selects GPU configuration ("gpu_0", "gpu_1", "gpu_multi", "cpu")
    
    Args:
        config_path: Path to config.yaml file
    
    Returns:
        Dictionary with configuration and profiles applied
    """
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply active profiles
    config = apply_active_profiles(config)
    
    return config


def apply_active_profiles(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply active optimizer and GPU profiles to configuration.
    
    Profile selection happens in active_profiles section:
      active_profiles:
        optimizer: "adamw"  # or "muon"
        gpu: "gpu_0"        # or "gpu_1", "gpu_multi"
    
    The selected profile settings are merged into the main config sections.
    
    Args:
        config: Configuration dictionary (from YAML)
    
    Returns:
        Configuration with profiles applied
    """
    # Get active profile selections
    active_profiles = config.get('active_profiles', {})
    optimizer_profile_name = active_profiles.get('optimizer', 'adamw')
    gpu_profile_name = active_profiles.get('gpu', 'gpu_0')
    
    # Get profile definitions
    optimizer_profiles = config.get('optimizer_profiles', {})
    gpu_profiles = config.get('gpu_profiles', {})
    
    # Merge optimizer profile into optimizer_config
    if optimizer_profile_name in optimizer_profiles:
        optimizer_config = optimizer_profiles[optimizer_profile_name]
        config['optimizer_config'] = optimizer_config
        logger.info(f"✓ Applied optimizer profile: {optimizer_profile_name}")
    else:
        logger.warning(f"⚠ Optimizer profile '{optimizer_profile_name}' not found, using defaults")
    
    # Merge GPU profile into gpu_config
    if gpu_profile_name in gpu_profiles:
        gpu_config = gpu_profiles[gpu_profile_name]
        config['gpu_config'] = gpu_config
        logger.info(f"✓ Applied GPU profile: {gpu_profile_name}")
    else:
        logger.warning(f"⚠ GPU profile '{gpu_profile_name}' not found, using defaults")
    
    return config


def config_dict_to_hf_training_args(config: Dict[str, Any]) -> HFTrainingArguments:
    """
    Convert config dictionary to HuggingFace TrainingArguments
    
    Args:
        config: Config dictionary (from YAML)
    
    Returns:
        HuggingFace TrainingArguments object
    """
    training_config = config.get('training_config', {})
    optimizer_config = config.get('optimizer_config', {})
    
    # Check if using custom optimizer (Muon)
    optimizer_type = optimizer_config.get('optimizer_type', 'adamw').lower()
    
    # Extract relevant training arguments
    args_dict = {
        # Output and checkpointing
        'output_dir': training_config.get('output_dir', './out'),
        'save_strategy': training_config.get('save_strategy', 'steps'),
        'save_steps': training_config.get('save_steps', 256),
        'save_total_limit': training_config.get('save_total_limit', 3),
        
        # Training loop
        'num_train_epochs': training_config.get('num_train_epochs', 1),
        'max_steps': training_config.get('max_steps', 1000),
        'per_device_train_batch_size': training_config.get('per_device_train_batch_size', 2),
        'per_device_eval_batch_size': training_config.get('per_device_eval_batch_size', 4),
        'gradient_accumulation_steps': training_config.get('gradient_accumulation_steps', 8),
        
        # Optimization
        'learning_rate': training_config.get('learning_rate', 1e-4),
        'lr_scheduler_type': training_config.get('lr_scheduler_type', 'linear'),
        'warmup_ratio': training_config.get('warmup_ratio', 0.03),
        'warmup_steps': training_config.get('warmup_steps', 0),
        'weight_decay': training_config.get('weight_decay', 0.01),
        'max_grad_norm': training_config.get('max_grad_norm', 1.0),
        
        # Only set 'optim' for non-Muon optimizers (HF doesn't know about Muon)
        'optim': 'adamw_torch' if optimizer_type != 'muon' else 'adamw_torch',
        
        # Precision
        'fp16': training_config.get('fp16', False),
        'bf16': training_config.get('bf16', True),
        
        # Evaluation
        'eval_strategy': training_config.get('eval_strategy', 'steps'),
        'eval_steps': training_config.get('eval_steps', 256),
        
        # Logging
        'logging_strategy': training_config.get('logging_strategy', 'steps'),
        'logging_steps': training_config.get('logging_steps', 10),
        'log_level': training_config.get('log_level', 'info'),
        
        # Reproducibility
        'seed': training_config.get('seed', 0),
        'data_seed': training_config.get('data_seed', 0),
        
        # Other
        'report_to': training_config.get('report_to', 'wandb').split(','),
        'run_name': training_config.get('run_name', 'training-run'),
        'disable_tqdm': training_config.get('disable_tqdm', False),
        'remove_unused_columns': training_config.get('remove_unused_columns', True),
        'local_rank': training_config.get('local_rank', -1),
    }
    
    # Add optimizer settings (only for AdamW)
    if optimizer_type != 'muon':
        args_dict['adam_beta1'] = optimizer_config.get('adam_beta1', 0.9)
        args_dict['adam_beta2'] = optimizer_config.get('adam_beta2', 0.999)
        args_dict['adam_epsilon'] = optimizer_config.get('adam_epsilon', 1e-8)
    
    # Filter out None values and create TrainingArguments
    args_dict = {k: v for k, v in args_dict.items() if v is not None}
    
    return HFTrainingArguments(**args_dict)


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get model configuration"""
    return config.get('model_config', {})


def get_tokenizer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get tokenizer configuration"""
    return config.get('tokenizer_config', {})


def get_lora_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get LoRA configuration"""
    return config.get('lora_config', {})


def get_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get dataset configuration"""
    return config.get('dataset_config', {})


def get_optimizer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get optimizer configuration"""
    return config.get('optimizer_config', {})


def get_multitask_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get multi-task configuration"""
    return config.get('multitask_config', {})


def get_gpu_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get GPU configuration"""
    return config.get('gpu_config', {})


def print_config_summary(config: Dict[str, Any]):
    """Print configuration summary"""
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    
    # Active Profiles
    active_profiles = config.get('active_profiles', {})
    print(f"\n📋 Active Profiles:")
    print(f"  Optimizer: {active_profiles.get('optimizer', 'adamw')}")
    print(f"  GPU: {active_profiles.get('gpu', 'gpu_0')}")
    
    # Model
    model_cfg = get_model_config(config)
    print(f"\n📦 Model:")
    print(f"  Model ID: {model_cfg.get('model_id')}")
    print(f"  Dtype: {model_cfg.get('torch_dtype')}")
    
    # LoRA
    lora_cfg = get_lora_config(config)
    print(f"\n🧬 LoRA:")
    print(f"  Rank: {lora_cfg.get('lora_rank')}")
    print(f"  Alpha: {lora_cfg.get('lora_alpha')}")
    print(f"  Dropout: {lora_cfg.get('lora_dropout')}")
    
    # Training
    train_cfg = config.get('training_config', {})
    print(f"\n⚙️ Training:")
    print(f"  Batch Size: {train_cfg.get('per_device_train_batch_size')}")
    print(f"  Gradient Accumulation: {train_cfg.get('gradient_accumulation_steps')}")
    print(f"  Max Steps: {train_cfg.get('max_steps')}")
    print(f"  Learning Rate: {train_cfg.get('learning_rate')}")
    print(f"  Scheduler: {train_cfg.get('lr_scheduler_type')}")
    
    # Optimizer
    opt_cfg = get_optimizer_config(config)
    optimizer_type = opt_cfg.get('optimizer_type', 'adamw').upper()
    print(f"\n🔧 Optimizer: {optimizer_type}")
    
    if optimizer_type == 'ADAMW':
        print(f"  Beta1 (Momentum): {opt_cfg.get('adam_beta1')}")
        print(f"  Beta2: {opt_cfg.get('adam_beta2')}")
        print(f"  Epsilon: {opt_cfg.get('adam_epsilon')}")
    elif optimizer_type == 'MUON':
        print(f"  Learning Rate: {opt_cfg.get('muon_lr')}")
        print(f"  Weight Decay: {opt_cfg.get('muon_weight_decay')}")
        print(f"  Momentum: {opt_cfg.get('muon_momentum')}")
        print(f"  Nesterov: {opt_cfg.get('muon_nesterov')}")
        print(f"  NS Steps: {opt_cfg.get('muon_ns_steps')}")
        print(f"  Adjust LR: {opt_cfg.get('muon_adjust_lr_fn')}")
    
    # Dataset
    ds_cfg = get_dataset_config(config)
    print(f"\n📊 Dataset:")
    print(f"  Path: {ds_cfg.get('dataset_path')}")
    print(f"  Datasets: {', '.join(ds_cfg.get('dataset_names', []))}")
    print(f"  Max Seq Length: {ds_cfg.get('max_seq_length')}")
    
    # GPU
    gpu_cfg = get_gpu_config(config)
    print(f"\n🖥️ GPU Configuration:")
    print(f"  Device IDs: {gpu_cfg.get('device_ids', '0')}")
    print(f"  Distributed: {gpu_cfg.get('use_distributed', False)}")
    print(f"  Device Map: {gpu_cfg.get('device_map', 'auto')}")
    print(f"  TF32: {gpu_cfg.get('use_tf32', True)}")
    
