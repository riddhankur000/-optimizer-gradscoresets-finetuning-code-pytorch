#!/usr/bin/env python
# coding=utf-8
"""
YAML Configuration Loader for Sequential Training

Loads and parses the config.yaml file, extracts optimizer and GPU profiles,
and converts them to training arguments.
"""

import os
import yaml
import torch
import torch.distributed as dist
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class OptimizerConfig:
    """Parsed optimizer configuration"""
    optimizer_type: str
    # AdamW settings
    adam_beta1: Optional[float] = None
    adam_beta2: Optional[float] = None
    adam_epsilon: Optional[float] = None
    # Muon settings
    muon_lr: Optional[float] = None
    muon_weight_decay: Optional[float] = None
    muon_momentum: Optional[float] = None
    muon_nesterov: Optional[bool] = None
    muon_ns_coefficients: Optional[List[float]] = None
    muon_eps: Optional[float] = None
    muon_ns_steps: Optional[int] = None
    muon_adjust_lr_fn: Optional[str] = None


@dataclass
class GPUConfig:
    """Parsed GPU configuration"""
    device_ids: Optional[str] = None
    use_distributed: bool = False
    device_map: str = "auto"
    use_tf32: bool = True
    num_gpus: int = 1


class ConfigLoader:
    """Load and parse YAML configuration for sequential training"""
    
    def __init__(self, config_path: str):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to config.yaml file
        """
        self.config_path = config_path
        self.config = self._load_yaml()
        self.active_profiles = self.config.get('active_profiles', {})
        self.optimizer_name = self.active_profiles.get('optimizer', 'adamw')
        self.gpu_profile_name = self.active_profiles.get('gpu', 'gpu_0')
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get('model_config', {})
    
    def get_tokenizer_config(self) -> Dict[str, Any]:
        """Get tokenizer configuration"""
        return self.config.get('tokenizer_config', {})
    
    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration"""
        return self.config.get('lora_config', {})
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration"""
        return self.config.get('dataset_config', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.get('training_config', {})
    
    def get_optimizer_config(self) -> OptimizerConfig:
        """
        Get optimizer configuration based on active profile.
        
        Returns:
            OptimizerConfig with selected optimizer settings
        """
        optimizer_profiles = self.config.get('optimizer_profiles', {})
        optimizer_profile = optimizer_profiles.get(self.optimizer_name, {})
        
        return OptimizerConfig(
            optimizer_type=optimizer_profile.get('optimizer_type', self.optimizer_name),
            adam_beta1=optimizer_profile.get('adam_beta1'),
            adam_beta2=optimizer_profile.get('adam_beta2'),
            adam_epsilon=optimizer_profile.get('adam_epsilon'),
            muon_lr=optimizer_profile.get('muon_lr'),
            muon_weight_decay=optimizer_profile.get('muon_weight_decay'),
            muon_momentum=optimizer_profile.get('muon_momentum'),
            muon_nesterov=optimizer_profile.get('muon_nesterov'),
            muon_ns_coefficients=optimizer_profile.get('muon_ns_coefficients'),
            muon_eps=optimizer_profile.get('muon_eps'),
            muon_ns_steps=optimizer_profile.get('muon_ns_steps'),
            muon_adjust_lr_fn=optimizer_profile.get('muon_adjust_lr_fn'),
        )
    
    def get_gpu_config(self) -> GPUConfig:
        """
        Get GPU configuration based on active profile.
        
        Returns:
            GPUConfig with selected GPU settings
        """
        gpu_profiles = self.config.get('gpu_profiles', {})
        gpu_profile = gpu_profiles.get(self.gpu_profile_name, {})
        
        device_ids = gpu_profile.get('device_ids')
        use_distributed = gpu_profile.get('use_distributed', False)
        
        # Count number of GPUs
        num_gpus = 1
        if device_ids and isinstance(device_ids, str):
            num_gpus = len(device_ids.split(','))
        
        return GPUConfig(
            device_ids=device_ids,
            use_distributed=use_distributed,
            device_map=gpu_profile.get('device_map', 'auto'),
            use_tf32=gpu_profile.get('use_tf32', True),
            num_gpus=num_gpus,
        )
    
    def get_sequential_tasks_config(self) -> Dict[str, Any]:
        """Get sequential tasks configuration for Riemannian method"""
        return self.config.get('sequential_tasks_config', {})
    
    def get_multitask_config(self) -> Dict[str, Any]:
        """Get multi-task configuration"""
        return self.config.get('multitask_config', {})
    
    def get_wandb_config(self) -> Dict[str, Any]:
        """Get WandB configuration"""
        return self.config.get('wandb_config', {})
    
    def setup_distributed_training(self, gpu_config: GPUConfig) -> None:
        """
        Setup distributed training if using multiple GPUs.
        
        Args:
            gpu_config: GPU configuration
        """
        if gpu_config.use_distributed and gpu_config.num_gpus > 1:
            if not dist.is_initialized():
                # Set device IDs
                if gpu_config.device_ids:
                    device_ids = gpu_config.device_ids.split(',')
                    local_rank = int(os.environ.get('LOCAL_RANK', 0))
                    torch.cuda.set_device(int(device_ids[local_rank]))
                
                dist.init_process_group(backend='nccl')
    
    def build_training_arguments(self) -> Dict[str, Any]:
        """
        Build HuggingFace TrainingArguments from config.
        
        Returns:
            Dictionary of training arguments
        """
        training_config = self.get_training_config()
        gpu_config = self.get_gpu_config()
        optimizer_config = self.get_optimizer_config()
        
        # Build base training args
        num_train_epochs = training_config.get('num_train_epochs', 3)
        if num_train_epochs is None:
            num_train_epochs = 1  # Default to 1 epoch if using max_steps
        
        max_steps = training_config.get('max_steps', -1)
        if max_steps is None:
            max_steps = -1  # Use epochs if max_steps is None
        
        args = {
            'output_dir': training_config.get('output_dir', './outputs'),
            'num_train_epochs': num_train_epochs,
            'max_steps': max_steps,
            'per_device_train_batch_size': training_config.get('per_device_train_batch_size', 8),
            'per_device_eval_batch_size': training_config.get('per_device_eval_batch_size', 8),
            'gradient_accumulation_steps': training_config.get('gradient_accumulation_steps', 1),
            'learning_rate': training_config.get('learning_rate', 2e-4),
            'lr_scheduler_type': training_config.get('lr_scheduler_type', 'linear'),
            'warmup_ratio': training_config.get('warmup_ratio', 0.1),
            'weight_decay': training_config.get('weight_decay', 0.01),
            'max_grad_norm': training_config.get('max_grad_norm', 1.0),
            'optim': 'adamw_torch' if optimizer_config.optimizer_type == 'adamw' else 'adamw_torch',  # Trainer doesn't have muon
            'bf16': training_config.get('bf16', True),
            'fp16': training_config.get('fp16', False),
            'save_strategy': training_config.get('save_strategy', 'steps'),
            'save_steps': training_config.get('save_steps', 256),
            'save_total_limit': training_config.get('save_total_limit', 3),
            'eval_strategy': training_config.get('eval_strategy', 'steps'),
            'eval_steps': training_config.get('eval_steps', 16),
            'logging_strategy': training_config.get('logging_strategy', 'steps'),
            'logging_steps': training_config.get('logging_steps', 10),
            'seed': training_config.get('seed', 0),
            'data_seed': training_config.get('data_seed', 0),
            'disable_tqdm': training_config.get('disable_tqdm', False),
            'remove_unused_columns': training_config.get('remove_unused_columns', True),
            'report_to': training_config.get('report_to', ['wandb']),
            'run_name': training_config.get('run_name', 'sequential_training'),
            'ddp_find_unused_parameters': False,
            'gradient_checkpointing': True,
        }
        
        # Set small_batch_ratio for subset selection (not a TrainingArguments parameter)
        # This will be set manually on the TrainingArguments object after creation
        # Default = 1.0 (use full batch size)
        
        # Add distributed training args
        if gpu_config.use_distributed and gpu_config.num_gpus > 1:
            args['ddp'] = True
        
        return args
    
    def print_config_summary(self) -> None:
        """Print a summary of the loaded configuration"""
        print("\n" + "=" * 80)
        print("CONFIGURATION SUMMARY")
        print("=" * 80)
        
        print(f"\nActive Profiles:")
        print(f"  Optimizer: {self.optimizer_name}")
        print(f"  GPU Profile: {self.gpu_profile_name}")
        
        model_config = self.get_model_config()
        print(f"\nModel Configuration:")
        print(f"  Model: {model_config.get('model_id', 'N/A')}")
        print(f"  Data Type: {model_config.get('torch_dtype', 'N/A')}")
        
        lora_config = self.get_lora_config()
        print(f"\nLoRA Configuration:")
        print(f"  Enabled: {lora_config.get('enabled', False)}")
        print(f"  Rank: {lora_config.get('lora_rank', 'N/A')}")
        print(f"  Alpha: {lora_config.get('lora_alpha', 'N/A')}")
        
        dataset_config = self.get_dataset_config()
        print(f"\nDataset Configuration:")
        print(f"  Path: {dataset_config.get('dataset_path', 'N/A')}")
        print(f"  Max Length: {dataset_config.get('max_seq_length', 'N/A')}")
        
        training_config = self.get_training_config()
        print(f"\nTraining Configuration:")
        print(f"  Output Dir: {training_config.get('output_dir', 'N/A')}")
        print(f"  Batch Size: {training_config.get('per_device_train_batch_size', 'N/A')}")
        print(f"  Learning Rate: {training_config.get('learning_rate', 'N/A')}")
        print(f"  Max Steps: {training_config.get('max_steps', 'N/A')}")
        
        optimizer_config = self.get_optimizer_config()
        print(f"\nOptimizer Configuration ({self.optimizer_name}):")
        if optimizer_config.optimizer_type == 'adamw':
            print(f"  Beta1: {optimizer_config.adam_beta1}")
            print(f"  Beta2: {optimizer_config.adam_beta2}")
            print(f"  Epsilon: {optimizer_config.adam_epsilon}")
        elif optimizer_config.optimizer_type == 'muon':
            print(f"  LR: {optimizer_config.muon_lr}")
            print(f"  Weight Decay: {optimizer_config.muon_weight_decay}")
            print(f"  Momentum: {optimizer_config.muon_momentum}")
            print(f"  Nesterov: {optimizer_config.muon_nesterov}")
        
        gpu_config = self.get_gpu_config()
        print(f"\nGPU Configuration ({self.gpu_profile_name}):")
        print(f"  Device IDs: {gpu_config.device_ids}")
        print(f"  Distributed: {gpu_config.use_distributed}")
        print(f"  Num GPUs: {gpu_config.num_gpus}")
        print(f"  Device Map: {gpu_config.device_map}")
        print(f"  TF32: {gpu_config.use_tf32}")
        
        print("\n" + "=" * 80 + "\n")


def load_config(config_path: str) -> ConfigLoader:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to config.yaml file
    
    Returns:
        ConfigLoader instance
    """
    return ConfigLoader(config_path)
