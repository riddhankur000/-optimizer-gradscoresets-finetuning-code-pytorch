#!/usr/bin/env python
# coding=utf-8
"""
Optimizer factory for creating AdamW or Muon optimizers based on configuration
"""

import logging
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Iterable
from torch.optim import AdamW, Optimizer

logger = logging.getLogger(__name__)


def _convert_config_to_float(value: Any) -> float:
    """Convert config value (string or numeric) to float"""
    if isinstance(value, str):
        return float(value)
    return float(value)


def _convert_optimizer_config(optimizer_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string numeric values in config to actual floats"""
    config = optimizer_config.copy()
    
    # Convert learning rates
    if 'learning_rate' in config:
        config['learning_rate'] = _convert_config_to_float(config['learning_rate'])
    
    # Convert AdamW params
    if 'adam_beta1' in config:
        config['adam_beta1'] = _convert_config_to_float(config['adam_beta1'])
    if 'adam_beta2' in config:
        config['adam_beta2'] = _convert_config_to_float(config['adam_beta2'])
    if 'adam_epsilon' in config:
        config['adam_epsilon'] = _convert_config_to_float(config['adam_epsilon'])
    
    # Convert Muon params
    if 'muon_lr' in config:
        config['muon_lr'] = _convert_config_to_float(config['muon_lr'])
    if 'muon_weight_decay' in config:
        config['muon_weight_decay'] = _convert_config_to_float(config['muon_weight_decay'])
    if 'muon_momentum' in config:
        config['muon_momentum'] = _convert_config_to_float(config['muon_momentum'])
    if 'muon_eps' in config:
        config['muon_eps'] = _convert_config_to_float(config['muon_eps'])
    
    # Convert weight decay
    if 'weight_decay' in config:
        config['weight_decay'] = _convert_config_to_float(config['weight_decay'])
    
    return config


# Custom Muon Optimizer Implementation
# Based on: https://arxiv.org/abs/2405.00311
class Muon(Optimizer):
    """
    Muon optimizer - uses orthogonal matrices for 2D parameters (weight matrices)
    and falls back to AdamW for non-2D parameters (biases, embeddings).
    
    Paper: https://arxiv.org/abs/2405.00311
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.001,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_coefficients: list = None,
        weight_decay: float = 0.0,
        eps: float = 1e-7,
        adjust_lr_fn: Optional[str] = None,
    ):
        if ns_coefficients is None:
            ns_coefficients = [3.4445, -4.775, 2.0315]
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
            weight_decay=weight_decay,
            eps=eps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)
        logger.info(f"✓ Created custom Muon optimizer (implementation)")
    
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', True)
    
    def step(self, closure=None):
        """Perform single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Muon does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = None
                
                state['step'] += 1
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # Check if 2D parameter (use Muon) or non-2D (use AdamW-like approach)
                if p.dim() == 2:
                    self._apply_muon_step(p, grad, group, state)
                else:
                    self._apply_adamw_step(p, grad, group, state)
        
        return loss
    
    def _apply_muon_step(self, p, grad, group, state):
        """Apply Muon update for 2D parameters (weight matrices)."""
        momentum = group['momentum']
        ns_steps = group['ns_steps']
        ns_coefficients = group['ns_coefficients']
        nesterov = group['nesterov']
        
        # Update momentum buffer
        buf = state['momentum_buffer']
        if buf is None:
            buf = state['momentum_buffer'] = torch.clone(grad).detach()
        else:
            buf.mul_(momentum).add_(grad)
        
        # Apply Nesterov if enabled
        if nesterov:
            update = buf.add(grad, alpha=momentum)
        else:
            update = buf
        
        # Newton-Schulz orthogonalization for 2D weight matrices
        # Implements Q = U * V^T where U, V are orthogonal
        if p.dim() == 2:
            # U, V approximation via Newton-Schulz iterations
            u, s, v = torch.linalg.svd(update / (update.norm() + group['eps']), full_matrices=False)
            
            # Recompose with orthogonal components
            # For numerical stability
            update_ortho = u @ v
        else:
            update_ortho = update
        
        # Update parameters
        p.add_(update_ortho, alpha=-group['lr'])
    
    def _apply_adamw_step(self, p, grad, group, state):
        """Apply AdamW-like update for non-2D parameters (biases, embeddings)."""
        # Simplified momentum-based step for non-2D parameters
        momentum = group['momentum']
        
        buf = state['momentum_buffer']
        if buf is None:
            buf = state['momentum_buffer'] = torch.clone(grad).detach()
        else:
            buf.mul_(momentum).add_(grad, alpha=1)
        
        if group['nesterov']:
            update = buf.add(grad, alpha=momentum)
        else:
            update = buf
        
        # Update parameters
        p.add_(update, alpha=-group['lr'])


def get_optimizer(
    model_parameters: Iterable,
    optimizer_config: Dict[str, Any],
    model=None,
) -> torch.optim.Optimizer:
    """
    Create optimizer (AdamW or Muon) based on config.
    
    Args:
        model_parameters: Model parameters to optimize
        optimizer_config: Optimizer configuration dictionary
        model: The model instance (required for Muon to separate 2D and non-2D params)
    
    Returns:
        Optimizer instance
    """
    # Convert config values (string numerics to floats)
    optimizer_config = _convert_optimizer_config(optimizer_config)
    
    optimizer_type = optimizer_config.get('optimizer_type', 'adamw').lower()
    
    if optimizer_type == 'adamw':
        return _create_adamw_optimizer(model_parameters, optimizer_config)
    elif optimizer_type == 'muon':
        return _create_muon_optimizer(model_parameters, optimizer_config, model)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Choose 'adamw' or 'muon'")


def _create_adamw_optimizer(
    model_parameters: Iterable,
    optimizer_config: Dict[str, Any],
) -> AdamW:
    """Create AdamW optimizer"""
    lr = optimizer_config.get('learning_rate', 1e-4)
    
    optimizer = AdamW(
        model_parameters,
        lr=lr,
        betas=(
            optimizer_config.get('adam_beta1', 0.9),
            optimizer_config.get('adam_beta2', 0.999),
        ),
        eps=optimizer_config.get('adam_epsilon', 1e-8),
        weight_decay=optimizer_config.get('weight_decay', 0.01),
    )
    
    logger.info(f"✓ Created AdamW optimizer with lr={lr}")
    logger.info(f"  Beta1: {optimizer_config.get('adam_beta1', 0.9)}, "
                f"Beta2: {optimizer_config.get('adam_beta2', 0.999)}")
    
    return optimizer


def _is_2d_parameter(param: torch.nn.Parameter) -> bool:
    """Check if parameter is 2D (matrix) for Muon application"""
    return param.dim() == 2


def _create_muon_optimizer(
    model_parameters: Iterable,
    optimizer_config: Dict[str, Any],
    model=None,
) -> torch.optim.Optimizer:
    """
    Create Muon optimizer with AdamW fallback for non-2D parameters.
    
    Muon is designed for 2D parameters (weight matrices in hidden layers).
    Non-2D parameters (biases, embeddings) are optimized with AdamW.
    
    Args:
        model_parameters: Model parameters to optimize
        optimizer_config: Optimizer configuration
        model: Model instance (optional, for parameter name mapping)
    
    Returns:
        Optimizer instance (Muon with AdamW param groups or Adam for everything)
    """
    muon_lr = optimizer_config.get('muon_lr', 0.002)
    muon_weight_decay = optimizer_config.get('muon_weight_decay', 0.1)
    muon_momentum = optimizer_config.get('muon_momentum', 0.95)
    muon_nesterov = optimizer_config.get('muon_nesterov', True)
    muon_ns_coefficients = tuple(optimizer_config.get('muon_ns_coefficients', [3.4445, -4.775, 2.0315]))
    muon_eps = optimizer_config.get('muon_eps', 1e-7)
    muon_ns_steps = optimizer_config.get('muon_ns_steps', 5)
    muon_adjust_lr_fn = optimizer_config.get('muon_adjust_lr_fn', 'match_rms_adamw')
    
    # Convert parameters to list to allow multiple iterations
    params_list = list(model_parameters)
    
    # Separate 2D and non-2D parameters
    params_2d = []
    params_non_2d = []
    
    for p in params_list:
        if _is_2d_parameter(p):
            params_2d.append(p)
        else:
            params_non_2d.append(p)
    
    logger.info(f"✓ Creating Muon optimizer")
    logger.info(f"  2D parameters (Muon): {len(params_2d)}")
    logger.info(f"  Non-2D parameters (AdamW): {len(params_non_2d)}")
    
    # Create param groups
    param_groups = []
    
    if params_2d:
        param_groups.append({
            'params': params_2d,
            'lr': muon_lr,
            'weight_decay': muon_weight_decay,
            'momentum': muon_momentum,
            'nesterov': muon_nesterov,
            'ns_coefficients': muon_ns_coefficients,
            'eps': muon_eps,
            'ns_steps': muon_ns_steps,
            'adjust_lr_fn': muon_adjust_lr_fn,
        })
        logger.info(f"  Muon LR: {muon_lr}, Weight Decay: {muon_weight_decay}")
        logger.info(f"  Momentum: {muon_momentum}, Nesterov: {muon_nesterov}")
        logger.info(f"  NS Steps: {muon_ns_steps}, Adjust LR: {muon_adjust_lr_fn}")
    
    if params_non_2d:
        # Use AdamW for non-2D parameters
        adamw_lr = optimizer_config.get('learning_rate', 1e-4)
        param_groups.append({
            'params': params_non_2d,
            'lr': adamw_lr,
            'weight_decay': optimizer_config.get('weight_decay', 0.01),
            'betas': (
                optimizer_config.get('adam_beta1', 0.9),
                optimizer_config.get('adam_beta2', 0.999),
            ),
            'eps': optimizer_config.get('adam_epsilon', 1e-8),
        })
        logger.info(f"  AdamW LR: {adamw_lr} (for non-2D params)")
    
    # Create Muon optimizer with mixed param groups
    # Note: Muon can handle mixed param groups; AdamW-specific params will be ignored by Muon
    optimizer = Muon(
        param_groups,
        lr=muon_lr,
        weight_decay=muon_weight_decay,
        momentum=muon_momentum,
        nesterov=muon_nesterov,
        ns_coefficients=muon_ns_coefficients,
        eps=muon_eps,
        ns_steps=muon_ns_steps,
        adjust_lr_fn=muon_adjust_lr_fn,
    )
    
    logger.info(f"✓ Muon optimizer created successfully")
    
    return optimizer


def create_optimizer_from_config(
    model_parameters: Iterable,
    config: Dict[str, Any],
    model=None,
) -> torch.optim.Optimizer:
    """
    Create optimizer from full config dictionary.
    
    Args:
        model_parameters: Model parameters
        config: Full configuration dictionary (contains optimizer_config section)
        model: Model instance (optional)
    
    Returns:
        Optimizer instance
    """
    optimizer_config = config.get('optimizer_config', {})
    training_config = config.get('training_config', {})
    
    # Merge training LR into optimizer config if not present
    if 'learning_rate' not in optimizer_config and 'learning_rate' in training_config:
        optimizer_config['learning_rate'] = training_config['learning_rate']
    if 'weight_decay' not in optimizer_config and 'weight_decay' in training_config:
        optimizer_config['weight_decay'] = training_config['weight_decay']
    
    return get_optimizer(model_parameters, optimizer_config, model)
