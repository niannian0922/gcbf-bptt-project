import torch
import torch.nn as nn
from typing import Any


class GDecay(torch.autograd.Function):
    """
    Temporal Gradient Decay (GDecay) autograd function.
    
    This function implements gradient decay for stabilizing long-horizon BPTT training.
    The forward pass simply returns the input tensor unchanged, but the backward pass
    multiplies the incoming gradient by a decay factor to prevent gradient explosion
    in recurrent computations.
    
    Reference: Based on the diffphysdrone implementation for stabilizing differentiable
    physics simulations with long rollouts.
    """
    
    @staticmethod
    def forward(ctx: Any, input_tensor: torch.Tensor, decay_factor: float) -> torch.Tensor:
        """
        Forward pass: simply return the input tensor unchanged.
        
        Args:
            ctx: Context object to save information for backward pass
            input_tensor: Input tensor to apply decay to
            decay_factor: Decay factor for gradient (saved for backward pass)
            
        Returns:
            The input tensor unchanged
        """
        # Save the decay factor for the backward pass
        ctx.decay_factor = decay_factor
        return input_tensor
    
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """
        Backward pass: multiply the incoming gradient by the decay factor.
        
        Args:
            ctx: Context object containing saved information from forward pass
            grad_output: Gradient flowing backward from the next layer
            
        Returns:
            Tuple of gradients for each input argument (input_tensor, decay_factor)
            Only the input_tensor gradient is meaningful; decay_factor gets None
        """
        # Apply decay to the gradient
        decayed_grad = grad_output * ctx.decay_factor
        
        # Return gradients for (input_tensor, decay_factor)
        # decay_factor doesn't need gradients, so return None for it
        return decayed_grad, None


def g_decay(input_tensor: torch.Tensor, decay_factor: float) -> torch.Tensor:
    """
    Convenience function to apply temporal gradient decay.
    
    This function applies the GDecay autograd function to stabilize gradients
    during backpropagation through time in differentiable physics simulations.
    
    Args:
        input_tensor: Tensor to apply gradient decay to
        decay_factor: Factor to multiply gradients by (should be < 1.0 for decay)
        
    Returns:
        Tensor with gradient decay applied (forward pass unchanged)
        
    Example:
        >>> import torch
        >>> x = torch.randn(10, 3, requires_grad=True)
        >>> decay_rate = 0.9
        >>> x_decayed = g_decay(x, decay_rate)
        >>> # x_decayed is identical to x in forward pass
        >>> # but gradients flowing to x will be multiplied by decay_rate
    """
    return GDecay.apply(input_tensor, decay_factor)


def apply_temporal_decay(
    tensor: torch.Tensor, 
    decay_rate: float, 
    dt: float,
    training: bool = True
) -> torch.Tensor:
    """
    Apply temporal gradient decay with time-step scaling.
    
    This function combines the decay rate with the simulation time step to create
    a time-scaled decay factor, and only applies decay during training.
    
    Args:
        tensor: Input tensor to apply decay to
        decay_rate: Base decay rate (typically 0.9-0.99)
        dt: Simulation time step
        training: Whether we're in training mode (decay only applied if True)
        
    Returns:
        Tensor with or without gradient decay applied
    """
    if training and decay_rate > 0.0:
        # Calculate time-scaled decay factor
        decay_factor = decay_rate ** dt
        return g_decay(tensor, decay_factor)
    else:
        # No decay during evaluation or if decay_rate is 0
        return tensor 