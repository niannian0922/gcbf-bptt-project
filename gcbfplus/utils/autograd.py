import torch
from typing import Tuple, Any

class GDecay(torch.autograd.Function):
    """
    Custom autograd function that implements temporal gradient decay.
    
    During the forward pass, this function acts as an identity function.
    During the backward pass, it scales the incoming gradient by a decay factor.
    This implements the concept from the "Back to Newton's Laws" paper where
    gradients are decayed across time to stabilize training.
    """
    
    @staticmethod
    def forward(ctx: Any, input_tensor: torch.Tensor, decay_factor: float = 0.95) -> torch.Tensor:
        """
        Forward pass is the identity function, but we save the decay factor
        for the backward pass.
        
        Args:
            ctx: Context object to save information for the backward pass
            input_tensor: Input tensor to process
            decay_factor: Factor to multiply the gradient by in backward pass
            
        Returns:
            The input tensor (identity function)
        """
        ctx.decay_factor = decay_factor
        return input_tensor.clone()
    
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass multiplies the incoming gradient by the decay factor.
        
        Args:
            ctx: Context object with saved decay factor
            grad_output: Incoming gradient from the next layer
            
        Returns:
            Scaled gradient and None for the decay_factor parameter
        """
        return grad_output * ctx.decay_factor, None


def g_decay(input_tensor: torch.Tensor, decay_factor: float = 0.95) -> torch.Tensor:
    """
    Convenience function to apply gradient decay.
    
    Args:
        input_tensor: Input tensor to process
        decay_factor: Factor to multiply the gradient by in backward pass
        
    Returns:
        Tensor with gradient decay applied during backpropagation
    """
    return GDecay.apply(input_tensor, decay_factor) 