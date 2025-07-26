import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union, Any


class PerceptionModule(nn.Module):
    """
    Perception module for processing sensory input.
    
    This module can be configured to handle different types of inputs:
    - Graph-based observations (using GNNs)
    - Dense vector observations (using MLPs)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the perception module.
        
        Args:
            config: Configuration dictionary with keys:
                - 'input_dim': Input dimension size
                - 'hidden_dim': Hidden dimension size
                - 'num_layers': Number of hidden layers
                - 'activation': Activation function name
                - 'use_batch_norm': Whether to use batch normalization
        """
        super(PerceptionModule, self).__init__()
        
        # Extract configuration parameters with a default that matches our environment
        input_dim = config.get('input_dim', 9)  # Changed default from 12 to 9 to match environment
        hidden_dim = config.get('hidden_dim', 64)
        num_layers = config.get('num_layers', 2)
        activation = config.get('activation', 'relu')
        use_batch_norm = config.get('use_batch_norm', False)
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.05)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Build MLP layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(self.activation)
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through perception module.
        
        Args:
            x: Input tensor [batch_size, n_agents, input_dim] or [batch_size, input_dim]
            
        Returns:
            Processed features [batch_size, output_dim] or [batch_size, n_agents, output_dim]
        """
        original_shape = x.shape
        
        # Handle multi-agent observations
        if len(original_shape) == 3:
            batch_size, n_agents, input_dim = original_shape
            
            # Reshape to [batch_size * n_agents, input_dim]
            x_flat = x.reshape(batch_size * n_agents, input_dim)
            
            # Process through network
            features = self.network(x_flat)
            
            # Reshape back to [batch_size, n_agents, output_dim]
            return features.view(batch_size, n_agents, -1)
        else:
            # For simple batch processing [batch_size, input_dim]
            return self.network(x)


class MemoryModule(nn.Module):
    """
    Memory module for handling temporal information.
    
    Uses a GRU cell to maintain a memory of past observations.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the memory module.
        
        Args:
            config: Configuration dictionary with keys:
                - 'input_dim': Input dimension size
                - 'hidden_dim': Hidden state dimension size
                - 'num_layers': Number of GRU layers
                - 'bidirectional': Whether to use bidirectional GRU
        """
        super(MemoryModule, self).__init__()
        
        # Extract configuration parameters
        input_dim = config.get('input_dim', 64)
        hidden_dim = config.get('hidden_dim', 64)
        num_layers = config.get('num_layers', 1)
        bidirectional = config.get('bidirectional', False)
        
        # Create GRU cell
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        
        # Initialize hidden state
        self.hidden_state = None
    
    def forward(self, x: torch.Tensor, reset_memory: bool = False) -> torch.Tensor:
        """
        Process input through memory module.
        
        Args:
            x: Input tensor [batch_size, input_dim] or [batch_size, n_agents, input_dim]
            reset_memory: Whether to reset the memory state
            
        Returns:
            Updated features with temporal context [batch_size, hidden_dim] or [batch_size, n_agents, hidden_dim]
        """
        original_shape = x.shape
        device = x.device
        
        # Handle multi-agent observations
        if len(original_shape) == 3:
            batch_size, n_agents, input_dim = original_shape
            
            # Reshape to [batch_size * n_agents, input_dim]
            x_flat = x.reshape(batch_size * n_agents, input_dim)
            
            # Initialize or reset hidden state if needed
            if self.hidden_state is None or reset_memory or self.hidden_state.shape[0] != batch_size * n_agents:
                hidden = torch.zeros(batch_size * n_agents, self.hidden_dim, device=device)
            else:
                # Create a new tensor instead of modifying in place
                hidden = self.hidden_state.detach().clone()
            
            # Update hidden state
            new_hidden = self.gru_cell(x_flat, hidden)
            
            # Store the new hidden state (without breaking computational graph)
            self.hidden_state = new_hidden.detach().clone()
            
            # Reshape back to [batch_size, n_agents, hidden_dim]
            return new_hidden.view(batch_size, n_agents, -1)
        else:
            # For simple batch processing
            batch_size = x.shape[0]
            
            # Initialize or reset hidden state if needed
            if self.hidden_state is None or reset_memory or self.hidden_state.shape[0] != batch_size:
                hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
            else:
                # Create a new tensor instead of modifying in place
                hidden = self.hidden_state.detach().clone()
            
            # Update hidden state
            new_hidden = self.gru_cell(x, hidden)
            
            # Store the new hidden state (without breaking computational graph)
            self.hidden_state = new_hidden.detach().clone()
            
            return new_hidden
    
    def reset(self) -> None:
        """Reset the memory state."""
        self.hidden_state = None


class PolicyHeadModule(nn.Module):
    """
    Policy head module for generating actions.
    
    Transforms features into action outputs, optionally applying
    action bounds and other transformations.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the policy head module.
        
        Args:
            config: Configuration dictionary with keys:
                - 'input_dim': Input dimension size
                - 'output_dim': Action dimension size
                - 'hidden_dims': List of hidden layer dimensions
                - 'activation': Activation function name
                - 'output_activation': Output activation function name
                - 'action_scaling': Whether to scale actions to a specific range
                - 'action_bound': Bound for action scaling (if used)
        """
        super(PolicyHeadModule, self).__init__()
        
        # Extract configuration parameters
        input_dim = config.get('input_dim', 64)
        output_dim = config.get('output_dim', 2)
        hidden_dims = config.get('hidden_dims', [64])
        activation = config.get('activation', 'relu')
        output_activation = config.get('output_activation', 'tanh')
        self.action_scaling = config.get('action_scaling', True)
        self.action_bound = config.get('action_bound', 1.0)
        
        # Select activation functions
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.05)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
            
        if output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        elif output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate actions from features.
        
        Args:
            x: Input features [batch_size, input_dim] or [batch_size, n_agents, input_dim]
            
        Returns:
            Action outputs [batch_size, output_dim] or [batch_size, n_agents, output_dim]
        """
        original_shape = x.shape
        
        # Handle multi-agent features
        if len(original_shape) == 3:
            batch_size, n_agents, input_dim = original_shape
            
            # Reshape to [batch_size * n_agents, input_dim]
            x_flat = x.reshape(batch_size * n_agents, input_dim)
            
            # Process through network
            actions = self.network(x_flat)
            
            # Apply output activation
            actions = self.output_activation(actions)
            
            # Scale actions if needed
            if self.action_scaling:
                actions = actions * self.action_bound
            
            # Reshape back to [batch_size, n_agents, output_dim]
            return actions.view(batch_size, n_agents, -1)
        else:
            # For simple batch processing
            actions = self.network(x)
            actions = self.output_activation(actions)
            
            # Scale actions if needed
            if self.action_scaling:
                actions = actions * self.action_bound
                
            return actions


class BPTTPolicy(nn.Module):
    """
    Backpropagation Through Time (BPTT) Policy Network.
    
    This policy network is designed for end-to-end training through differentiable
    physics simulators. It consists of three main components:
    1. Perception module for processing observations
    2. Memory module for handling temporal information
    3. Policy head module for generating actions
    
    The network is fully differentiable and can be trained using BPTT.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the BPTT policy network.
        
        Args:
            config: Configuration dictionary with separate sections for
                perception, memory, and policy_head.
        """
        super(BPTTPolicy, self).__init__()
        
        # Extract sub-configurations
        perception_config = config.get('perception', {})
        memory_config = config.get('memory', {})
        policy_head_config = config.get('policy_head', {})
        
        # Create perception module
        self.perception = PerceptionModule(perception_config)
        
        # Update memory input dimension based on perception output
        if 'input_dim' not in memory_config:
            memory_config['input_dim'] = self.perception.output_dim
        
        # Create memory module
        self.memory = MemoryModule(memory_config)
        
        # Update policy head input dimension based on memory output
        if 'input_dim' not in policy_head_config:
            policy_head_config['input_dim'] = self.memory.output_dim
        
        # Create policy head module
        self.head = PolicyHeadModule(policy_head_config)
        
        # Store configuration
        self.config = config
    
    def forward(self, x: torch.Tensor, reset_memory: bool = False) -> torch.Tensor:
        """
        Process observations and generate actions.
        
        Args:
            x: Observation tensor [batch_size, *obs_shape]
            reset_memory: Whether to reset the memory state
            
        Returns:
            Action outputs [batch_size, action_dim]
        """
        # Process through perception module
        features = self.perception(x)
        
        # Process through memory module
        memory_features = self.memory(features, reset_memory)
        
        # Generate actions through policy head
        actions = self.head(memory_features)
        
        return actions
    
    def reset(self) -> None:
        """Reset the policy's internal state (e.g., memory)."""
        if hasattr(self, 'memory'):
            self.memory.reset()


class EnsemblePolicy(nn.Module):
    """
    Ensemble of policy networks.
    
    This policy combines multiple policy networks and can be used
    for techniques like bootstrapped ensembles or mixture of experts.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the ensemble policy.
        
        Args:
            config: Configuration dictionary with keys:
                - 'num_policies': Number of policies in the ensemble
                - 'policy_config': Configuration for each policy
                - 'ensemble_method': Method for combining outputs ('mean', 'weighted', etc.)
        """
        super(EnsemblePolicy, self).__init__()
        
        # Extract configuration parameters
        num_policies = config.get('num_policies', 3)
        policy_config = config.get('policy_config', {})
        self.ensemble_method = config.get('ensemble_method', 'mean')
        
        # Create ensemble of policies
        self.policies = nn.ModuleList(
            [BPTTPolicy(policy_config) for _ in range(num_policies)]
        )
        
        # If using weighted ensemble, create weights parameter
        if self.ensemble_method == 'weighted':
            self.weights = nn.Parameter(torch.ones(num_policies) / num_policies)
        
        # Store configuration
        self.config = config
    
    def forward(self, x: torch.Tensor, reset_memory: bool = False) -> torch.Tensor:
        """
        Process observations through the ensemble and combine outputs.
        
        Args:
            x: Observation tensor [batch_size, *obs_shape]
            reset_memory: Whether to reset the memory state
            
        Returns:
            Combined action outputs [batch_size, action_dim]
        """
        # Get actions from each policy
        policy_actions = [policy(x, reset_memory) for policy in self.policies]
        
        # Stack actions for combining [num_policies, batch_size, action_dim]
        stacked_actions = torch.stack(policy_actions)
        
        # Combine actions based on ensemble method
        if self.ensemble_method == 'mean':
            # Simple average
            actions = torch.mean(stacked_actions, dim=0)
        elif self.ensemble_method == 'weighted':
            # Weighted average
            weights = F.softmax(self.weights, dim=0)
            actions = torch.sum(stacked_actions * weights.view(-1, 1, 1), dim=0)
        else:
            # Default to mean
            actions = torch.mean(stacked_actions, dim=0)
        
        return actions
    
    def reset(self) -> None:
        """Reset all policies in the ensemble."""
        for policy in self.policies:
            policy.reset()


def create_policy_from_config(config: Dict) -> nn.Module:
    """
    Factory function to create a policy from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Policy network instance
    """
    policy_type = config.get('type', 'bptt')
    
    if policy_type == 'bptt':
        return BPTTPolicy(config)
    elif policy_type == 'ensemble':
        return EnsemblePolicy(config)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}") 