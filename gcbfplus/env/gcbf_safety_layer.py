import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Callable, Any, List, Union
from dataclasses import dataclass

from .multi_agent_env import MultiAgentState


class GCBFSafetyLayer(nn.Module):
    """
    Differentiable safety layer implementing Control Barrier Function (CBF) constraints.
    
    This layer takes raw actions and filters them to ensure safety constraints
    are satisfied. It is designed to be used as part of the environment's apply_safety_layer
    method.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the safety layer.
        
        Args:
            config: Dictionary containing safety layer parameters
                Required keys:
                - 'alpha': CBF parameter alpha (h_dot + alpha * h >= 0)
                - 'eps': Small positive parameter for numerical stability
                - 'safety_margin': Safety distance margin for collision avoidance
                Optional keys:
                - 'use_qp': Whether to use QP solver (otherwise uses simpler projection)
                - 'qp_relaxation_weight': Weight for constraint relaxation
                - 'max_iterations': Maximum number of iterations for solvers
        """
        super(GCBFSafetyLayer, self).__init__()
        
        # CBF parameters
        self.alpha = config.get('alpha', 1.0)
        self.eps = config.get('eps', 0.02)
        self.safety_margin = config.get('safety_margin', 0.05)
        
        # QP parameters
        self.use_qp = config.get('use_qp', True)
        self.qp_relaxation_weight = config.get('qp_relaxation_weight', 10.0)
        self.max_iterations = config.get('max_iterations', 10)
        
        # Register parameters
        self.register_buffer('alpha_tensor', torch.tensor([self.alpha], dtype=torch.float32))
        
    def barrier_function(self, state: MultiAgentState) -> torch.Tensor:
        """
        Compute barrier function values for the current state.
        
        The barrier function h(x) is defined such that h(x) >= 0 for safe states
        and h(x) < 0 for unsafe states.
        
        Args:
            state: Current environment state
            
        Returns:
            Barrier function values [batch_size, n_agents, n_constraints]
        """
        batch_size = state.batch_size
        n_agents = state.positions.shape[1]
        device = state.positions.device
        
        # Compute agent-agent barrier functions (collision avoidance)
        # For each pair of agents, h(x) = ||p_i - p_j||^2 - (2r)^2 where r is agent radius
        
        # Compute pairwise differences between positions
        # Shape: [batch, n_agents, n_agents, pos_dim]
        pos_diff = state.positions.unsqueeze(2) - state.positions.unsqueeze(1)
        
        # Compute squared distances
        # Shape: [batch, n_agents, n_agents]
        dist_squared = torch.sum(pos_diff**2, dim=3)
        
        # Compute threshold (2 * radius + margin)^2
        threshold = (2 * (self.safety_margin + 0.05))**2
        
        # Create barrier values: h(x) = dist_squared - threshold
        # Shape: [batch, n_agents, n_agents]
        h_agent_agent = dist_squared - threshold
        
        # Set diagonal to large values (no self-collision)
        mask = torch.eye(n_agents, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        h_agent_agent = h_agent_agent.masked_fill(mask == 1, 1000.0)
        
        # If there are obstacles, compute agent-obstacle barrier functions
        if state.obstacles is not None:
            # Extract obstacle positions and radii
            obstacle_positions = state.obstacles[..., :-1]  # [batch, n_obs, pos_dim]
            obstacle_radii = state.obstacles[..., -1:]     # [batch, n_obs, 1]
            
            # Compute differences between agent and obstacle positions
            # Shape: [batch, n_agents, n_obs, pos_dim]
            obs_diff = state.positions.unsqueeze(2) - obstacle_positions.unsqueeze(1)
            
            # Compute squared distances
            # Shape: [batch, n_agents, n_obs]
            obs_dist_squared = torch.sum(obs_diff**2, dim=3)
            
            # Compute threshold: (agent_radius + obstacle_radius + margin)^2
            # Shape: [batch, 1, n_obs]
            obs_threshold = (0.05 + obstacle_radii.squeeze(-1).unsqueeze(1) + self.safety_margin)**2
            
            # Create barrier values: h(x) = dist_squared - threshold
            # Shape: [batch, n_agents, n_obs]
            h_agent_obs = obs_dist_squared - obs_threshold
            
            # Combine agent-agent and agent-obstacle constraints
            # Shape: [batch, n_agents, n_agents + n_obs]
            h = torch.cat([h_agent_agent, h_agent_obs], dim=2)
        else:
            # Just agent-agent constraints
            h = h_agent_agent
        
        return h
    
    def barrier_jacobian(self, state: MultiAgentState) -> torch.Tensor:
        """
        Compute the Jacobian (gradient) of the barrier function with respect to states.
        
        Args:
            state: Current environment state
            
        Returns:
            Jacobian tensor [batch_size, n_agents, n_constraints, state_dim]
        """
        # We need to compute gradients, so enable autograd
        batch_size = state.batch_size
        n_agents = state.positions.shape[1]
        pos_dim = state.positions.shape[2]
        device = state.positions.device
        
        # Create computation graph for automatic differentiation
        positions = state.positions.clone().requires_grad_(True)
        
        # Compute pairwise differences between positions
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        
        # Compute squared distances
        dist_squared = torch.sum(pos_diff**2, dim=3)
        
        # Compute threshold
        threshold = (2 * (self.safety_margin + 0.05))**2
        
        # Create barrier values: h(x) = dist_squared - threshold
        h_agent_agent = dist_squared - threshold
        
        # Set diagonal to large values (no self-collision)
        mask = torch.eye(n_agents, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        h_agent_agent = h_agent_agent.masked_fill(mask == 1, 1000.0)
        
        # Get number of constraints
        if state.obstacles is not None:
            n_obs = state.obstacles.shape[1]
            n_constraints = n_agents + n_obs
        else:
            n_constraints = n_agents
        
        # Initialize Jacobian tensor
        # We only compute gradients w.r.t. positions for simplicity
        # Full version would include velocities
        jacobian = torch.zeros(batch_size, n_agents, n_constraints, pos_dim*2, device=device)
        
        # For each agent and each constraint, compute the gradient
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    # Compute gradient of h_ij w.r.t. positions
                    # We're computing ∂h_ij/∂p_i and ∂h_ij/∂p_j
                    grad_outputs = torch.zeros_like(h_agent_agent)
                    grad_outputs[:, i, j] = 1.0
                    
                    # Autograd to get gradients
                    grads = torch.autograd.grad(
                        outputs=h_agent_agent,
                        inputs=positions,
                        grad_outputs=grad_outputs,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True
                    )[0]
                    
                    # Store gradients in Jacobian tensor
                    # For agent i, constraint j (which comes from agent j)
                    jacobian[:, i, j, :pos_dim] = grads[:, i]
        
        # If we have obstacles, compute gradients for agent-obstacle constraints
        if state.obstacles is not None:
            # Extract obstacle positions and radii
            obstacle_positions = state.obstacles[..., :-1]  # [batch, n_obs, pos_dim]
            obstacle_radii = state.obstacles[..., -1:]     # [batch, n_obs, 1]
            
            # Compute differences between agent and obstacle positions
            obs_diff = positions.unsqueeze(2) - obstacle_positions.unsqueeze(1)
            
            # Compute squared distances
            obs_dist_squared = torch.sum(obs_diff**2, dim=3)
            
            # Compute threshold
            obs_threshold = (0.05 + obstacle_radii.squeeze(-1).unsqueeze(1) + self.safety_margin)**2
            
            # Create barrier values: h(x) = dist_squared - threshold
            h_agent_obs = obs_dist_squared - obs_threshold
            
            # For each agent and each obstacle, compute the gradient
            for i in range(n_agents):
                for j in range(n_obs):
                    # Compute gradient of h_ij w.r.t. positions
                    grad_outputs = torch.zeros_like(h_agent_obs)
                    grad_outputs[:, i, j] = 1.0
                    
                    # Autograd to get gradients
                    grads = torch.autograd.grad(
                        outputs=h_agent_obs,
                        inputs=positions,
                        grad_outputs=grad_outputs,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True
                    )[0]
                    
                    # Store gradients in Jacobian tensor
                    # For agent i, constraint n_agents+j (which comes from obstacle j)
                    jacobian[:, i, n_agents+j, :pos_dim] = grads[:, i]
        
        return jacobian
    
    def control_affine_dynamics(self, state: MultiAgentState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the control-affine dynamics matrices f(x) and g(x).
        
        For a double integrator system:
        dx/dt = f(x) + g(x)u
        
        where f(x) = [vx, vy, 0, 0]^T
        and g(x) = [0, 0; 0, 0; 1/m, 0; 0, 1/m]
        
        Args:
            state: Current environment state
            
        Returns:
            Tuple of (f, g) where:
            - f: Drift term [batch_size, n_agents, state_dim]
            - g: Control input term [batch_size, n_agents, state_dim, action_dim]
        """
        batch_size = state.batch_size
        n_agents = state.positions.shape[1]
        pos_dim = state.positions.shape[2]
        device = state.positions.device
        
        # For a double integrator with state [x, y, vx, vy], we have:
        # dx/dt = vx
        # dy/dt = vy
        # dvx/dt = 1/m * fx
        # dvy/dt = 1/m * fy
        
        # Drift term f(x) = [vx, vy, 0, 0]^T
        f = torch.zeros(batch_size, n_agents, 2*pos_dim, device=device)
        f[:, :, :pos_dim] = state.velocities  # Position derivatives = velocities
        
        # Control input term g(x) = [0, 0; 0, 0; 1/m, 0; 0, 1/m]
        g = torch.zeros(batch_size, n_agents, 2*pos_dim, pos_dim, device=device)
        
        # Default mass = 1.0 if not specified
        m = 0.1  # Default mass
        
        # Set up control matrix - each force component affects only its corresponding velocity
        for i in range(pos_dim):
            g[:, :, pos_dim+i, i] = 1.0 / m
        
        return f, g
    
    def forward(
        self, 
        state: MultiAgentState, 
        raw_action: torch.Tensor, 
        dynamics_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Apply CBF-based safety filtering to raw actions.
        
        Args:
            state: Current environment state
            raw_action: Raw actions from policy [batch_size, n_agents, action_dim]
            dynamics_fn: Optional function to compute control-affine dynamics
            
        Returns:
            Safe actions [batch_size, n_agents, action_dim]
        """
        # Compute barrier function values
        h = self.barrier_function(state)
        
        # Compute barrier function Jacobian
        dh_dx = self.barrier_jacobian(state)
        
        # Compute control-affine dynamics
        if dynamics_fn is not None:
            f, g = dynamics_fn(state)
        else:
            f, g = self.control_affine_dynamics(state)
        
        # Compute Lie derivatives
        # L_f h = dh/dx * f(x)
        # L_g h = dh/dx * g(x)
        
        # Compute L_f h (drift term): dh/dx * f(x)
        # [batch, n_agents, n_constraints, state_dim] x [batch, n_agents, state_dim]
        # -> [batch, n_agents, n_constraints]
        L_f_h = torch.sum(dh_dx * f.unsqueeze(2), dim=3)
        
        # Compute L_g h (control term): dh/dx * g(x)
        # [batch, n_agents, n_constraints, state_dim] x [batch, n_agents, state_dim, action_dim]
        # -> [batch, n_agents, n_constraints, action_dim]
        L_g_h = torch.matmul(dh_dx.view(*dh_dx.shape[:-1], 1, -1), 
                            g.view(*g.shape[:-2], -1, g.shape[-1]))
        L_g_h = L_g_h.squeeze(-2)
        
        # CBF constraint: L_f h + L_g h * u + alpha * h >= 0
        # Rearranging: L_g h * u >= -L_f h - alpha * h
        
        # Right-hand side of constraint: -L_f h - alpha * h
        rhs = -L_f_h - self.alpha * h
        
        # If using QP solver
        if self.use_qp:
            # Solve quadratic program for each agent
            safe_action = raw_action.clone()
            
            batch_size = state.batch_size
            n_agents = state.positions.shape[1]
            
            # Process each batch and agent separately
            for b in range(batch_size):
                for i in range(n_agents):
                    # Skip if no constraints for this agent
                    if L_g_h[b, i].shape[0] == 0:
                        continue
                    
                    # QP formulation:
                    # min_u 0.5 * (u - u_raw)^T * (u - u_raw)
                    # s.t. L_g_h * u >= rhs
                    
                    # Filter constraints to include only active ones
                    # Active means the constraint is either violated or close to being violated
                    # h < margin or L_f h + L_g h * u_raw + alpha * h < 0
                    active_margin = 0.1
                    constraint_values = h[b, i]
                    constraint_derivatives = L_f_h[b, i] + torch.bmm(L_g_h[b, i].unsqueeze(1), 
                                                                  raw_action[b, i].unsqueeze(-1)).squeeze(-1)
                    
                    active_constraints = (constraint_values < active_margin) | \
                                       (constraint_derivatives + self.alpha * constraint_values < 0)
                    
                    # Skip if no active constraints
                    if not torch.any(active_constraints):
                        continue
                    
                    # Extract active constraints
                    A = L_g_h[b, i][active_constraints]
                    b_qp = rhs[b, i][active_constraints]
                    
                    # Solve QP using a simple projection method
                    # This is a simplification; a full QP solver would be more robust
                    u = raw_action[b, i]
                    for _ in range(self.max_iterations):
                        # Check constraint violations
                        violations = torch.mm(A, u.unsqueeze(-1)).squeeze(-1) - b_qp
                        if torch.all(violations >= 0):
                            break
                            
                        # Compute projections for violated constraints
                        violated = violations < 0
                        if torch.any(violated):
                            # Update for each violated constraint
                            for j in torch.nonzero(violated):
                                # Compute projection
                                a_j = A[j]
                                b_j = b_qp[j]
                                
                                # Project onto constraint: u -= (a_j^T u - b_j) * a_j / ||a_j||^2
                                a_j_norm = torch.sum(a_j * a_j)
                                if a_j_norm > 1e-6:  # Avoid division by zero
                                    u = u - (torch.dot(a_j, u) - b_j) * a_j / a_j_norm
                    
                    safe_action[b, i] = u
        else:
            # Simpler safety filtering approach for each constraint
            # Instead of solving a QP, we just project the action if constraints are violated
            
            # Check which constraints are violated: L_f h + L_g h * u_raw + alpha * h < 0
            constraint_values = L_f_h + torch.matmul(L_g_h, raw_action.unsqueeze(-1)).squeeze(-1) + self.alpha * h
            violations = constraint_values < 0
            
            # Initialize safe action as raw action
            safe_action = raw_action.clone()
            
            # Process each batch and agent separately
            batch_size = state.batch_size
            n_agents = state.positions.shape[1]
            
            for b in range(batch_size):
                for i in range(n_agents):
                    # Skip if no violations for this agent
                    if not torch.any(violations[b, i]):
                        continue
                    
                    # For each violated constraint, project the action
                    for j in torch.nonzero(violations[b, i]):
                        # Skip if L_g_h is too small (constraint not controllable)
                        a_j = L_g_h[b, i, j]
                        a_j_norm = torch.sum(a_j * a_j)
                        if a_j_norm < 1e-6:
                            continue
                            
                        # Compute minimum value to satisfy constraint
                        b_j = rhs[b, i, j]
                        u = safe_action[b, i]
                        
                        # Project onto constraint: u -= (a_j^T u - b_j) * a_j / ||a_j||^2 if a_j^T u < b_j
                        if torch.dot(a_j, u) < b_j:
                            safe_action[b, i] = u - (torch.dot(a_j, u) - b_j) * a_j / a_j_norm
        
        return safe_action


class GCBFPlusAgent(nn.Module):
    """
    Agent that combines a policy network with a GCBF safety layer.
    
    This agent encapsulates both the policy network and safety layer,
    providing a unified interface for generating safe actions.
    """
    
    def __init__(self, policy_network: nn.Module, safety_layer: GCBFSafetyLayer, cbf_network: Optional[nn.Module] = None):
        """
        Initialize the GCBF+ agent.
        
        Args:
            policy_network: Neural network policy
            safety_layer: CBF safety layer for action filtering
            cbf_network: Optional neural network for learned barrier functions
        """
        super(GCBFPlusAgent, self).__init__()
        
        self.policy_network = policy_network
        self.safety_layer = safety_layer
        self.cbf_network = cbf_network
        
    def forward(self, state: MultiAgentState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate safe actions for the current state.
        
        Args:
            state: Current environment state
            
        Returns:
            Tuple of (safe_action, raw_action):
            - safe_action: Actions after safety filtering
            - raw_action: Raw actions from policy before filtering
        """
        # Get observations from state
        observations = self.get_observations(state)
        
        # Generate raw actions from policy
        raw_action = self.policy_network(observations)
        
        # Apply safety filtering
        safe_action = self.safety_layer(state, raw_action)
        
        return safe_action, raw_action
    
    def get_observations(self, state: MultiAgentState) -> torch.Tensor:
        """
        Extract observations from environment state.
        
        Args:
            state: Current environment state
            
        Returns:
            Observation tensor for the policy network
        """
        # Default implementation: concatenate positions, velocities, and goals
        # This can be overridden for more complex observation spaces
        observations = torch.cat([
            state.positions,
            state.velocities,
            state.goals
        ], dim=2)
        
        return observations 