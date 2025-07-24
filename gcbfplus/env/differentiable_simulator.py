import torch
import torch.nn as nn
import numpy as np


class DifferentiableDoubleIntegrator(nn.Module):
    """
    A differentiable implementation of the double integrator dynamics for multi-agent systems.
    This simulator is designed to be part of the PyTorch computation graph to enable
    backpropagation through time (BPTT) for end-to-end training.
    
    The state of each agent is [x, y, vx, vy] and the control input is [fx, fy].
    """
    
    def __init__(self, dt=0.03, mass=0.1):
        super(DifferentiableDoubleIntegrator, self).__init__()
        self.dt = dt
        self.mass = mass
        
        # Define state transition matrices as PyTorch tensors
        # State transition: x_{t+1} = A * x_t + B * u_t
        A = torch.zeros(4, 4)
        A[0, 0] = 1.0  # x position
        A[1, 1] = 1.0  # y position
        A[2, 2] = 1.0  # vx
        A[3, 3] = 1.0  # vy
        A[0, 2] = dt   # x += vx * dt
        A[1, 3] = dt   # y += vy * dt
        self.register_buffer('A', A)
        
        B = torch.zeros(4, 2)
        B[2, 0] = dt / mass  # dvx = fx * dt / m
        B[3, 1] = dt / mass  # dvy = fy * dt / m
        self.register_buffer('B', B)
    
    def forward(self, states, actions):
        """
        Compute the next states given current states and actions.
        
        Args:
            states: Tensor of shape [num_agents, 4] representing the current state
                   of all agents. Each state is [x, y, vx, vy].
            actions: Tensor of shape [num_agents, 2] representing the control inputs.
                    Each action is [fx, fy].
        
        Returns:
            Tensor of shape [num_agents, 4] representing the next state.
        """
        # Apply dynamics: x_{t+1} = A * x_t + B * u_t
        next_states = torch.matmul(states, self.A.T) + torch.matmul(actions, self.B.T)
        
        return next_states
    
    def step_exact(self, states, actions):
        """
        Alternative implementation using exact integration for the double integrator.
        
        Args:
            states: Tensor of shape [num_agents, 4] representing the current state
                   of all agents. Each state is [x, y, vx, vy].
            actions: Tensor of shape [num_agents, 2] representing the control inputs.
                    Each action is [fx, fy].
        
        Returns:
            Tensor of shape [num_agents, 4] representing the next state.
        """
        # Extract positions and velocities
        positions = states[:, :2]  # [x, y]
        velocities = states[:, 2:] # [vx, vy]
        
        # Compute acceleration
        accelerations = actions / self.mass
        
        # Exact integration for double integrator
        # p_{t+1} = p_t + v_t * dt + 0.5 * a_t * dt^2
        # v_{t+1} = v_t + a_t * dt
        new_positions = positions + velocities * self.dt + 0.5 * accelerations * self.dt**2
        new_velocities = velocities + accelerations * self.dt
        
        # Concatenate to form new state
        next_states = torch.cat([new_positions, new_velocities], dim=1)
        
        return next_states


class DifferentiableObstacleCollision(nn.Module):
    """
    A differentiable module to detect and compute gradients for collisions between
    agents and circular obstacles.
    """
    
    def __init__(self, agent_radius=0.05):
        super(DifferentiableObstacleCollision, self).__init__()
        self.agent_radius = agent_radius
    
    def forward(self, agent_positions, obstacle_positions, obstacle_radii):
        """
        Compute a differentiable measure of collision between agents and obstacles.
        
        Args:
            agent_positions: Tensor of shape [num_agents, 2] representing agent positions
            obstacle_positions: Tensor of shape [num_obstacles, 2] representing obstacle positions
            obstacle_radii: Tensor of shape [num_obstacles] representing obstacle radii
        
        Returns:
            A tensor of shape [num_agents] where positive values indicate collision penetration
        """
        # Compute distances between all agents and obstacles
        # Shape: [num_agents, num_obstacles, 2]
        relative_positions = agent_positions.unsqueeze(1) - obstacle_positions.unsqueeze(0)
        
        # Compute squared distances
        # Shape: [num_agents, num_obstacles]
        distances_squared = torch.sum(relative_positions**2, dim=2)
        distances = torch.sqrt(distances_squared + 1e-8)  # Add small epsilon for numerical stability
        
        # Compute penetration depth (positive if collision)
        # Shape: [num_agents, num_obstacles]
        penetration = self.agent_radius + obstacle_radii.unsqueeze(0) - distances
        
        # Use softmax to get a differentiable maximum penetration per agent
        # This ensures the strongest collision gets the gradient
        temperature = 10.0  # Higher value for sharper softmax
        softmax_weights = torch.softmax(temperature * penetration, dim=1)
        weighted_penetration = softmax_weights * penetration
        
        # Sum the weighted penetrations for each agent
        collision_measure = torch.sum(weighted_penetration, dim=1)
        
        return collision_measure


class DifferentiableAgentCollision(nn.Module):
    """
    A differentiable module to detect and compute gradients for collisions between agents.
    """
    
    def __init__(self, agent_radius=0.05):
        super(DifferentiableAgentCollision, self).__init__()
        self.agent_radius = agent_radius
        self.min_safe_distance = 2 * agent_radius
    
    def forward(self, agent_positions):
        """
        Compute a differentiable measure of collision between agents.
        
        Args:
            agent_positions: Tensor of shape [num_agents, 2] representing agent positions
        
        Returns:
            A tensor of shape [num_agents] representing collision measures per agent
        """
        num_agents = agent_positions.shape[0]
        
        # Compute distances between all agents
        # Shape: [num_agents, num_agents, 2]
        relative_positions = agent_positions.unsqueeze(1) - agent_positions.unsqueeze(0)
        
        # Compute distances
        # Shape: [num_agents, num_agents]
        distances_squared = torch.sum(relative_positions**2, dim=2)
        distances = torch.sqrt(distances_squared + 1e-8)  # Add small epsilon for numerical stability
        
        # Add large values to diagonal to ignore self-collisions
        eye_mask = torch.eye(num_agents, device=agent_positions.device) * 1e6
        distances = distances + eye_mask
        
        # Compute penetration depth (positive if collision)
        # Shape: [num_agents, num_agents]
        penetration = self.min_safe_distance - distances
        
        # Use ReLU to zero out non-collisions and softmax to focus on worst collision
        penetration = torch.relu(penetration)
        
        # Use softmax to get a differentiable maximum penetration per agent
        temperature = 10.0  # Higher value for sharper softmax
        softmax_weights = torch.softmax(temperature * penetration, dim=1)
        weighted_penetration = softmax_weights * penetration
        
        # Sum the weighted penetrations for each agent
        collision_measure = torch.sum(weighted_penetration, dim=1)
        
        return collision_measure


class DifferentiableGraphBuilder(nn.Module):
    """
    A differentiable module that builds graph structures from agent positions
    to use with GNN-based policies and CBFs.
    """
    
    def __init__(self, comm_radius=0.5):
        super(DifferentiableGraphBuilder, self).__init__()
        self.comm_radius = comm_radius
    
    def forward(self, agent_states, goals):
        """
        Build a graph structure for GNN processing.
        
        Args:
            agent_states: Tensor of shape [num_agents, 4] representing agent states [x, y, vx, vy]
            goals: Tensor of shape [num_agents, 2] representing goal positions [x, y]
        
        Returns:
            Dictionary containing:
                - node_features: Tensor of shape [num_agents, node_feat_dim]
                - edge_index: Tensor of shape [2, num_edges] representing connectivity
                - edge_features: Tensor of shape [num_edges, edge_feat_dim]
        """
        num_agents = agent_states.shape[0]
        agent_positions = agent_states[:, :2]
        
        # Compute pairwise distances between agents
        # Shape: [num_agents, num_agents]
        pos_diff = agent_positions.unsqueeze(1) - agent_positions.unsqueeze(0)
        distances = torch.norm(pos_diff, dim=2)
        
        # Create adjacency matrix based on communication radius
        # We add self-loops (diagonal=1)
        adjacency = (distances < self.comm_radius).float()
        eye_mask = torch.eye(num_agents, device=agent_states.device)
        adjacency = adjacency + eye_mask
        adjacency = torch.clamp(adjacency, 0, 1)  # Ensure values are 0 or 1
        
        # Convert adjacency matrix to edge_index format
        # This is a differentiable operation as we're just selecting indices
        edges = torch.nonzero(adjacency).T  # Shape [2, num_edges]
        
        # Compute edge features (relative state differences)
        senders = edges[0]
        receivers = edges[1]
        
        # Compute relative states for edges
        # For each edge (i,j), we compute state_j - state_i
        edge_features = agent_states[receivers] - agent_states[senders]
        
        # For graph features that are too long, clip them to maintain locality
        pos_edge_features = edge_features[:, :2]  # Position differences
        pos_norms = torch.norm(pos_edge_features, dim=1, keepdim=True)
        safe_norms = torch.clamp(pos_norms, min=self.comm_radius)
        
        # Create scaling factors for position differences (only scale those beyond comm_radius)
        scaling = torch.where(
            pos_norms > self.comm_radius,
            self.comm_radius / safe_norms,
            torch.ones_like(pos_norms)
        )
        
        # Apply scaling to position differences
        edge_features = torch.cat([
            pos_edge_features * scaling,  # Scaled position differences
            edge_features[:, 2:],  # Original velocity differences
        ], dim=1)
        
        # Compute agent-to-goal features (these are node features)
        goal_features = goals - agent_positions
        goal_norms = torch.norm(goal_features, dim=1, keepdim=True)
        safe_goal_norms = torch.clamp(goal_norms, min=self.comm_radius)
        
        goal_scaling = torch.where(
            goal_norms > self.comm_radius,
            self.comm_radius / safe_goal_norms,
            torch.ones_like(goal_norms)
        )
        
        # Apply scaling to goal differences
        goal_features = goal_features * goal_scaling
        
        # Create indicator features for nodes (one-hot encoding for agent type)
        # In the original code, this was a one-hot for agent/goal/obstacle
        # Here we just have agent indicators (always 1) as goals are handled separately
        agent_indicators = torch.ones(num_agents, 1, device=agent_states.device)
        
        # Combine state and indicator features for nodes
        node_features = torch.cat([
            agent_states,       # Agent state [x, y, vx, vy]
            goal_features,      # Goal direction (scaled) [gx, gy]
            agent_indicators,   # Agent type indicator
        ], dim=1)
        
        return {
            'node_features': node_features,      # Shape [num_agents, node_feat_dim]
            'edge_index': edges,                 # Shape [2, num_edges]
            'edge_features': edge_features,      # Shape [num_edges, edge_feat_dim]
        } 