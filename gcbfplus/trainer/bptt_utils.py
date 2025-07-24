import torch
import math
from typing import Tuple, Dict, Any


def initialize_states_and_goals(
    num_agents: int, 
    state_dim: int, 
    area_size: float, 
    min_dist: float = 0.2, 
    max_travel: float = None,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initializes random start states and non-colliding goal positions for agents.
    
    Args:
        num_agents: Number of agents to initialize
        state_dim: Dimension of agent state (typically 4 for [x, y, vx, vy])
        area_size: Size of the square working area
        min_dist: Minimum distance between agents and between goals
        max_travel: Maximum allowed distance from agent to its goal (if None, no restriction)
        device: PyTorch device to create tensors on
    
    Returns:
        states: Tensor of shape [num_agents, state_dim]. Velocities are zero initially.
        goals: Tensor of shape [num_agents, state_dim//2] for (x, y) goal positions.
    """
    # Initialize empty tensors
    states = torch.zeros((num_agents, state_dim), device=device)
    pos_dim = state_dim // 2  # Position dimensions (x, y)
    goals = torch.zeros((num_agents, pos_dim), device=device)
    
    # Place agents one by one, ensuring minimum distance
    for i in range(num_agents):
        # Try to place agent until a valid position is found
        max_attempts = 100
        for _ in range(max_attempts):
            # Generate random position
            pos = torch.rand(pos_dim, device=device) * area_size
            
            # Check distance from previously placed agents
            valid = True
            if i > 0:
                prev_pos = states[:i, :pos_dim]
                distances = torch.norm(prev_pos - pos.unsqueeze(0), dim=1)
                if torch.min(distances) < min_dist:
                    valid = False
                    continue
            
            # If valid, set the agent's position
            if valid:
                states[i, :pos_dim] = pos
                # Initialize with zero velocity
                states[i, pos_dim:] = 0.0
                break
        
        # If no valid position found after max attempts, raise error
        if not valid:
            raise RuntimeError(f"Could not place agent {i} after {max_attempts} attempts. Try reducing min_dist or increasing area_size.")
    
    # Place goals one by one, ensuring minimum distance
    for i in range(num_agents):
        max_attempts = 100
        for _ in range(max_attempts):
            # Generate random goal position
            if max_travel is None:
                # Random position in the area
                goal = torch.rand(pos_dim, device=device) * area_size
            else:
                # Random position within max_travel from the agent
                angle = torch.rand(1, device=device) * 2 * math.pi
                distance = torch.rand(1, device=device) * max_travel
                goal = states[i, :pos_dim] + torch.tensor([
                    torch.cos(angle) * distance,
                    torch.sin(angle) * distance
                ], device=device)
                # Ensure goal is within bounds
                goal = torch.clamp(goal, min=0, max=area_size)
            
            # Check distance from agent position (should be far enough)
            agent_pos = states[i, :pos_dim]
            agent_to_goal_dist = torch.norm(agent_pos - goal)
            if agent_to_goal_dist < min_dist:
                continue
            
            # Check distance from other goals
            valid = True
            if i > 0:
                prev_goals = goals[:i]
                goal_distances = torch.norm(prev_goals - goal.unsqueeze(0), dim=1)
                if torch.min(goal_distances) < min_dist:
                    valid = False
                    continue
            
            # If valid, set the goal
            if valid:
                goals[i] = goal
                break
        
        # If no valid position found after max attempts, raise error
        if not valid:
            raise RuntimeError(f"Could not place goal {i} after {max_attempts} attempts. Try reducing min_dist or increasing area_size.")
    
    return states, goals


def build_graph_features(
    states: torch.Tensor,
    goals: torch.Tensor,
    sensing_radius: float,
    edge_feature_radius: float = None,
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """
    Builds graph features required by the GNNs from the current system state.
    
    Args:
        states: Current states of all agents [N, state_dim]
        goals: Goal positions for all agents [N, pos_dim]
        sensing_radius: The radius for determining neighbors
        edge_feature_radius: The radius for normalizing edge features (if None, use sensing_radius)
        device: PyTorch device for computations
    
    Returns:
        Dictionary containing:
            node_features: Features for each node [N, node_feat_dim]
            edge_index: Index pairs for edges [2, E]
            edge_features: Features for each edge [E, edge_feat_dim]
    """
    if device is None and states.device.type != 'cpu':
        device = states.device
    
    if edge_feature_radius is None:
        edge_feature_radius = sensing_radius
    
    num_agents = states.shape[0]
    state_dim = states.shape[1]
    pos_dim = goals.shape[1]  # Typically 2 for (x, y)
    
    # Extract position information
    positions = states[:, :pos_dim]  # [N, pos_dim]
    
    # Compute pairwise distances between agents
    # Using cdist for efficient distance computation
    distances = torch.cdist(positions, positions)  # [N, N]
    
    # Create adjacency matrix based on sensing radius
    # Add self-loops (diagonal = 1)
    adjacency = (distances < sensing_radius).float()
    eye_mask = torch.eye(num_agents, device=device)
    adjacency = adjacency + eye_mask
    adjacency = torch.clamp(adjacency, 0, 1)  # Ensure values are 0 or 1
    
    # Convert adjacency to edge_index format [2, num_edges]
    # This contains the indices of connected nodes
    edge_index = torch.nonzero(adjacency).t()  # [2, num_edges]
    
    # Compute edge features (relative state differences)
    senders = edge_index[0]
    receivers = edge_index[1]
    
    # Compute relative states for edges: receiver_state - sender_state
    edge_features = states[receivers] - states[senders]  # [num_edges, state_dim]
    
    # For graph features that are too long, clip them to maintain locality
    pos_edge_features = edge_features[:, :pos_dim]  # Position differences [num_edges, pos_dim]
    pos_norms = torch.norm(pos_edge_features, dim=1, keepdim=True)  # [num_edges, 1]
    
    # Create scaling factors for position differences that exceed edge_feature_radius
    scaling = torch.ones_like(pos_norms)
    mask = pos_norms > edge_feature_radius
    scaling[mask] = edge_feature_radius / pos_norms[mask]
    
    # Apply scaling to position differences
    scaled_pos_diff = pos_edge_features * scaling
    
    # Combine scaled position differences with velocity differences
    edge_features = torch.cat([
        scaled_pos_diff,  # Scaled position differences [num_edges, pos_dim]
        edge_features[:, pos_dim:],  # Velocity differences [num_edges, state_dim - pos_dim]
    ], dim=1)
    
    # Create node features
    # 1. Extract goal direction for each agent and scale if necessary
    goal_directions = goals - positions  # [N, pos_dim]
    goal_norms = torch.norm(goal_directions, dim=1, keepdim=True)  # [N, 1]
    
    # Scale goal directions that exceed edge_feature_radius
    goal_scaling = torch.ones_like(goal_norms)
    mask = goal_norms > edge_feature_radius
    goal_scaling[mask] = edge_feature_radius / goal_norms[mask]
    scaled_goal_directions = goal_directions * goal_scaling  # [N, pos_dim]
    
    # 2. Add node type indicator (in this case, all are agents)
    agent_indicators = torch.ones(num_agents, 1, device=device)  # [N, 1]
    
    # Combine all node features:
    # - Agent state [x, y, vx, vy]
    # - Goal direction (scaled) [gx, gy]
    # - Agent indicator [1]
    node_features = torch.cat([
        states,  # Agent state [N, state_dim]
        scaled_goal_directions,  # Goal direction [N, pos_dim]
        agent_indicators,  # Agent indicator [N, 1]
    ], dim=1)
    
    return {
        'node_features': node_features,      # [N, node_feat_dim]
        'edge_index': edge_index,            # [2, num_edges]
        'edge_features': edge_features,      # [num_edges, edge_feat_dim]
    }


def calculate_cbf_derivative(
    cbf_values: torch.Tensor, 
    next_cbf_values: torch.Tensor, 
    dt: float
) -> torch.Tensor:
    """
    Calculate the time derivative of the CBF using finite differences.
    
    Args:
        cbf_values: Current CBF values [N, 1]
        next_cbf_values: Next step CBF values [N, 1]
        dt: Time step
    
    Returns:
        CBF derivative approximation [N, 1]
    """
    return (next_cbf_values - cbf_values) / dt


def check_cbf_condition(
    cbf_values: torch.Tensor, 
    cbf_derivatives: torch.Tensor, 
    alpha: float
) -> torch.Tensor:
    """
    Check if the CBF condition h_dot + alpha * h >= 0 is satisfied.
    
    Args:
        cbf_values: Current CBF values [N, 1]
        cbf_derivatives: Time derivatives of CBF [N, 1]
        alpha: Positive constant in the CBF condition
    
    Returns:
        Violation measure, should be zero if condition is satisfied [N, 1]
    """
    # CBF condition: h_dot + alpha * h >= 0
    # If violated, return positive violation amount
    condition = cbf_derivatives + alpha * cbf_values
    return torch.relu(-condition)  # ReLU: Return violation only if condition < 0 