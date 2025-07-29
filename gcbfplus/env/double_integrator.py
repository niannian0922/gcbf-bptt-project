import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .multi_agent_env import MultiAgentEnv, MultiAgentState, StepResult
from ..utils.autograd import g_decay, apply_temporal_decay

# Vision-related imports
from .vision_renderer import SimpleDepthRenderer, create_simple_renderer


@dataclass
class DoubleIntegratorState(MultiAgentState):
    """State representation for double integrator dynamics."""
    # Inherits all fields from MultiAgentState


class DoubleIntegratorEnv(MultiAgentEnv):
    """
    Differentiable environment for multi-agent double integrator dynamics.
    
    Each agent has a state [x, y, vx, vy] and control inputs [fx, fy].
    The dynamics follow the double integrator model: ẍ = f/m.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the double integrator environment.
        
        Args:
            config: Dictionary containing environment parameters.
                Required keys:
                - 'num_agents': Number of agents
                - 'area_size': Size of the square environment
                - 'dt': Simulation timestep
                - 'mass': Mass of each agent
                - 'car_radius': Radius of each agent (for collision detection)
                - 'comm_radius': Communication radius (for graph construction)
                Optional keys:
                - 'max_steps': Maximum episode length
                - 'max_force': Maximum force magnitude
                - 'gradient_decay_rate': Rate at which gradients decay through time
        """
        super(DoubleIntegratorEnv, self).__init__(config)
        
        # Store additional parameters
        self.mass = config.get('mass', 0.1)
        self.max_force = config.get('max_force', 1.0)
        self.cbf_alpha = config.get('cbf_alpha', 1.0)  # Default CBF alpha parameter
        self.pos_dim = 2  # 2D positions (x, y)
        self.vel_dim = 2  # 2D velocities (vx, vy)
        self.state_dim = 4  # x, y, vx, vy
        self.action_dim = 2  # fx, fy
        
        # Gradient decay parameters
        training_config = config.get('training', {})
        self.gradient_decay_rate = training_config.get('gradient_decay_rate', 0.95)
        self.use_gradient_decay = self.gradient_decay_rate > 0.0
        self.training = True  # Default to training mode
        
        # Vision-based observation parameters
        vision_config = config.get('vision', {})
        self.use_vision = vision_config.get('enabled', False)
        
        # Initialize renderer if vision is enabled
        if self.use_vision:
            renderer_config = {
                'image_size': vision_config.get('image_size', 64),
                'camera_fov': vision_config.get('camera_fov', 90.0),
                'camera_range': vision_config.get('camera_range', 3.0),
                'agent_radius': self.agent_radius,
                'obstacle_base_height': 0.5
            }
            self.depth_renderer = create_simple_renderer(renderer_config)
        
        # Register the state transition matrices as buffers
        # State transition: x_{t+1} = A * x_t + B * u_t
        A = torch.zeros(self.state_dim, self.state_dim)
        A[0, 0] = 1.0  # x position
        A[1, 1] = 1.0  # y position
        A[2, 2] = 1.0  # vx
        A[3, 3] = 1.0  # vy
        A[0, 2] = self.dt  # x += vx * dt
        A[1, 3] = self.dt  # y += vy * dt
        self.register_buffer('A', A)
        
        B = torch.zeros(self.state_dim, self.action_dim)
        B[2, 0] = self.dt / self.mass  # dvx = fx * dt / m
        B[3, 1] = self.dt / self.mass  # dvy = fy * dt / m
        self.register_buffer('B', B)

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Get observation shape: [n_agents, obs_dim] or [n_agents, channels, height, width] for vision"""
        if self.use_vision:
            # Vision-based: depth images
            image_size = self.depth_renderer.image_size
            return (self.num_agents, 1, image_size, image_size)  # Depth images
        else:
            # State-based observations
            if self.obstacles_config is not None:
                # Include position (2) and radius (1) of closest obstacle
                return (self.num_agents, self.state_dim + self.pos_dim + self.pos_dim + 1)
            else:
                return (self.num_agents, self.state_dim + self.pos_dim)  # state + goal position

    @property
    def action_shape(self) -> Tuple[int, ...]:
        """Get action shape: [n_agents, action_dim]"""
        return (self.num_agents, self.action_dim)
    
    def get_action_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get bounds of valid actions."""
        lower_bound = -self.max_force * torch.ones(self.action_shape, device=self.device)
        upper_bound = self.max_force * torch.ones(self.action_shape, device=self.device)
        return lower_bound, upper_bound
    
    def render_depth_image(self, state: DoubleIntegratorState, agent_idx: int) -> torch.Tensor:
        """
        Render a depth image from the perspective of a specific agent using the simplified renderer.
        
        Args:
            state: Current environment state
            agent_idx: Index of the agent whose perspective to render from
            
        Returns:
            Depth image tensor [1, H, W] with values in [0, 1]
        """
        if not self.use_vision:
            raise RuntimeError("Vision rendering is not enabled")
        
        # Get agent position and goal
        agent_pos = state.positions[0, agent_idx]  # [2]
        goal_pos = state.goals[0, agent_idx]       # [2]
        
        # Get other agent positions (excluding the viewing agent)
        all_agent_positions = state.positions[0]  # [num_agents, 2]
        other_agents = torch.cat([
            all_agent_positions[:agent_idx],
            all_agent_positions[agent_idx+1:]
        ], dim=0) if len(all_agent_positions) > 1 else torch.empty(0, 2, device=agent_pos.device)
        
        # Render depth image using the simplified renderer
        depth_image = self.depth_renderer.render_depth_from_position(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            other_agents=other_agents,
            obstacles=state.obstacles[0] if state.obstacles is not None else None
        )
        
        # Add some realism with noise (optional)
        depth_image = self.depth_renderer.add_noise_and_realism(depth_image, noise_level=0.01)
        
        return depth_image  # [1, H, W]
    
    def reset(self, batch_size: int = 1, randomize: bool = True) -> DoubleIntegratorState:
        """
        Reset the environment to an initial state.
        
        Args:
            batch_size: Number of parallel environments
            randomize: Whether to randomize initial positions and goals
            
        Returns:
            Initial environment state
        """
        # Create a new random generator on the correct device
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        # Initialize positions randomly within the environment bounds
        if randomize:
            # Random positions within the environment bounds
            positions = torch.rand(batch_size, self.num_agents, self.pos_dim, device=device) * self.area_size
            
            # Random goals within the environment bounds
            goals = torch.rand(batch_size, self.num_agents, self.pos_dim, device=device) * self.area_size
            
            # Ensure goals are not too close to initial positions
            min_goal_dist = 0.3 * self.area_size
            
            # Compute distances between positions and goals
            dist = torch.norm(positions - goals, dim=2)
            
            # Reposition goals that are too close
            too_close = dist < min_goal_dist
            while torch.any(too_close):
                # Replace goals that are too close
                new_goals = torch.rand(batch_size, self.num_agents, self.pos_dim, device=device) * self.area_size
                goals = torch.where(too_close.unsqueeze(-1), new_goals, goals)
                
                # Recompute distances and check again
                dist = torch.norm(positions - goals, dim=2)
                too_close = dist < min_goal_dist
        else:
            # Default initialization: agents in a grid, goals opposite side
            positions = torch.zeros(batch_size, self.num_agents, self.pos_dim, device=device)
            goals = torch.zeros(batch_size, self.num_agents, self.pos_dim, device=device)
            
            # Create a grid arrangement for agents
            grid_size = int(np.ceil(np.sqrt(self.num_agents)))
            spacing = self.area_size / (grid_size + 1)
            
            for i in range(self.num_agents):
                row, col = i // grid_size, i % grid_size
                positions[:, i, 0] = spacing * (col + 1)
                positions[:, i, 1] = spacing * (row + 1)
                
                # Goals on opposite side
                goals[:, i, 0] = self.area_size - positions[0, i, 0]
                goals[:, i, 1] = self.area_size - positions[0, i, 1]
        
        # Initialize velocities as zeros
        velocities = torch.zeros(batch_size, self.num_agents, self.vel_dim, device=device)
        
        # Get obstacles tensor
        obstacles = self.get_obstacle_tensor(batch_size)
        
        # If obstacles exist, ensure agents and goals are not initialized inside obstacles
        if obstacles is not None:
            # Extract obstacle positions and radii
            obstacle_positions = obstacles[..., :-1]  # [batch_size, n_obstacles, pos_dim]
            obstacle_radii = obstacles[..., -1:]     # [batch_size, n_obstacles, 1]
            
            # Ensure agents are not inside obstacles
            for b in range(batch_size):
                for i in range(self.num_agents):
                    # Check distance to all obstacles
                    agent_pos = positions[b, i].unsqueeze(0)  # [1, pos_dim]
                    
                    # Compute distances
                    dists = torch.norm(agent_pos - obstacle_positions[b], dim=1)
                    min_dists = dists - obstacle_radii[b].squeeze(-1) - self.agent_radius
                    
                    # If agent is inside any obstacle, move it
                    if torch.any(min_dists < 0):
                        # Try several random positions until we find one that's valid
                        max_tries = 50
                        for _ in range(max_tries):
                            new_pos = torch.rand(self.pos_dim, device=device) * self.area_size
                            new_dists = torch.norm(new_pos - obstacle_positions[b], dim=1)
                            new_min_dists = new_dists - obstacle_radii[b].squeeze(-1) - self.agent_radius
                            
                            if torch.all(new_min_dists >= 0):
                                positions[b, i] = new_pos
                                break
            
            # Ensure goals are not inside obstacles
            for b in range(batch_size):
                for i in range(self.num_agents):
                    # Check distance to all obstacles
                    goal_pos = goals[b, i].unsqueeze(0)  # [1, pos_dim]
                    
                    # Compute distances
                    dists = torch.norm(goal_pos - obstacle_positions[b], dim=1)
                    min_dists = dists - obstacle_radii[b].squeeze(-1) - self.agent_radius
                    
                    # If goal is inside any obstacle, move it
                    if torch.any(min_dists < 0):
                        # Try several random positions until we find one that's valid
                        max_tries = 50
                        for _ in range(max_tries):
                            new_pos = torch.rand(self.pos_dim, device=device) * self.area_size
                            new_dists = torch.norm(new_pos - obstacle_positions[b], dim=1)
                            new_min_dists = new_dists - obstacle_radii[b].squeeze(-1) - self.agent_radius
                            
                            if torch.all(new_min_dists >= 0):
                                goals[b, i] = new_pos
                                break
        
        # Create state object
        state = DoubleIntegratorState(
            positions=positions,
            velocities=velocities,
            goals=goals,
            obstacles=obstacles,  # Include obstacles in state
            batch_size=batch_size,
            step_count=0
        )
        
        return state
    
    def step(self, state: DoubleIntegratorState, action: torch.Tensor, alpha: Optional[torch.Tensor] = None) -> StepResult:
        """
        Take a step in the environment using the given actions and dynamic alpha values.
        
        Args:
            state: Current state
            action: Actions to take [batch_size, n_agents, action_dim]
            alpha: Dynamic CBF alpha values [batch_size, n_agents, 1] (optional)
            
        Returns:
            StepResult containing next_state, reward, cost, done, info
        """
        # Ensure action is on the correct device first
        action = action.to(self.device)
        
        # Ensure alpha is on the correct device if provided
        if alpha is not None:
            alpha = alpha.to(self.device)
        
        # Apply safety layer with dynamic alpha if it exists
        safe_action = self.apply_safety_layer(state, action, alpha)
        
        # Ensure safe_action is also on the correct device
        safe_action = safe_action.to(self.device)
        
        # Clip action to bounds
        # Ensure bounds are on the same device as the state and actions
        lower_bound = torch.tensor([-1.0, -1.0], device=self.device, dtype=safe_action.dtype)
        upper_bound = torch.tensor([1.0, 1.0], device=self.device, dtype=safe_action.dtype)
        safe_action = torch.clamp(safe_action, lower_bound, upper_bound)
        
        # Compute state derivatives using dynamics
        derivatives = self.dynamics(state, safe_action)
        
        # Update positions and velocities using Euler integration with temporal gradient decay
        if self.use_gradient_decay and self.training:
            # Apply temporal gradient decay to stabilize long-horizon BPTT training
            positions_decayed = apply_temporal_decay(
                state.positions, self.gradient_decay_rate, self.dt, self.training
            )
            velocities_decayed = apply_temporal_decay(
                state.velocities, self.gradient_decay_rate, self.dt, self.training
            )
            
            new_positions = positions_decayed + velocities_decayed * self.dt
            new_velocities = velocities_decayed + (safe_action / self.mass) * self.dt
        else:
            # Standard update without gradient decay
            new_positions = state.positions + state.velocities * self.dt
            new_velocities = state.velocities + (safe_action / self.mass) * self.dt
        
        # Create next state
        next_state = DoubleIntegratorState(
            positions=new_positions,
            velocities=new_velocities,
            goals=state.goals,
            obstacles=state.obstacles,  # Preserve obstacles in next state
            batch_size=state.batch_size,
            step_count=state.step_count + 1
        )
        
        # Compute reward, cost, and done
        reward = self.compute_reward(state, safe_action, next_state)
        cost = self.compute_cost(state, safe_action, next_state)
        done = self.is_done(next_state)
        
        # Prepare info dict
        info = {
            'goal_distance': self.get_goal_distance(next_state),
            'collision': self.check_collision(next_state),
            'action': safe_action,
            'raw_action': action,
            'alpha': alpha if alpha is not None else torch.ones_like(action[..., :1]) * self.cbf_alpha  # Default alpha
        }
        
        return StepResult(next_state, reward, cost, done, info)
    
    def get_observation(self, state: DoubleIntegratorState) -> torch.Tensor:
        """
        Extract observation from environment state.
        
        Returns either state-based or vision-based observations depending on configuration.
        
        State-based observations include:
        - Agent position (x, y), velocity (vx, vy), goal position (gx, gy)
        - Obstacle information (if present)
        
        Vision-based observations include:
        - Depth image from each agent's perspective [n_agents, 1, H, W]
        
        Args:
            state: Current environment state
            
        Returns:
            Observation tensor: 
            - State-based: [batch_size, n_agents, obs_dim]
            - Vision-based: [batch_size, n_agents, 1, H, W]
        """
        batch_size = state.batch_size
        device = state.positions.device
        
        if self.use_vision:
            # Vision-based observations: render depth images for each agent
            vision_obs = []
            
            for agent_idx in range(self.num_agents):
                depth_image = self.render_depth_image(state, agent_idx)  # [1, H, W]
                vision_obs.append(depth_image)
            
            # Stack depth images for all agents
            stacked_obs = torch.stack(vision_obs, dim=0)  # [n_agents, 1, H, W]
            
            # Add batch dimension
            observation = stacked_obs.unsqueeze(0)  # [1, n_agents, 1, H, W]
            
            return observation
        
        else:
            # State-based observations (original implementation)
            # First combine position, velocity and goal information
            base_obs = torch.cat([
                state.positions,                              # positions
                state.velocities,                             # velocities
                state.goals                                   # goals
            ], dim=2)
        
        # If we have obstacles, include information about closest obstacle to each agent
        if state.obstacles is not None:
            # Extract obstacle positions and radii
            obstacle_positions = state.obstacles[..., :-1]  # [batch_size, n_obstacles, pos_dim]
            obstacle_radii = state.obstacles[..., -1:]     # [batch_size, n_obstacles, 1]
            
            # For each agent, find the closest obstacle and include it in observation
            closest_obstacles = torch.zeros(batch_size, self.num_agents, self.pos_dim + 1, device=device)
            
            for b in range(batch_size):
                for i in range(self.num_agents):
                    # Compute distances to all obstacles
                    agent_pos = state.positions[b, i].unsqueeze(0)  # [1, pos_dim]
                    dists = torch.norm(agent_pos - obstacle_positions[b], dim=1)
                    
                    # Find the closest obstacle
                    closest_idx = torch.argmin(dists)
                    
                    # Store position and radius of closest obstacle relative to agent
                    closest_pos = obstacle_positions[b, closest_idx] - state.positions[b, i]
                    closest_rad = obstacle_radii[b, closest_idx]
                    
                    closest_obstacles[b, i, :self.pos_dim] = closest_pos
                    closest_obstacles[b, i, self.pos_dim] = closest_rad
            
            # Combine with base observation
            observation = torch.cat([base_obs, closest_obstacles], dim=2)
        else:
            observation = base_obs
        
        return observation
    
    def render(self, state: DoubleIntegratorState) -> Any:
        """
        Render the environment state. This is a placeholder implementation.
        
        Args:
            state: Environment state to render
            
        Returns:
            Rendering of the environment (matplotlib figure)
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Only render the first batch
        batch_idx = 0
        
        # Draw agents as circles
        for i in range(self.num_agents):
            agent_pos = state.positions[batch_idx, i].detach().cpu().numpy()
            agent = Circle(agent_pos, self.agent_radius, color='blue', alpha=0.7)
            ax.add_patch(agent)
            
            # Draw velocity vector
            vel = state.velocities[batch_idx, i].detach().cpu().numpy()
            ax.arrow(agent_pos[0], agent_pos[1], vel[0]*0.1, vel[1]*0.1, 
                     head_width=0.02, head_length=0.03, fc='black', ec='black')
            
            # Draw goal
            goal_pos = state.goals[batch_idx, i].detach().cpu().numpy()
            ax.scatter(goal_pos[0], goal_pos[1], color='green', marker='*', s=100)
        
        # Draw obstacles if present
        if state.obstacles is not None:
            obstacles = state.obstacles[batch_idx].detach().cpu().numpy()
            for obs in obstacles:
                obs_pos = obs[:-1]
                obs_radius = obs[-1]
                obstacle = Circle(obs_pos, obs_radius, color='red', alpha=0.3)
                ax.add_patch(obstacle)
        
        # Set limits and labels
        ax.set_xlim(0, self.area_size)
        ax.set_ylim(0, self.area_size)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Double Integrator Environment')
        
        plt.tight_layout()
        
        return fig
    
    def apply_safety_layer(self, state: DoubleIntegratorState, raw_action: torch.Tensor, alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply safety constraints to raw actions using dynamic alpha values.
        
        Args:
            state: Current environment state
            raw_action: Raw action from policy [batch_size, n_agents, action_dim]
            alpha: Dynamic CBF alpha values [batch_size, n_agents, 1] (optional)
            
        Returns:
            Safe action [batch_size, n_agents, action_dim]
        """
        # Store alpha values for logging and potential future use
        if alpha is not None:
            self._current_alpha = alpha
        
        # If a safety layer is available, use it with dynamic alphas
        if hasattr(self, 'safety_layer') and self.safety_layer is not None:
            try:
                # Call safety layer with dynamic alphas
                safe_action = self.safety_layer.forward(state, raw_action, alphas=alpha)
                return safe_action
            except Exception as e:
                # If safety layer fails, fall back to raw action
                # This ensures robustness during training
                print(f"Warning: Safety layer failed: {e}. Using raw action.")
                return raw_action
        else:
            # Default implementation: return raw action if no safety layer
        return raw_action

    def dynamics(self, state: DoubleIntegratorState, action: torch.Tensor) -> torch.Tensor:
        """
        Apply double integrator dynamics to compute state derivatives.
        
        Args:
            state: Current state
            action: Control inputs [batch_size, n_agents, 2]
            
        Returns:
            State derivatives [batch_size, n_agents, 4]
        """
        # Extract batch size for convenience
        batch_size = state.batch_size
        
        # Reshape state and action for matrix multiplication
        # Combine batch and agent dimensions: [batch_size * n_agents, state_dim/action_dim]
        states_flat = torch.cat([
            state.positions.reshape(batch_size * self.num_agents, self.pos_dim),
            state.velocities.reshape(batch_size * self.num_agents, self.vel_dim)
        ], dim=1)
        actions_flat = action.reshape(batch_size * self.num_agents, self.action_dim)
        
        # Apply dynamics: x_dot = Ax + Bu
        # We don't use the full state transition here, just the derivatives
        acceleration = actions_flat / self.mass
        
        # Create derivatives
        derivatives = torch.cat([
            state.velocities.reshape(batch_size * self.num_agents, self.vel_dim),
            acceleration
        ], dim=1)
        
        # Reshape back to [batch_size, n_agents, state_dim]
        derivatives = derivatives.reshape(batch_size, self.num_agents, self.state_dim)
        
        return derivatives

    def train(self) -> None:
        """Set environment to training mode. This enables gradient decay."""
        self.training = True
    
    def eval(self) -> None:
        """Set environment to evaluation mode. This disables gradient decay."""
        self.training = False
