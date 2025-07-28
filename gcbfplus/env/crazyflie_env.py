import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass, field

from .multi_agent_env import MultiAgentEnv, MultiAgentState, StepResult


@dataclass
class CrazyFlieState(MultiAgentState):
    """State representation for CrazyFlie quadrotor dynamics."""
    # Full state: [x, y, z, psi, theta, phi, u, v, w, r, q, p]
    # We need to define only additional fields here
    # positions, velocities, goals are inherited and required
    # orientations, obstacles, batch_size, step_count are inherited with defaults
    orientations: torch.Tensor = field(default=None)           # Override to make required
    angular_velocities: torch.Tensor = field(default=None)     # Required field
    
    @property
    def state_tensor(self) -> torch.Tensor:
        """Create a combined state tensor for all agents."""
        return torch.cat([
            self.positions,          # x, y, z
            self.orientations,       # psi, theta, phi
            self.velocities,         # u, v, w
            self.angular_velocities  # r, q, p
        ], dim=2)


class CrazyFlieEnv(MultiAgentEnv):
    """
    Differentiable environment for CrazyFlie quadrotor dynamics.
    
    Each agent has a state [x, y, z, psi, theta, phi, u, v, w, r, q, p]
    and control inputs [f_1, f_2, f_3, f_4] (motor thrusts).
    """
    
    # State indices for easy reference
    X, Y, Z, PSI, THETA, PHI, U, V, W, R, Q, P = range(12)
    
    # Control indices
    F_1, F_2, F_3, F_4 = range(4)
    
    def __init__(self, config: Dict):
        """
        Initialize the CrazyFlie quadrotor environment.
        
        Args:
            config: Dictionary containing environment parameters
                Required keys:
                - 'num_agents': Number of agents
                - 'area_size': Size of the environment
                - 'dt': Simulation timestep
                Optional keys with defaults:
                - 'drone_radius': Radius of drone (default: 0.05)
                - 'comm_radius': Communication radius (default: 1.0)
                - 'm': Mass of the drone (default: 0.0299 kg)
                - 'Ixx', 'Iyy', 'Izz': Moment of inertia (defaults from Crazyflie)
                - 'CT', 'CD': Thrust and drag coefficients
                - 'd': Arm length
                - 'max_thrust': Maximum motor thrust
                - 'use_gcbf': Whether to use GCBF safety layer (default: True)
                - 'gcbf_alpha': Alpha parameter for CBF (default: 1.0)
                - 'safety_margin': Safety margin for collision avoidance (default: 0.05)
        """
        super(CrazyFlieEnv, self).__init__(config)
        
        # Physical parameters
        self.m = config.get('m', 0.0299)  # kg
        self.Ixx = config.get('Ixx', 1.395e-5)  # kg*m^2
        self.Iyy = config.get('Iyy', 1.395e-5)  # kg*m^2
        self.Izz = config.get('Izz', 2.173e-5)  # kg*m^2
        self.CT = config.get('CT', 3.1582e-10)  # Thrust coefficient
        self.CD = config.get('CD', 7.9379e-12)  # Drag coefficient
        self.d = config.get('d', 0.03973)  # m, arm length
        self.g = 9.81  # m/s^2, gravity
        
        # Control limits
        self.max_thrust = config.get('max_thrust', 0.6)  # N, maximum thrust
        self.min_thrust = 0.0  # N, minimum thrust
        
        # State dimensions
        self.pos_dim = 3  # 3D positions (x, y, z)
        self.vel_dim = 3  # 3D velocities (u, v, w)
        self.ori_dim = 3  # 3D orientations (psi, theta, phi)
        self.ang_vel_dim = 3  # 3D angular velocities (r, q, p)
        self.state_dim = 12  # Full state dimension
        self.action_dim = 4  # Four motors
        
        # Safety parameters
        self.drone_radius = config.get('drone_radius', 0.05)
        
        # Register constants as buffers
        self.register_buffer('inertia_tensor', torch.tensor([
            [self.Ixx, 0, 0],
            [0, self.Iyy, 0],
            [0, 0, self.Izz]
        ]))
        
        self.register_buffer('inertia_tensor_inv', torch.tensor([
            [1.0/self.Ixx, 0, 0],
            [0, 1.0/self.Iyy, 0],
            [0, 0, 1.0/self.Izz]
        ]))
        
        self.register_buffer('g_vec', torch.tensor([0, 0, -self.g]))
        
        # Motor configuration matrix
        # Maps [f_1, f_2, f_3, f_4] to [F_z, tau_x, tau_y, tau_z]
        # F_z: total thrust
        # tau_x: roll torque
        # tau_y: pitch torque
        # tau_z: yaw torque
        motor_config = torch.tensor([
            [self.CT, self.CT, self.CT, self.CT],  # F_z
            [0, -self.d*self.CT, 0, self.d*self.CT],  # tau_x
            [self.d*self.CT, 0, -self.d*self.CT, 0],  # tau_y
            [self.CD, -self.CD, self.CD, -self.CD]  # tau_z
        ], dtype=torch.float32)
        self.register_buffer('motor_config', motor_config)
        
        # Try to compute the inverse of the motor configuration matrix
        # This will be used to convert desired forces to motor thrusts
        try:
            motor_config_inv = torch.inverse(motor_config)
            self.register_buffer('motor_config_inv', motor_config_inv)
        except:
            # If the matrix is not invertible, we'll handle this differently
            self.motor_config_inv = None
            
        # Initialize GCBF safety layer if requested
        self.use_gcbf = config.get('use_gcbf', True)
        if self.use_gcbf:
            from .gcbf_safety_layer import GCBFSafetyLayer
            
            # Create safety layer configuration
            safety_config = {
                'alpha': config.get('cbf_alpha', config.get('gcbf_alpha', 1.0)),
                'eps': config.get('gcbf_eps', 0.02),
                'safety_margin': config.get('safety_margin', self.drone_radius),
                'use_qp': config.get('use_qp', True),
                'qp_relaxation_weight': config.get('qp_relaxation_weight', 10.0),
                'max_iterations': config.get('max_iterations', 10)
            }
            
            # Create GCBF safety layer
            self.safety_layer = GCBFSafetyLayer(safety_config)
        else:
            self.safety_layer = None
    
    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Get observation shape: [n_agents, obs_dim]"""
        return (self.num_agents, self.state_dim + self.pos_dim)  # state + goal position
    
    @property
    def action_shape(self) -> Tuple[int, ...]:
        """Get action shape: [n_agents, action_dim]"""
        return (self.num_agents, self.action_dim)
    
    def get_action_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get bounds of valid actions (motor thrusts)."""
        lower_bound = self.min_thrust * torch.ones(self.action_shape, device=self.device)
        upper_bound = self.max_thrust * torch.ones(self.action_shape, device=self.device)
        return lower_bound, upper_bound
    
    def get_rotation_matrix(self, phi: torch.Tensor, theta: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """
        Compute the rotation matrix from body to world frame.
        
        Args:
            phi: Roll angle (radians)
            theta: Pitch angle (radians)
            psi: Yaw angle (radians)
            
        Returns:
            Rotation matrix [batch_size*n_agents, 3, 3]
        """
        # Compute trigonometric values
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)
        
        # Build rotation matrix
        R_W_B = torch.zeros(phi.shape[0], 3, 3, device=phi.device)
        
        # First row
        R_W_B[:, 0, 0] = cos_psi * cos_theta
        R_W_B[:, 0, 1] = cos_psi * sin_theta * sin_phi - sin_psi * cos_phi
        R_W_B[:, 0, 2] = cos_psi * sin_theta * cos_phi + sin_psi * sin_phi
        
        # Second row
        R_W_B[:, 1, 0] = sin_psi * cos_theta
        R_W_B[:, 1, 1] = sin_psi * sin_theta * sin_phi + cos_psi * cos_phi
        R_W_B[:, 1, 2] = sin_psi * sin_theta * cos_phi - cos_psi * sin_phi
        
        # Third row
        R_W_B[:, 2, 0] = -sin_theta
        R_W_B[:, 2, 1] = cos_theta * sin_phi
        R_W_B[:, 2, 2] = cos_theta * cos_phi
        
        return R_W_B
    
    def reset(self, batch_size: int = 1, randomize: bool = True) -> CrazyFlieState:
        """
        Reset the environment to an initial state.
        
        Args:
            batch_size: Number of parallel environments
            randomize: Whether to randomize initial positions and goals
            
        Returns:
            Initial environment state
        """
        # Create a device for tensors
        device = self.device
        
        # Initialize positions randomly within the environment bounds
        if randomize:
            # Random positions with z-value above ground
            positions = torch.rand(batch_size, self.num_agents, self.pos_dim, device=device)
            positions[..., 0:2] *= self.area_size  # x, y within area_size
            positions[..., 2] = 1.0  # z at 1.0 meters above ground
            
            # Random goals within the environment bounds
            goals = torch.rand(batch_size, self.num_agents, self.pos_dim, device=device)
            goals[..., 0:2] *= self.area_size  # x, y within area_size
            goals[..., 2] = 1.0  # z at 1.0 meters above ground
            
            # Ensure goals are not too close to initial positions
            min_goal_dist = 0.3 * self.area_size
            
            # Compute distances between positions and goals (in xy-plane)
            dist = torch.norm(positions[..., 0:2] - goals[..., 0:2], dim=2)
            
            # Reposition goals that are too close
            too_close = dist < min_goal_dist
            while torch.any(too_close):
                # Replace goals that are too close
                new_goals = torch.rand(batch_size, self.num_agents, self.pos_dim, device=device)
                new_goals[..., 0:2] *= self.area_size  # x, y within area_size
                new_goals[..., 2] = 1.0  # z at 1.0 meters
                
                goals = torch.where(too_close.unsqueeze(-1), new_goals, goals)
                
                # Recompute distances and check again
                dist = torch.norm(positions[..., 0:2] - goals[..., 0:2], dim=2)
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
                positions[:, i, 0] = spacing * (col + 1)  # x
                positions[:, i, 1] = spacing * (row + 1)  # y
                positions[:, i, 2] = 1.0  # z at 1.0 meters
                
                # Goals on opposite side
                goals[:, i, 0] = self.area_size - positions[0, i, 0]  # x
                goals[:, i, 1] = self.area_size - positions[0, i, 1]  # y
                goals[:, i, 2] = 1.0  # z at 1.0 meters
        
        # Initialize velocities as zeros
        velocities = torch.zeros(batch_size, self.num_agents, self.vel_dim, device=device)
        
        # Initialize orientations as zeros (level orientation)
        orientations = torch.zeros(batch_size, self.num_agents, self.ori_dim, device=device)
        
        # Initialize angular velocities as zeros
        angular_velocities = torch.zeros(batch_size, self.num_agents, self.ang_vel_dim, device=device)
        
        # Create state object
        return CrazyFlieState(
            positions=positions,
            velocities=velocities,
            goals=goals,
            orientations=orientations,
            angular_velocities=angular_velocities,
            batch_size=batch_size,
            step_count=0,
            obstacles=None
        )
    
    def dynamics(self, state: CrazyFlieState, action: torch.Tensor) -> torch.Tensor:
        """
        Apply CrazyFlie quadrotor dynamics to compute state derivatives.
        
        Args:
            state: Current state
            action: Control inputs (motor thrusts) [batch_size, n_agents, 4]
            
        Returns:
            State derivatives [batch_size, n_agents, 12]
        """
        # Extract batch size and flatten tensors for vectorized operations
        batch_size = state.batch_size
        total_agents = batch_size * self.num_agents
        
        # Reshape state components for easier handling
        # Each becomes [batch_size*n_agents, dim]
        positions_flat = state.positions.reshape(total_agents, self.pos_dim)
        velocities_flat = state.velocities.reshape(total_agents, self.vel_dim)
        orientations_flat = state.orientations.reshape(total_agents, self.ori_dim)
        angular_velocities_flat = state.angular_velocities.reshape(total_agents, self.ang_vel_dim)
        
        # Extract individual components
        phi = orientations_flat[:, 2]   # Roll
        theta = orientations_flat[:, 1]  # Pitch
        psi = orientations_flat[:, 0]    # Yaw
        
        p = angular_velocities_flat[:, 2]  # Roll rate
        q = angular_velocities_flat[:, 1]  # Pitch rate
        r = angular_velocities_flat[:, 0]  # Yaw rate
        
        # Reshape actions and ensure they're on the same device as motor_config
        actions_flat = action.reshape(total_agents, self.action_dim).to(self.motor_config.device)
        
        # Apply motor configuration to get forces and torques
        # [F_z, tau_x, tau_y, tau_z]
        forces_torques = torch.matmul(actions_flat, self.motor_config.T)
        
        # Extract components
        thrust = forces_torques[:, 0].unsqueeze(1)  # Total thrust [batch_size*n_agents, 1]
        torques = forces_torques[:, 1:4]            # Torques [batch_size*n_agents, 3]
        
        # Compute rotation matrix from body to world frame
        R_W_B = self.get_rotation_matrix(phi, theta, psi)
        
        # Compute force in world frame (only in z-direction in body frame)
        thrust_body = torch.zeros(total_agents, 3, device=self.device)
        thrust_body[:, 2] = thrust.squeeze()
        
        # Convert thrust from body to world frame
        thrust_world = torch.bmm(R_W_B, thrust_body.unsqueeze(2)).squeeze(2)
        
        # Compute linear acceleration (F = ma)
        # a = g + R_W_B * [0, 0, T] / m
        linear_accel = self.g_vec.unsqueeze(0).expand(total_agents, -1) + thrust_world / self.m
        
        # Compute angular acceleration
        # I^(-1) * (torques - w × (I * w))
        I_w = torch.matmul(self.inertia_tensor.unsqueeze(0).expand(total_agents, -1, -1), 
                          angular_velocities_flat.unsqueeze(2)).squeeze(2)
        
        # Cross product: w × (I * w)
        w_cross_I_w = torch.zeros(total_agents, 3, device=self.device)
        w_cross_I_w[:, 0] = angular_velocities_flat[:, 1] * I_w[:, 2] - angular_velocities_flat[:, 2] * I_w[:, 1]
        w_cross_I_w[:, 1] = angular_velocities_flat[:, 2] * I_w[:, 0] - angular_velocities_flat[:, 0] * I_w[:, 2]
        w_cross_I_w[:, 2] = angular_velocities_flat[:, 0] * I_w[:, 1] - angular_velocities_flat[:, 1] * I_w[:, 0]
        
        # Angular acceleration
        angular_accel = torch.matmul(self.inertia_tensor_inv.unsqueeze(0).expand(total_agents, -1, -1),
                                    (torques - w_cross_I_w).unsqueeze(2)).squeeze(2)
        
        # Compute orientation derivatives
        # Transform from body angular rates to Euler angle rates
        # This transformation depends on the current orientation
        ori_derivatives = torch.zeros(total_agents, 3, device=self.device)
        
        # yaw rate = r*cos(phi)/cos(theta) + q*sin(phi)/cos(theta)
        ori_derivatives[:, 0] = (r * torch.cos(phi) + q * torch.sin(phi)) / torch.cos(theta)
        
        # pitch rate = q*cos(phi) - r*sin(phi)
        ori_derivatives[:, 1] = q * torch.cos(phi) - r * torch.sin(phi)
        
        # roll rate = p + (q*sin(phi) + r*cos(phi))*tan(theta)
        ori_derivatives[:, 2] = p + (q * torch.sin(phi) + r * torch.cos(phi)) * torch.tan(theta)
        
        # Combine all derivatives
        derivatives = torch.cat([
            velocities_flat,           # Position derivatives = velocities
            ori_derivatives,           # Orientation derivatives
            linear_accel,              # Velocity derivatives = accelerations
            angular_accel              # Angular velocity derivatives = angular accelerations
        ], dim=1)
        
        # Reshape back to [batch_size, n_agents, state_dim]
        derivatives = derivatives.reshape(batch_size, self.num_agents, self.state_dim)
        
        return derivatives
    
    def rk4_step(self, state: CrazyFlieState, action: torch.Tensor) -> CrazyFlieState:
        """
        Perform a single step using 4th-order Runge-Kutta integration.
        
        Args:
            state: Current state
            action: Control inputs [batch_size, n_agents, 4]
            
        Returns:
            Next state
        """
        # Extract state components for easier handling
        state_tensor = torch.cat([
            state.positions,
            state.orientations,
            state.velocities,
            state.angular_velocities
        ], dim=2)  # [batch_size, n_agents, 12]
        
        # RK4 integration
        h = self.dt
        
        # Compute k1
        k1 = self.dynamics(state, action)
        
        # Compute k2
        # Create intermediate state at t + h/2 using k1
        pos_k1 = state.positions + 0.5 * h * k1[:, :, 0:3]
        ori_k1 = state.orientations + 0.5 * h * k1[:, :, 3:6]
        vel_k1 = state.velocities + 0.5 * h * k1[:, :, 6:9]
        ang_vel_k1 = state.angular_velocities + 0.5 * h * k1[:, :, 9:12]
        
        state_k1 = CrazyFlieState(
            positions=pos_k1,
            velocities=vel_k1,
            goals=state.goals,
            orientations=ori_k1,
            angular_velocities=ang_vel_k1,
            batch_size=state.batch_size,
            step_count=state.step_count,
            obstacles=state.obstacles
        )
        
        k2 = self.dynamics(state_k1, action)
        
        # Compute k3
        # Create intermediate state at t + h/2 using k2
        pos_k2 = state.positions + 0.5 * h * k2[:, :, 0:3]
        ori_k2 = state.orientations + 0.5 * h * k2[:, :, 3:6]
        vel_k2 = state.velocities + 0.5 * h * k2[:, :, 6:9]
        ang_vel_k2 = state.angular_velocities + 0.5 * h * k2[:, :, 9:12]
        
        state_k2 = CrazyFlieState(
            positions=pos_k2,
            velocities=vel_k2,
            goals=state.goals,
            orientations=ori_k2,
            angular_velocities=ang_vel_k2,
            batch_size=state.batch_size,
            step_count=state.step_count,
            obstacles=state.obstacles
        )
        
        k3 = self.dynamics(state_k2, action)
        
        # Compute k4
        # Create intermediate state at t + h using k3
        pos_k3 = state.positions + h * k3[:, :, 0:3]
        ori_k3 = state.orientations + h * k3[:, :, 3:6]
        vel_k3 = state.velocities + h * k3[:, :, 6:9]
        ang_vel_k3 = state.angular_velocities + h * k3[:, :, 9:12]
        
        state_k3 = CrazyFlieState(
            positions=pos_k3,
            velocities=vel_k3,
            goals=state.goals,
            orientations=ori_k3,
            angular_velocities=ang_vel_k3,
            batch_size=state.batch_size,
            step_count=state.step_count,
            obstacles=state.obstacles
        )
        
        k4 = self.dynamics(state_k3, action)
        
        # Combine all steps
        new_state_tensor = state_tensor + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Extract new state components
        new_positions = new_state_tensor[:, :, 0:3]
        new_orientations = new_state_tensor[:, :, 3:6]
        new_velocities = new_state_tensor[:, :, 6:9]
        new_angular_velocities = new_state_tensor[:, :, 9:12]
        
        # Create new state object
        new_state = CrazyFlieState(
            positions=new_positions,
            velocities=new_velocities,
            goals=state.goals,
            orientations=new_orientations,
            angular_velocities=new_angular_velocities,
            batch_size=state.batch_size,
            step_count=state.step_count + 1,
            obstacles=state.obstacles
        )
        
        return new_state
    
    def step(self, state: CrazyFlieState, action: torch.Tensor) -> StepResult:
        """
        Take a step in the environment using the given actions.
        
        Args:
            state: Current state
            action: Actions to take [batch_size, n_agents, action_dim]
            
        Returns:
            StepResult containing next_state, reward, cost, done, info
        """
        # Ensure action is on the correct device first
        action = action.to(self.device)
        
        # Apply safety layer if it exists (default implementation just returns the action)
        safe_action = self.apply_safety_layer(state, action)
        
        # Ensure safe_action is also on the correct device
        safe_action = safe_action.to(self.device)
        
        # Clip action to bounds
        lower_bound, upper_bound = self.get_action_bounds()
        safe_action = torch.clamp(safe_action, lower_bound, upper_bound)
        
        # Update state using RK4 integration for better accuracy
        next_state = self.rk4_step(state, safe_action)
        
        # Compute reward, cost, and done
        reward = self.compute_reward(state, safe_action, next_state)
        cost = self.compute_cost(state, safe_action, next_state)
        done = self.is_done(next_state)
        
        # Prepare info dict
        info = {
            'goal_distance': self.get_goal_distance(next_state),
            'collision': self.check_collision(next_state),
            'action': safe_action,
            'raw_action': action
        }
        
        return StepResult(next_state, reward, cost, done, info)
    
    def get_observation(self, state: CrazyFlieState) -> torch.Tensor:
        """
        Extract observation from environment state.
        
        Args:
            state: Current environment state
            
        Returns:
            Observation tensor [batch_size, n_agents, obs_dim]
        """
        # Combine state components into observation
        # Full state + goal position
        observation = torch.cat([
            state.positions,           # positions
            state.orientations,        # orientations
            state.velocities,          # velocities
            state.angular_velocities,  # angular velocities
            state.goals                # goals
        ], dim=2)
        
        return observation
    
    def render(self, state: CrazyFlieState) -> Any:
        """
        Render the environment state. This is a placeholder implementation.
        
        Args:
            state: Environment state to render
            
        Returns:
            Rendering of the environment (matplotlib figure)
        """
        # This method would typically create a visualization of the environment
        # For now, we'll just return a simple message
        return "Rendering not yet implemented for CrazyFlieEnv"
    
    def apply_safety_layer(self, state: CrazyFlieState, raw_action: torch.Tensor) -> torch.Tensor:
        """
        Apply safety constraints to raw actions using GCBF.
        
        This method integrates the GCBF safety layer from the original implementation.
        
        Args:
            state: Current environment state
            raw_action: Raw action from policy [batch_size, n_agents, action_dim]
            
        Returns:
            Safe action [batch_size, n_agents, action_dim]
        """
        if not self.use_gcbf or self.safety_layer is None:
            # No safety filtering, just return the raw action
            return raw_action
        
        # Apply GCBF safety layer
        # This calls the forward method of GCBFSafetyLayer which performs:
        # 1. Compute barrier function values
        # 2. Compute barrier function gradients
        # 3. Compute control-affine dynamics
        # 4. Solve optimization problem to find safe actions
        safe_action = self.safety_layer(state, raw_action, dynamics_fn=self.get_control_affine_dynamics)
        
        return safe_action
    
    def get_control_affine_dynamics(self, state: CrazyFlieState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get control-affine dynamics for GCBF safety layer.
        
        For the quadrotor, we need to simplify the dynamics to fit the control-affine form:
        dx/dt = f(x) + g(x)u
        
        We'll use a double integrator approximation for the horizontal plane:
        dx/dt = vx
        dy/dt = vy
        dvx/dt = 1/m * fx
        dvy/dt = 1/m * fy
        
        And ensure that it remains at safe altitude:
        dz/dt = vz
        dvz/dt = 1/m * fz - g
        
        Args:
            state: Current state
            
        Returns:
            Tuple of (f, g) where:
            - f: Drift term [batch_size, n_agents, state_dim]
            - g: Control input term [batch_size, n_agents, state_dim, action_dim]
        """
        batch_size = state.batch_size
        n_agents = state.positions.shape[1]
        device = state.positions.device
        
        # Simplify the state to position and velocity only (ignoring orientation and angular velocity)
        pos_dim = state.positions.shape[2]
        vel_dim = state.velocities.shape[2]
        state_dim = pos_dim + vel_dim  # Positions + velocities
        
        # Drift term f(x) = [vx, vy, vz, 0, 0, -g]
        f = torch.zeros(batch_size, n_agents, state_dim, device=device)
        f[:, :, :pos_dim] = state.velocities  # Position derivatives = velocities
        f[:, :, pos_dim+2] = -self.g  # Add gravity effect for z-axis
        
        # Control input term g(x) = [0, 0, 0; 0, 0, 0; 0, 0, 0; 1/m, 0, 0; 0, 1/m, 0; 0, 0, 1/m]
        g = torch.zeros(batch_size, n_agents, state_dim, state.velocities.shape[2], device=device)
        
        # Each force component affects only its corresponding velocity
        for i in range(vel_dim):
            g[:, :, pos_dim+i, i] = 1.0 / self.m
        
        return f, g
    
    def vel_targets_to_thrusts(self, vel_targets: torch.Tensor, state: CrazyFlieState) -> torch.Tensor:
        """
        Convert velocity targets to motor thrusts using a control allocation method.
        
        This allows using a higher-level control interface similar to the original implementation.
        
        Args:
            vel_targets: Desired velocities [batch_size, n_agents, 4] (vx, vy, vz, yaw_rate)
            state: Current state
            
        Returns:
            Motor thrusts [batch_size, n_agents, 4]
        """
        # This is a simplified implementation
        # A full implementation would use a cascaded PID controller
        
        # Extract batch size and flatten tensors for vectorized operations
        batch_size = state.batch_size
        total_agents = batch_size * self.num_agents
        
        # Target velocities
        vx_target = vel_targets[..., 0].reshape(total_agents)
        vy_target = vel_targets[..., 1].reshape(total_agents)
        vz_target = vel_targets[..., 2].reshape(total_agents)
        yaw_rate_target = vel_targets[..., 3].reshape(total_agents)
        
        # Current velocities
        vx = state.velocities[..., 0].reshape(total_agents)
        vy = state.velocities[..., 1].reshape(total_agents)
        vz = state.velocities[..., 2].reshape(total_agents)
        yaw_rate = state.angular_velocities[..., 0].reshape(total_agents)
        
        # Simple PD controller gains
        k_p = 0.5
        k_d = 0.1
        
        # Compute desired accelerations
        ax_des = k_p * (vx_target - vx)
        ay_des = k_p * (vy_target - vy)
        az_des = k_p * (vz_target - vz) + self.g  # Compensate for gravity
        yaw_acc_des = k_p * (yaw_rate_target - yaw_rate)
        
        # Extract orientations
        phi = state.orientations[..., 2].reshape(total_agents)    # Roll
        theta = state.orientations[..., 1].reshape(total_agents)  # Pitch
        
        # Simple attitude controller to achieve desired accelerations
        # Desired roll and pitch based on desired accelerations
        g = self.g
        phi_des = torch.atan2(-ay_des, torch.sqrt(ax_des**2 + (az_des - g)**2))
        theta_des = torch.atan2(ax_des, (az_des - g))
        
        # Compute attitude errors
        phi_err = phi_des - phi
        theta_err = theta_des - theta
        
        # Compute desired torques
        tau_x = 5.0 * phi_err - 0.5 * state.angular_velocities[..., 2].reshape(total_agents)
        tau_y = 5.0 * theta_err - 0.5 * state.angular_velocities[..., 1].reshape(total_agents)
        tau_z = yaw_acc_des * self.Izz
        
        # Compute desired thrust (in body frame)
        thrust = self.m * torch.sqrt(ax_des**2 + ay_des**2 + (az_des - g)**2)
        
        # Combine into force/torque vector [thrust, tau_x, tau_y, tau_z]
        forces_torques = torch.stack([thrust, tau_x, tau_y, tau_z], dim=1)
        
        # Convert to motor thrusts using the inverse of motor configuration
        if self.motor_config_inv is not None:
            motor_thrusts = torch.matmul(forces_torques, self.motor_config_inv)
        else:
            # Fallback if inverse doesn't exist
            # This is a simple approximation that works for basic cases
            motor_thrusts = torch.zeros(total_agents, 4, device=self.device)
            motor_thrusts[:, 0] = thrust/4 + tau_y/(4*self.d) + tau_z/(4*self.CD)
            motor_thrusts[:, 1] = thrust/4 - tau_x/(4*self.d) - tau_z/(4*self.CD)
            motor_thrusts[:, 2] = thrust/4 - tau_y/(4*self.d) + tau_z/(4*self.CD)
            motor_thrusts[:, 3] = thrust/4 + tau_x/(4*self.d) - tau_z/(4*self.CD)
        
        # Clip motor thrusts to valid range
        motor_thrusts = torch.clamp(motor_thrusts, self.min_thrust, self.max_thrust)
        
        # Reshape back to [batch_size, n_agents, 4]
        motor_thrusts = motor_thrusts.reshape(batch_size, self.num_agents, 4)
        
        return motor_thrusts 