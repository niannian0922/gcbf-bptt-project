#!/usr/bin/env python3
import os
import torch
import yaml
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

from gcbfplus.env import DoubleIntegratorEnv, CrazyFlieEnv
from gcbfplus.policy import BPTTPolicy, create_policy_from_config


def visualize_trajectory(env, policy_network, cbf_network, device, config, save_path=None):
    """
    Run a simulation using the trained policy and CBF networks and visualize the trajectory.
    
    Args:
        env: Environment to simulate in
        policy_network: Trained policy network
        cbf_network: Trained CBF network (can be None)
        device: Device to run computation on
        config: Configuration dictionary
        save_path: Path to save the animation
    """
    # Set networks to evaluation mode
    policy_network.eval()
    if cbf_network is not None:
        cbf_network.eval()
    
    # Initialize environment
    state = env.reset(batch_size=1, randomize=True)
    
    # Parameters for visualization
    num_agents = env.num_agents
    area_size = env.area_size
    car_radius = env.agent_radius
    horizon = config.get('eval_horizon', 100)
    
    # Initialize metrics tracking
    time_to_goal = 0
    total_path_length = 0
    min_separation_distance = float('inf')
    total_control_effort = 0
    goal_reached = False
    collision_detected = False
    
    # For path length calculation
    prev_positions = state.positions.clone()
    
    # Store per-step metrics
    min_distances = []
    control_efforts = []
    
    # Store all states for visualization
    all_positions = [state.positions.clone().cpu().numpy()]
    all_cbf_values = []
    all_actions = []
    
    # Extract obstacle information if available
    has_obstacles = state.obstacles is not None
    if has_obstacles:
        obstacles = state.obstacles[0].cpu().numpy()  # Shape [n_obstacles, pos_dim+1]
    else:
        obstacles = None
    
    with torch.no_grad():
        for t in range(horizon):
            # Get observations from state
            observations = env.get_observation(state)
            
            # Move observations to device for network processing
            observations = observations.to(device)
            
            # Get CBF values if CBF network is available
            if cbf_network is not None:
                h_vals = cbf_network(observations)
                all_cbf_values.append(h_vals.cpu().numpy())
            
            # Get actions from policy network
            actions = policy_network(observations)
            all_actions.append(actions.clone())
            
            # Calculate metrics before stepping
            
            # 1. Minimum inter-agent distance
            positions = state.positions[0]  # [num_agents, 2]
            min_distance = float('inf')
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    dist = torch.norm(positions[i] - positions[j]).item()
                    min_distance = min(min_distance, dist)
            min_distances.append(min_distance)
            min_separation_distance = min(min_separation_distance, min_distance)
            
            # 2. Control effort
            control_effort = torch.sum(actions ** 2).item()
            control_efforts.append(control_effort)
            total_control_effort += control_effort
            
            # Step simulation
            step_result = env.step(state, actions)
            next_state = step_result.next_state
            
            # 3. Path length calculation
            delta_positions = next_state.positions - prev_positions
            step_path_length = torch.sum(torch.norm(delta_positions, dim=2)).item()
            total_path_length += step_path_length
            prev_positions = next_state.positions.clone()
            
            # Store positions for visualization
            all_positions.append(next_state.positions.clone().cpu().numpy())
            
            # Check if all agents reached their goals
            distances = env.get_goal_distance(next_state)
            if torch.all(distances < 2 * car_radius) and not goal_reached:
                time_to_goal = t + 1
                goal_reached = True
                print(f"All agents reached their goals at step {time_to_goal}")
                
            # Check for collisions
            if torch.any(step_result.cost > 0) and not collision_detected:
                print(f"Collision detected at step {t+1}")
                collision_detected = True
            
            # Update state
            state = next_state
            
            # Early termination if all goals reached and we're past the time to goal
            if goal_reached and t >= time_to_goal + 5:
                break
    
    # If goals were never reached, record the maximum time
    if not goal_reached:
        time_to_goal = len(all_positions) - 1
    
    # Convert lists to numpy arrays
    all_positions = np.array(all_positions)
    if cbf_network is not None:
        all_cbf_values = np.array(all_cbf_values)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Define agent and goal colors
    agent_colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    # Create agent patches
    agent_patches = []
    for i in range(num_agents):
        agent = patches.Circle((0, 0), radius=car_radius, fc=agent_colors[i], ec='black', alpha=0.7)
        ax.add_patch(agent)
        agent_patches.append(agent)
    
    # Add goal markers
    goal_markers = []
    for i in range(num_agents):
        goal_pos = state.goals[0, i].cpu().numpy()
        goal = ax.plot(goal_pos[0], goal_pos[1], 'x', markersize=10, color=agent_colors[i])[0]
        goal_markers.append(goal)
    
    # Add trajectory lines
    trajectory_lines = []
    for i in range(num_agents):
        line, = ax.plot([], [], '-', linewidth=1, color=agent_colors[i], alpha=0.5)
        trajectory_lines.append(line)
    
    # Add obstacles if present
    obstacle_patches = []
    if has_obstacles:
        for i in range(obstacles.shape[0]):
            obs_pos = obstacles[i, :2]
            obs_radius = obstacles[i, 2]
            obstacle = patches.Circle(obs_pos, radius=obs_radius, fc='red', ec='darkred', alpha=0.3)
            ax.add_patch(obstacle)
            obstacle_patches.append(obstacle)
    
    # Set plot limits and labels
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_aspect('equal')
    
    # Get CBF alpha value for title if available
    cbf_alpha = config.get('cbf_alpha', None)
    if cbf_alpha is not None:
        title = f'Multi-Agent Navigation with GCBF+BTN (alpha = {cbf_alpha})'
        if has_obstacles:
            title += ' with Obstacles'
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Multi-Agent Navigation with GCBF+BTN', fontsize=14)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    # Add text elements for information display
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, ha='left', va='top')
    metrics_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, ha='left', va='top', fontsize=8)
    cbf_text = ax.text(0.02, 0.86, '', transform=ax.transAxes, ha='left', va='top')
    
    def init():
        """Initialize the animation."""
        for i, patch in enumerate(agent_patches):
            patch.center = all_positions[0, 0, i, 0], all_positions[0, 0, i, 1]
        
        for line in trajectory_lines:
            line.set_data([], [])
        
        time_text.set_text('')
        metrics_text.set_text('')
        cbf_text.set_text('')
        
        return agent_patches + trajectory_lines + [time_text, metrics_text, cbf_text]
    
    def animate(frame):
        """Update the animation for each frame."""
        # Update agent positions
        for i, patch in enumerate(agent_patches):
            patch.center = all_positions[frame, 0, i, 0], all_positions[frame, 0, i, 1]
        
        # Update trajectory lines
        for i, line in enumerate(trajectory_lines):
            line.set_data(all_positions[:frame+1, 0, i, 0], all_positions[:frame+1, 0, i, 1])
        
        # Update text information
        time_text.set_text(f'Time step: {frame}')
        
        # Update metrics text if available for this frame
        if frame > 0 and frame < len(min_distances) + 1:
            metrics_info = (
                f"Min Distance: {min_distances[frame-1]:.3f}\n"
                f"Control Effort: {control_efforts[frame-1]:.3f}"
            )
            metrics_text.set_text(metrics_info)
        
        if cbf_network is not None and frame < len(all_cbf_values) + 1:
            min_cbf = all_cbf_values[frame-1].min() if frame > 0 else 0
            cbf_text.set_text(f'Min CBF value: {min_cbf:.4f}')
        
        return agent_patches + trajectory_lines + [time_text, metrics_text, cbf_text]
    
    # Create animation
    num_frames = len(all_positions)
    animation = FuncAnimation(fig, animate, frames=num_frames, init_func=init, 
                              interval=100, blit=True)
    
    # Save animation if requested
    if save_path:
        try:
            animation.save(save_path, writer='ffmpeg', fps=10, dpi=200)
        except (ValueError, RuntimeError):
            # If ffmpeg is not available, use default writer
            animation.save(save_path, fps=10, dpi=100)
        print(f"Animation saved to {save_path}")
    
    # Print quantitative metrics summary
    print("\n" + "="*40)
    print("          QUANTITATIVE ANALYSIS          ")
    print("="*40)
    print(f"Time to Goal: {time_to_goal} steps")
    print(f"Total Path Length: {total_path_length:.3f} units")
    print(f"Minimum Separation Distance: {min_separation_distance:.3f} units")
    print(f"Total Control Effort: {total_control_effort:.3f}")
    if collision_detected:
        print("WARNING: Collisions detected during simulation!")
    print("="*40)
    
    plt.close()
    
    # Return both the animation and metrics
    metrics = {
        'time_to_goal': time_to_goal,
        'total_path_length': total_path_length,
        'min_separation_distance': min_separation_distance,
        'total_control_effort': total_control_effort,
        'collisions_detected': collision_detected,
        'has_obstacles': has_obstacles
    }
    
    return animation, metrics


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize trained BPTT models")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing trained models")
    parser.add_argument("--step", type=int, default=None, help="Model step to visualize (default: latest)")
    parser.add_argument("--output", type=str, default=None, help="Path to save the animation")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--env_type", type=str, default="double_integrator", help="Environment type (double_integrator or crazyflie)")
    parser.add_argument("--metrics_output", type=str, default=None, help="Path to save metrics as JSON (optional)")
    args = parser.parse_args()
    
    # Set up the device - Force GPU usage if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Legacy support for cuda flag
    use_cuda = torch.cuda.is_available()
    
    # Load configuration
    config_path = Path(args.model_dir) / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine which model step to use
    model_dir = Path(args.model_dir) / "models"
    if args.step is None:
        # First check if there's a final directory
        final_dir = model_dir / "final"
        if final_dir.is_dir() and (final_dir / "policy.pt").exists():
            step = "final"
        else:
            # Find the latest step
            steps = [int(d.name) for d in model_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if not steps:
                raise ValueError(f"No model checkpoints found in {model_dir}")
            step = max(steps)
    else:
        step = args.step
    
    print(f"Using model checkpoint from step {step}")
    
    # Create environment based on config and type
    env_config = config['env']
    env_type = args.env_type.lower()
    
    if env_type == "double_integrator":
        env = DoubleIntegratorEnv(env_config)
    elif env_type == "crazyflie":
        env = CrazyFlieEnv(env_config)
    else:
        raise ValueError(f"Unsupported environment type: {env_type}. Choose 'double_integrator' or 'crazyflie'")
    
    # Move environment to device
    env = env.to(device)
    
    # Create policy network
    # Prepare policy configuration
    policy_config = {
        'type': 'bptt',
        'perception': {
            'input_dim': env.observation_shape[-1],
            'hidden_dim': config['networks']['policy']['hidden_dim'],
            'num_layers': config['networks']['policy']['n_layers'],
            'activation': 'relu',
            'use_batch_norm': False
        },
        'memory': {
            'hidden_dim': config['networks']['policy']['hidden_dim'],
            'num_layers': 1
        },
        'policy_head': {
            'output_dim': env.action_shape[-1],
            'hidden_dims': [config['networks']['policy']['hidden_dim']],
            'action_scaling': True,
            'action_bound': 1.0
        }
    }
    
    # Create policy network
    policy_network = create_policy_from_config(policy_config)
    policy_network = policy_network.to(device)
    
    # Create CBF network if available
    cbf_network = None
    cbf_config = config['networks'].get('cbf')
    
    if cbf_config:
        # For now, we'll use a simple MLP as CBF network
        cbf_input_dim = env.observation_shape[-1]
        cbf_network = torch.nn.Sequential(
            torch.nn.Linear(cbf_input_dim, cbf_config['hidden_dim']),
            torch.nn.ReLU(),
            torch.nn.Linear(cbf_config['hidden_dim'], cbf_config['hidden_dim']),
            torch.nn.ReLU(),
            torch.nn.Linear(cbf_config['hidden_dim'], 1)
        ).to(device)
    
    # Load the trained models
    policy_path = model_dir / str(step) / "policy.pt"
    policy_network.load_state_dict(torch.load(policy_path, map_location=device))
    print(f"Loaded policy from {policy_path}")
    
    if cbf_network is not None:
        cbf_path = model_dir / str(step) / "cbf.pt"
        if cbf_path.exists():
            cbf_network.load_state_dict(torch.load(cbf_path, map_location=device))
            print(f"Loaded CBF network from {cbf_path}")
        else:
            print(f"Warning: CBF network file not found at {cbf_path}")
    
    # Output path
    if args.output is None:
        output_path = Path(args.model_dir) / f"visualization_step_{step}.gif"
    else:
        output_path = Path(args.output)
        # Ensure the extension is .gif if not specified
        if output_path.suffix.lower() not in ['.gif', '.png', '.jpg', '.jpeg']:
            output_path = output_path.with_suffix('.gif')
            
    # Make sure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get CBF alpha value for metrics
    cbf_alpha = env_config.get('cbf_alpha', config['training'].get('cbf_alpha', 1.0))
    
    # Add CBF alpha to training config for visualization
    training_config = config['training'].copy()
    training_config['cbf_alpha'] = cbf_alpha
    
    # Run visualization
    animation, metrics = visualize_trajectory(
        env=env,
        policy_network=policy_network,
        cbf_network=cbf_network,
        device=device,
        config=training_config,
        save_path=str(output_path)
    )
    
    # Enhance metrics with experiment details
    metrics.update({
        'cbf_alpha': cbf_alpha,
        'num_agents': env.num_agents,
        'area_size': env.area_size,
        'model_step': step
    })
    
    # Save metrics to file if specified
    if args.metrics_output:
        import json
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
    
    # Save metrics to standard location in model directory
    metrics_file = Path(args.model_dir) / "metrics.json"
    import json
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics also saved to {metrics_file}")
    
    # Print summary message
    print(f"Visualization completed and saved to {output_path}")
    print(f"CBF Alpha: {cbf_alpha}, Time to Goal: {metrics['time_to_goal']} steps, Min Separation: {metrics['min_separation_distance']:.3f}, Control Effort: {metrics['total_control_effort']:.3f}")
    
    return metrics


if __name__ == "__main__":
    main() 