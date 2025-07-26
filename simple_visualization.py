#!/usr/bin/env python3
import torch
import numpy as np
import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

from gcbfplus.env import DoubleIntegratorEnv, CrazyFlieEnv
from gcbfplus.policy import BPTTPolicy, create_policy_from_config

def visualize_trajectory(env, policy, save_path=None, title=None, show_metrics=True):
    """
    Run a simple trajectory visualization with quantitative metrics tracking.
    
    Args:
        env: Environment instance
        policy: Policy network
        save_path: Path to save the animation
        title: Custom title for the visualization
        show_metrics: Whether to display metrics in the animation
    """
    # Set policy to evaluation mode
    policy.eval()
    
    # Reset environment
    state = env.reset(batch_size=1, randomize=True)
    
    # Get environment parameters
    num_agents = env.num_agents
    area_size = env.area_size
    agent_radius = env.agent_radius
    
    # Initialize metrics tracking
    time_to_goal = 0
    total_path_length = 0
    min_separation_distance = float('inf')
    total_control_effort = 0
    goal_reached = False
    
    # For path length calculation
    prev_positions = state.positions.clone()
    
    # Tracking per-step metrics
    min_distances = []
    control_efforts = []
    
    # Run simulation
    trajectory = [state.positions.clone().cpu().numpy()]
    actions_history = []
    
    with torch.no_grad():
        for step in range(50):  # Run for 50 steps
            # Get observation
            obs = env.get_observation(state)
            
            # Get action
            action = policy(obs, reset_memory=(step==0))
            actions_history.append(action.clone())
            
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
            
            # 2. Control effort (sum of squared norm of actions)
            control_effort = torch.sum(action ** 2).item()
            control_efforts.append(control_effort)
            total_control_effort += control_effort
            
            # Step environment
            step_result = env.step(state, action)
            next_state = step_result.next_state
            
            # 3. Path length calculation
            delta_positions = next_state.positions - prev_positions
            step_path_length = torch.sum(torch.norm(delta_positions, dim=2)).item()
            total_path_length += step_path_length
            prev_positions = next_state.positions.clone()
            
            # Store position
            trajectory.append(next_state.positions.clone().cpu().numpy())
            
            # Check if all agents reached their goals
            distances = env.get_goal_distance(next_state)
            if torch.all(distances < 0.1) and not goal_reached:
                time_to_goal = step + 1
                goal_reached = True
                print(f"All agents reached their goals at step {time_to_goal}!")
            
            # Update state
            state = next_state
            
            # Early termination if all goals reached
            if goal_reached:
                break
    
    # If goals were never reached, record the maximum time
    if not goal_reached:
        time_to_goal = len(trajectory) - 1
    
    # Convert trajectory to numpy array
    trajectory = np.array(trajectory)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Define agent colors
    agent_colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    # Create agent patches
    agent_patches = []
    for i in range(num_agents):
        agent = patches.Circle((0, 0), radius=agent_radius, fc=agent_colors[i], ec='black', alpha=0.7)
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
    
    # Set plot limits and labels
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_aspect('equal')
    
    # Set title (use custom title if provided)
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Agent Trajectory Visualization', fontsize=14)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    # Add text elements for information display
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, ha='left', va='top')
    metrics_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, ha='left', va='top', fontsize=8)
    
    def init():
        """Initialize the animation."""
        for i, patch in enumerate(agent_patches):
            patch.center = trajectory[0, 0, i, 0], trajectory[0, 0, i, 1]
        
        for line in trajectory_lines:
            line.set_data([], [])
        
        time_text.set_text('')
        metrics_text.set_text('')
        
        return agent_patches + trajectory_lines + [time_text, metrics_text]
    
    def animate(frame):
        """Update the animation for each frame."""
        # Update agent positions
        for i, patch in enumerate(agent_patches):
            patch.center = trajectory[frame, 0, i, 0], trajectory[frame, 0, i, 1]
        
        # Update trajectory lines
        for i, line in enumerate(trajectory_lines):
            line.set_data(trajectory[:frame+1, 0, i, 0], trajectory[:frame+1, 0, i, 1])
        
        # Update text information
        time_text.set_text(f'Time step: {frame}')
        
        # Update metrics text if enabled
        if show_metrics and frame < len(min_distances):
            metrics_info = (
                f"Min Distance: {min_distances[frame]:.3f}\n"
                f"Control Effort: {control_efforts[frame]:.3f}"
            )
            metrics_text.set_text(metrics_info)
        else:
            metrics_text.set_text('')
        
        return agent_patches + trajectory_lines + [time_text, metrics_text]
    
    # Create animation
    num_frames = len(trajectory)
    animation = FuncAnimation(fig, animate, frames=num_frames, init_func=init, 
                              interval=100, blit=True)
    
    # Save animation if requested
    if save_path:
        animation.save(save_path, writer='ffmpeg', fps=10, dpi=100)
        print(f"Animation saved to {save_path}")
    
    # Print quantitative metrics summary
    print("\n" + "="*40)
    print("          QUANTITATIVE ANALYSIS          ")
    print("="*40)
    print(f"Time to Goal: {time_to_goal} steps")
    print(f"Total Path Length: {total_path_length:.3f} units")
    print(f"Minimum Separation Distance: {min_separation_distance:.3f} units")
    print(f"Total Control Effort: {total_control_effort:.3f}")
    print("="*40)
    
    # Return both animation and metrics
    metrics = {
        'time_to_goal': time_to_goal,
        'total_path_length': total_path_length,
        'min_separation_distance': min_separation_distance,
        'total_control_effort': total_control_effort
    }
    
    return animation, metrics

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize trained policies")
    parser.add_argument("--model_dir", type=str, default="logs/simple_training",
                        help="Directory containing trained models")
    parser.add_argument("--output", type=str, default="logs/visualization.gif",
                        help="Output file path for the visualization")
    parser.add_argument("--env_type", type=str, default="double_integrator",
                        help="Environment type (double_integrator or crazyflie)")
    parser.add_argument("--metrics_output", type=str, default=None,
                        help="Path to save metrics as JSON (optional)")
    args = parser.parse_args()
    
    print(f"Running visualization from model directory: {args.model_dir}")
    
    # Load configuration if available
    config_path = Path(args.model_dir) / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        
        # Extract necessary configurations
        env_config = config.get('env', {})
        network_config = config.get('networks', {})
        policy_network_config = network_config.get('policy', {})
        
        # Get CBF alpha value for title
        cbf_alpha = env_config.get('cbf_alpha', 1.0)
        title = f"GCBF Agent Trajectories (alpha = {cbf_alpha})"
    else:
        print("No configuration file found, using default values")
        env_config = {
            'num_agents': 3,
            'area_size': 1.0,
            'dt': 0.03,
            'mass': 0.1,
            'car_radius': 0.05,
            'comm_radius': 0.5,
            'max_steps': 100
        }
        policy_network_config = {}
        cbf_alpha = 1.0
        title = "Agent Trajectory Visualization"
    
    # Create environment
    env_type = args.env_type.lower()
    if env_type == "double_integrator":
        env = DoubleIntegratorEnv(env_config)
    elif env_type == "crazyflie":
        env = CrazyFlieEnv(env_config)
    else:
        raise ValueError(f"Unsupported environment type: {env_type}")
    
    print(f"Created environment: {env.__class__.__name__}")
    
    # Create policy
    policy_config = {
        'type': 'bptt',
        'perception': {
            'input_dim': env.observation_shape[-1],
            'hidden_dim': policy_network_config.get('hidden_dim', 32),
            'num_layers': policy_network_config.get('n_layers', 2)
        },
        'memory': {
            'hidden_dim': policy_network_config.get('hidden_dim', 32),
            'num_layers': 1
        },
        'policy_head': {
            'output_dim': env.action_shape[-1],
            'hidden_dims': [policy_network_config.get('hidden_dim', 32)],
            'action_scaling': True,
            'action_bound': 1.0
        }
    }
    
    policy = create_policy_from_config(policy_config)
    print(f"Created policy: {policy.__class__.__name__}")
    
    # Try to load the policy weights
    # First try the models/final directory
    model_path = Path(args.model_dir) / "models" / "final" / "policy.pt"
    if not model_path.exists():
        # If not found, look for the highest numbered subdirectory in the models directory
        models_dir = Path(args.model_dir) / "models"
        if models_dir.exists():
            model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if model_dirs:
                highest_step_dir = max(model_dirs, key=lambda d: int(d.name))
                model_path = highest_step_dir / "policy.pt"
    
    if model_path.exists():
        policy.load_state_dict(torch.load(model_path))
        print(f"Loaded policy weights from {model_path}")
    else:
        print("No trained policy found, using random policy")
    
    # Make sure the output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Visualize
    animation, metrics = visualize_trajectory(env, policy, save_path=str(output_path), title=title)
    
    # Enhance metrics with experiment details
    metrics.update({
        'cbf_alpha': cbf_alpha,
        'num_agents': env.num_agents,
        'area_size': env.area_size,
    })
    
    # Save metrics to file if specified
    if args.metrics_output:
        import json
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
    
    # If this is part of a comparative run, save metrics to the standard location
    metrics_file = Path(args.model_dir) / "metrics.json"
    import json
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics also saved to {metrics_file}")
    
    # Print a summary message
    print(f"Visualization completed and saved to {output_path}")
    print(f"CBF Alpha: {cbf_alpha}, Time to Goal: {metrics['time_to_goal']} steps, Min Separation: {metrics['min_separation_distance']:.3f}, Control Effort: {metrics['total_control_effort']:.3f}")
    
    return metrics

if __name__ == "__main__":
    main() 