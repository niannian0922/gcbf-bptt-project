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

from gcbfplus.nn.torch_gnn import PolicyGNN, CBFGNN
from gcbfplus.env.differentiable_simulator import (
    DifferentiableDoubleIntegrator
)
from gcbfplus.trainer.bptt_utils import (
    initialize_states_and_goals,
    build_graph_features
)


def visualize_trajectory(policy_network, cbf_network, simulator, device, config, save_path=None):
    """
    Run a simulation using the trained policy and CBF networks and visualize the trajectory.
    
    Args:
        policy_network: Trained policy network
        cbf_network: Trained CBF network
        simulator: Differentiable simulator
        device: Device to run computation on
        config: Configuration dictionary
        save_path: Path to save the animation
    """
    # Set networks to evaluation mode
    policy_network.eval()
    cbf_network.eval()
    
    # Initialize scenario
    num_agents = config['env']['num_agents']
    area_size = config['env']['area_size']
    car_radius = config['env']['car_radius']
    
    # Initialize states and goals using the utility function
    states, goals = initialize_states_and_goals(
        num_agents=num_agents,
        state_dim=4,  # x, y, vx, vy
        area_size=area_size,
        min_dist=4 * car_radius,
        max_travel=None,
        device=device
    )
    
    # Run simulation and collect trajectory
    horizon = config['training']['eval_horizon']
    
    # Store all states for visualization
    all_states = [states.clone().cpu().numpy()]
    all_cbf_values = []
    
    with torch.no_grad():
        for t in range(horizon):
            # Build graph features using the utility function
            graph_features = build_graph_features(
                states=states,
                goals=goals,
                sensing_radius=config['env']['comm_radius'],
                device=device
            )
            
            # Get CBF values
            h_vals = cbf_network(graph_features)
            all_cbf_values.append(h_vals.cpu().numpy())
            
            # Get actions from policy network
            actions = policy_network(graph_features)
            
            # Step simulation
            states = simulator(states, actions)
            
            # Store state for visualization
            all_states.append(states.clone().cpu().numpy())
            
            # Check if all agents reached their goals
            distances = torch.norm(states[:, :2] - goals, dim=1)
            if torch.all(distances < 2 * car_radius):
                print(f"All agents reached their goals at step {t+1}")
                break
    
    # Convert lists to numpy arrays
    all_states = np.array(all_states)
    all_cbf_values = np.array(all_cbf_values)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Define agent and goal colors
    agent_colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    
    # Create agent and goal patches
    agent_patches = []
    for i in range(num_agents):
        agent = patches.Circle((0, 0), radius=car_radius, fc=agent_colors[i], ec='black', alpha=0.7)
        ax.add_patch(agent)
        agent_patches.append(agent)
    
    # Add goal markers
    goal_markers = []
    for i in range(num_agents):
        goal_pos = goals[i].cpu().numpy()
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
    ax.set_title('Multi-Agent Navigation with GCBF+BTN', fontsize=14)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    # Add time and CBF value text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, ha='left', va='top')
    cbf_text = ax.text(0.02, 0.94, '', transform=ax.transAxes, ha='left', va='top')
    
    def init():
        """Initialize the animation."""
        for i, patch in enumerate(agent_patches):
            patch.center = all_states[0, i, 0], all_states[0, i, 1]
        
        for line in trajectory_lines:
            line.set_data([], [])
        
        time_text.set_text('')
        cbf_text.set_text('')
        
        return agent_patches + trajectory_lines + [time_text, cbf_text]
    
    def animate(frame):
        """Update the animation for each frame."""
        # Update agent positions
        for i, patch in enumerate(agent_patches):
            patch.center = all_states[frame, i, 0], all_states[frame, i, 1]
        
        # Update trajectory lines
        for i, line in enumerate(trajectory_lines):
            line.set_data(all_states[:frame+1, i, 0], all_states[:frame+1, i, 1])
        
        # Update text information
        time_text.set_text(f'Time step: {frame}')
        
        if frame < len(all_cbf_values):
            min_cbf = all_cbf_values[frame].min()
            cbf_text.set_text(f'Min CBF value: {min_cbf:.4f}')
        
        return agent_patches + trajectory_lines + [time_text, cbf_text]
    
    # Create animation
    num_frames = len(all_states)
    animation = FuncAnimation(fig, animate, frames=num_frames, init_func=init, 
                              interval=100, blit=True)
    
    # Save animation if requested
    if save_path:
        animation.save(save_path, writer='ffmpeg', fps=10, dpi=200)
        print(f"Animation saved to {save_path}")
    
    plt.close()
    return animation


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize trained BPTT models")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing trained models")
    parser.add_argument("--step", type=int, default=None, help="Model step to visualize (default: latest)")
    parser.add_argument("--output", type=str, default=None, help="Path to save the animation")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()
    
    # Set up the device
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    config_path = Path(args.model_dir) / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine which model step to use
    model_dir = Path(args.model_dir) / "models"
    if args.step is None:
        # Find the latest step
        steps = [int(d.name) for d in model_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not steps:
            raise ValueError(f"No model checkpoints found in {model_dir}")
        step = max(steps)
    else:
        step = args.step
    
    print(f"Using model checkpoint from step {step}")
    
    # Create neural networks
    policy_network = PolicyGNN(
        node_dim=config['networks']['policy']['node_dim'],
        edge_dim=config['networks']['policy']['edge_dim'],
        action_dim=2,
        hidden_dim=config['networks']['policy']['hidden_dim'],
        n_layers=config['networks']['policy']['n_layers']
    ).to(device)
    
    cbf_network = CBFGNN(
        node_dim=config['networks']['cbf']['node_dim'],
        edge_dim=config['networks']['cbf']['edge_dim'],
        hidden_dim=config['networks']['cbf']['hidden_dim'],
        n_layers=config['networks']['cbf']['n_layers']
    ).to(device)
    
    # Load the trained models
    policy_path = model_dir / str(step) / "policy.pt"
    cbf_path = model_dir / str(step) / "cbf.pt"
    
    policy_network.load_state_dict(torch.load(policy_path, map_location=device))
    cbf_network.load_state_dict(torch.load(cbf_path, map_location=device))
    
    # Create simulator
    simulator = DifferentiableDoubleIntegrator(
        dt=config['env']['dt'],
        mass=config['env']['mass']
    ).to(device)
    
    # Output path
    if args.output is None:
        output_path = Path(args.model_dir) / f"visualization_step_{step}.mp4"
    else:
        output_path = Path(args.output)
    
    # Run visualization
    animation = visualize_trajectory(
        policy_network=policy_network,
        cbf_network=cbf_network,
        simulator=simulator,
        device=device,
        config=config,
        save_path=str(output_path)
    )
    
    print(f"Visualization completed and saved to {output_path}")


if __name__ == "__main__":
    main() 