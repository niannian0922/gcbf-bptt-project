#!/usr/bin/env python3
"""
Test Script for Episode Logging System

This script tests the complete data logging and plotting pipeline using
a simple test scenario.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gcbfplus.utils.episode_logger import EpisodeLogger, compute_min_distances_to_obstacles, compute_goal_distances
from plot_results import EpisodePlotter


def create_synthetic_episode_data():
    """Create synthetic episode data for testing."""
    print("🧪 Creating synthetic episode data...")
    
    # Episode parameters
    n_agents = 3
    n_steps = 100
    batch_size = 1
    pos_dim = 2
    action_dim = 2
    
    # Create logger
    log_dir = "test_episode_logs"
    logger = EpisodeLogger(log_dir=log_dir, prefix="test_episode")
    
    # Define environment parameters
    obstacles = np.array([
        [0.5, 0.5, 0.2],   # x, y, radius
        [-0.3, 0.8, 0.15],
        [1.2, -0.4, 0.18]
    ])
    
    goals = np.array([[[1.5, 1.5], [-1.5, 1.5], [0.0, -1.5]]])  # [batch, n_agents, pos_dim]
    
    # Start episode
    episode_id = logger.start_episode(
        batch_size=batch_size,
        n_agents=n_agents,
        obstacles=torch.from_numpy(obstacles),
        goals=torch.from_numpy(goals),
        safety_radius=0.15,
        area_size=2.0
    )
    
    print(f"📊 Started episode: {episode_id}")
    
    # Generate trajectory data
    # Start positions (agents start from different corners)
    start_positions = np.array([
        [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]]  # [batch, n_agents, pos_dim]
    ])
    
    positions = start_positions.copy()
    velocities = np.zeros_like(positions)
    
    # Simulate episode
    collision_occurred = False
    success_achieved = False
    
    for step in range(n_steps):
        # Simple controller: move towards goal with obstacle avoidance
        actions = np.zeros((batch_size, n_agents, action_dim))
        
        for agent_idx in range(n_agents):
            agent_pos = positions[0, agent_idx]
            goal_pos = goals[0, agent_idx]
            
            # Basic goal-seeking behavior
            direction_to_goal = goal_pos - agent_pos
            distance_to_goal = np.linalg.norm(direction_to_goal)
            
            if distance_to_goal > 0.1:
                # Normalize and scale
                direction_to_goal = direction_to_goal / distance_to_goal
                actions[0, agent_idx] = direction_to_goal * 0.5
                
                # Simple obstacle avoidance
                for obs in obstacles:
                    obs_pos = obs[:2]
                    obs_radius = obs[2]
                    
                    dist_to_obs = np.linalg.norm(agent_pos - obs_pos)
                    if dist_to_obs < obs_radius + 0.3:  # Safety margin
                        # Add repulsive force
                        repulsion = (agent_pos - obs_pos) / max(dist_to_obs, 0.01)
                        actions[0, agent_idx] += repulsion * 0.3
        
        # Physics simulation (simple Euler integration)
        dt = 0.05
        mass = 0.1
        
        # Update velocities and positions
        accelerations = actions / mass
        velocities += accelerations * dt
        positions += velocities * dt
        
        # Add some noise for realism
        positions += np.random.normal(0, 0.005, positions.shape)
        velocities += np.random.normal(0, 0.01, velocities.shape)
        
        # Compute additional data
        min_distances = compute_min_distances_to_obstacles(positions, obstacles)
        goal_distances = compute_goal_distances(positions, goals)
        
        # Generate synthetic CBF and alpha values
        h_values = np.random.uniform(-0.5, 2.0, (batch_size, n_agents, 1))
        alpha_values = np.random.uniform(0.5, 2.0, (batch_size, n_agents, 1))
        
        # Compute rewards and costs
        rewards = -goal_distances * 0.1  # Negative distance as reward
        costs = np.maximum(0, 0.15 - min_distances)  # Cost if too close to obstacles
        
        # Log step data
        logger.log_step(
            positions=torch.from_numpy(positions),
            velocities=torch.from_numpy(velocities),
            actions=torch.from_numpy(actions),
            raw_actions=torch.from_numpy(actions + np.random.normal(0, 0.05, actions.shape)),
            alpha_values=torch.from_numpy(alpha_values),
            h_values=torch.from_numpy(h_values),
            min_distances=torch.from_numpy(min_distances),
            goal_distances=torch.from_numpy(goal_distances),
            rewards=torch.from_numpy(rewards),
            costs=torch.from_numpy(costs)
        )
        
        # Check for collision
        if np.any(costs > 0):
            collision_occurred = True
            break
        
        # Check for success
        if np.all(goal_distances < 0.2):
            success_achieved = True
            break
    
    # Determine episode status
    if collision_occurred:
        status = "COLLISION"
    elif success_achieved:
        status = "SUCCESS"
    else:
        status = "TIMEOUT"
    
    # End episode
    filename = logger.end_episode(status)
    
    print(f"✅ Episode completed: {status}")
    print(f"💾 Data saved to: {filename}")
    
    return filename


def test_plotting_system(episode_file):
    """Test the plotting system with episode data."""
    print(f"\n🎨 Testing plotting system with: {episode_file}")
    
    try:
        # Create plotter
        plotter = EpisodePlotter(episode_file)
        
        # Test individual plots
        import matplotlib.pyplot as plt
        
        print("   📈 Testing 3D trajectory plot...")
        fig_3d = plotter.plot_3d_trajectories()
        if fig_3d is not None:
            print("   ✅ 3D trajectory plot created successfully")
            plt.close(fig_3d)
        
        print("   📈 Testing safety distance plot...")
        fig_safety = plotter.plot_safety_distances()
        if fig_safety is not None:
            print("   ✅ Safety distance plot created successfully")
            plt.close(fig_safety)
        
        print("   📈 Testing CBF analysis plot...")
        fig_cbf = plotter.plot_cbf_analysis()
        if fig_cbf is not None:
            print("   ✅ CBF analysis plot created successfully")
            plt.close(fig_cbf)
        else:
            print("   ⚠️ CBF analysis plot skipped (expected for test data)")
        
        print("   📈 Testing comprehensive analysis plot...")
        fig_comp = plotter.plot_comprehensive_analysis()
        if fig_comp is not None:
            print("   ✅ Comprehensive analysis plot created successfully")
            plt.close(fig_comp)
        
        # Test save functionality
        print("   💾 Testing plot saving...")
        plot_dir = "test_episode_plots"
        plotter.save_all_plots(plot_dir)
        print(f"   ✅ All plots saved to: {plot_dir}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Plotting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plot_script_cli():
    """Test the command-line plot script."""
    print(f"\n🖥️ Testing plot_results.py CLI script...")
    
    # Find the most recent test episode file
    log_dir = "test_episode_logs"
    if not os.path.exists(log_dir):
        print("   ❌ No test episode logs found")
        return False
    
    # Get the newest .npz file
    episode_files = [f for f in os.listdir(log_dir) if f.endswith('.npz')]
    if not episode_files:
        print("   ❌ No episode files found")
        return False
    
    episode_files.sort()
    latest_file = os.path.join(log_dir, episode_files[-1])
    
    # Test the CLI script
    import subprocess
    try:
        cmd = [sys.executable, "plot_results.py", latest_file, "--save-plots", "--output-dir", "test_cli_plots"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ CLI plot script executed successfully")
            print(f"   📁 Output saved to: test_cli_plots")
            return True
        else:
            print(f"   ❌ CLI script failed with code {result.returncode}")
            print(f"   Error output: {result.stderr}")
            return False
    
    except subprocess.TimeoutExpired:
        print("   ⏰ CLI script timed out")
        return False
    except Exception as e:
        print(f"   ❌ CLI script test failed: {e}")
        return False


def cleanup_test_files():
    """Clean up test files."""
    print(f"\n🧹 Cleaning up test files...")
    
    import shutil
    
    # Directories to clean
    test_dirs = ["test_episode_logs", "test_episode_plots", "test_cli_plots"]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
                print(f"   ✅ Removed: {test_dir}")
            except Exception as e:
                print(f"   ⚠️ Failed to remove {test_dir}: {e}")


def main():
    """Main test function."""
    print("🧪 Testing Episode Logging and Plotting System")
    print("=" * 60)
    
    all_tests_passed = True
    
    try:
        # Test 1: Create synthetic episode data
        print("\n🔬 Test 1: Episode Data Creation")
        episode_file = create_synthetic_episode_data()
        
        if not os.path.exists(episode_file):
            print(f"❌ Episode file was not created: {episode_file}")
            all_tests_passed = False
        else:
            print(f"✅ Episode file created successfully")
        
        # Test 2: Plotting system
        print(f"\n🔬 Test 2: Plotting System")
        if not test_plotting_system(episode_file):
            all_tests_passed = False
        
        # Test 3: CLI script
        print(f"\n🔬 Test 3: CLI Plot Script")
        if not test_plot_script_cli():
            all_tests_passed = False
        
        # Final results
        print("\n" + "=" * 60)
        if all_tests_passed:
            print("🎉 All tests passed! Episode logging system is working correctly.")
            print("\n💡 You can now use the system for real evaluations:")
            print("   1. Use evaluate_with_logging.py for comprehensive evaluation")
            print("   2. Use plot_results.py for individual episode analysis")
            print("   3. Add 'enable_episode_logging: true' to your config files")
        else:
            print("❌ Some tests failed. Please check the error messages above.")
        
    except Exception as e:
        print(f"\n❌ Test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_tests_passed = False
    
    finally:
        # Ask user if they want to keep test files
        print(f"\n🗑️ Clean up test files? (y/n): ", end="")
        try:
            response = input().strip().lower()
            if response in ['y', 'yes', '']:
                cleanup_test_files()
            else:
                print("   Test files preserved for inspection")
        except (KeyboardInterrupt, EOFError):
            print("\n   Test files preserved")
    
    # Return exit code
    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
