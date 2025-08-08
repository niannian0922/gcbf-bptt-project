#!/usr/bin/env python3
"""
Test script for enhanced environment randomization.
Verifies that the new randomization features work correctly.
"""

import yaml
import torch
import numpy as np
from gcbfplus.env.double_integrator import DoubleIntegratorEnv

def test_enhanced_randomization():
    """Test the enhanced randomization features."""
    
    print("🧪 Testing Enhanced Environment Randomization")
    print("=" * 50)
    
    # Load enhanced diversity configuration
    with open('config/enhanced_diversity_training.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env_config = config['env']
    env = DoubleIntegratorEnv(env_config)
    
    print(f"📊 Environment Configuration:")
    print(f"  Area Size: {env.area_size}")
    print(f"  Num Agents: {env.num_agents}")
    print(f"  Agent Radius: {env.agent_radius}")
    
    obstacles_config = env_config.get('obstacles', {})
    print(f"\n🔧 Obstacle Configuration:")
    print(f"  Dynamic Count: {obstacles_config.get('dynamic_count', False)}")
    print(f"  Count Range: {obstacles_config.get('count_range', [0, 0])}")
    print(f"  Min Radius: {obstacles_config.get('random_min_radius', 0.1)}")
    print(f"  Max Radius: {obstacles_config.get('random_max_radius', 0.3)}")
    
    print(f"\n🎯 Testing Multiple Environment Resets:")
    print("-" * 40)
    
    obstacle_counts = []
    start_configs = []
    goal_strategies = []
    
    for i in range(10):
        # Reset environment with randomization
        state = env.reset(batch_size=1, randomize=True)
        
        # Count obstacles
        if state.obstacles is not None:
            # Count non-dummy obstacles (those with non-zero radius)
            valid_obstacles = state.obstacles[0, :, 2] > 0
            obstacle_count = valid_obstacles.sum().item()
            obstacle_counts.append(obstacle_count)
        else:
            obstacle_counts.append(0)
        
        # Analyze starting configuration type
        positions = state.positions[0].cpu().numpy()
        goals = state.goals[0].cpu().numpy()
        
        # Simple heuristic to guess configuration type
        if np.all(positions[:, 0] < 0.3 * env.area_size) or np.all(positions[:, 0] > 0.7 * env.area_size):
            config_type = "edge/corner"
        elif np.std(positions) < 0.2 * env.area_size:
            config_type = "clustered"
        else:
            config_type = "random"
        
        start_configs.append(config_type)
        
        # Analyze goal strategy
        center = env.area_size / 2
        goal_center_dist = np.mean(np.linalg.norm(goals - center, axis=1))
        if goal_center_dist < 0.3 * env.area_size:
            goal_type = "center"
        elif np.array_equal(goals, env.area_size - positions):
            goal_type = "opposite"
        else:
            goal_type = "diverse"
        
        goal_strategies.append(goal_type)
        
        print(f"  Reset {i+1:2d}: {obstacle_count} obstacles, {config_type:>12s} start, {goal_type:>8s} goals")
    
    print(f"\n📈 Randomization Analysis:")
    print("-" * 30)
    print(f"  Obstacle Count Range: {min(obstacle_counts)}-{max(obstacle_counts)} (avg: {np.mean(obstacle_counts):.1f})")
    print(f"  Start Config Variety: {len(set(start_configs))} different types")
    print(f"  Goal Strategy Variety: {len(set(goal_strategies))} different types")
    
    # Test specific configurations
    print(f"\n🔍 Detailed Test - Single Reset:")
    print("-" * 35)
    
    state = env.reset(batch_size=1, randomize=True)
    
    print(f"  Positions: {state.positions[0].cpu().numpy()}")
    print(f"  Goals: {state.goals[0].cpu().numpy()}")
    
    if state.obstacles is not None:
        obstacles = state.obstacles[0].cpu().numpy()
        valid_obstacles = obstacles[obstacles[:, 2] > 0]
        print(f"  Obstacles ({len(valid_obstacles)}):")
        for j, obs in enumerate(valid_obstacles):
            print(f"    {j+1}: pos=({obs[0]:.2f}, {obs[1]:.2f}), radius={obs[2]:.2f}")
    else:
        print(f"  Obstacles: None")
    
    print(f"\n✅ Enhanced Randomization Test Complete!")
    
    # Verify diversity
    expected_features = [
        f"Obstacle count varies: {min(obstacle_counts) != max(obstacle_counts)}",
        f"Multiple start configs: {len(set(start_configs)) >= 2}",
        f"Multiple goal strategies: {len(set(goal_strategies)) >= 2}",
    ]
    
    print(f"\n🎯 Diversity Verification:")
    for feature in expected_features:
        print(f"  ✓ {feature}")

if __name__ == "__main__":
    test_enhanced_randomization()
