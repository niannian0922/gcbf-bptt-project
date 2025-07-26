import torch
import yaml
import matplotlib.pyplot as plt
from gcbfplus.env import DoubleIntegratorEnv

def test_obstacle_config():
    # Load config file with obstacles
    with open('config/alpha_low_obs.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    print("Creating environment with obstacles...")
    env = DoubleIntegratorEnv(config['env'])
    
    # Reset environment and check for obstacles
    print("Resetting environment...")
    state = env.reset(batch_size=1)
    
    # Check if obstacles exist in state
    if state.obstacles is not None:
        print(f"Environment has obstacles: YES")
        print(f"Number of obstacles: {state.obstacles.shape[1]}")
        
        # Print obstacle details
        obstacles = state.obstacles[0].detach().cpu().numpy()
        for i, obs in enumerate(obstacles):
            print(f"Obstacle {i+1}: position={obs[:-1]}, radius={obs[-1]}")
    else:
        print("Environment has obstacles: NO")
    
    # Render the environment to visualize obstacles
    print("\nRendering environment with obstacles...")
    fig = env.render(state)
    plt.savefig("obstacle_test.png")
    print("Saved visualization to obstacle_test.png")
    
    # Test GCBF safety layer with obstacles
    print("\nTesting safety layer with random actions...")
    random_actions = torch.rand(1, env.num_agents, env.action_dim, device=state.positions.device) * 2 - 1
    safe_actions = env.apply_safety_layer(state, random_actions)
    
    # Step the environment with safe actions
    step_result = env.step(state, safe_actions)
    
    # Check for collisions
    collisions = env.check_collision(step_result.next_state)
    print(f"Collisions after step: {collisions.any().item()}")
    
    # Check obstacle collisions specifically
    obstacle_collisions = env.check_obstacle_collisions(step_result.next_state)
    print(f"Obstacle collisions after step: {obstacle_collisions.any().item()}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_obstacle_config() 