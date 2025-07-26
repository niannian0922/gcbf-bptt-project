import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, List, Union, Callable

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not found. Training will proceed without logging to wandb.")

from ..env.base_env import BaseEnv, EnvState
from ..env.multi_agent_env import MultiAgentEnv, MultiAgentState
from ..policy.bptt_policy import BPTTPolicy


class BPTTTrainer:
    """
    A trainer that implements Backpropagation Through Time (BPTT) for end-to-end optimization
    of both the policy and CBF networks through a differentiable physics simulator.
    
    This trainer eliminates the need for Q-learning, expert policies, and replay buffers
    by directly optimizing both networks using gradients through the simulator.
    """
    
    def __init__(
        self,
        env: BaseEnv,
        policy_network: nn.Module,
        cbf_network: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the BPTT trainer.
        
        Args:
            env: Differentiable environment instance
            policy_network: Policy network to train
            cbf_network: Optional CBF network for safety
            optimizer: Optional optimizer (will create default if None)
            config: Configuration dictionary
        """
        # Store environment and networks
        self.env = env
        self.policy_network = policy_network
        self.cbf_network = cbf_network
        
        # Get device from policy network
        self.device = next(policy_network.parameters()).device
        
        # Set default configuration if none provided
        self.config = {} if config is None else config
        
        # Extract parameters from config
        self.log_dir = self.config.get('log_dir', 'logs/bptt')
        self.run_name = self.config.get('run_name', 'BPTT_Run')
        self.num_agents = self.config.get('num_agents', 8)
        self.area_size = self.config.get('area_size', 1.0)
        
        # Training parameters
        self.training_steps = self.config.get('training_steps', 10000)
        self.eval_interval = self.config.get('eval_interval', 100)
        self.save_interval = self.config.get('save_interval', 1000)
        self.horizon_length = self.config.get('horizon_length', 50)
        self.eval_horizon = self.config.get('eval_horizon', 100)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)
        
        # Loss weights
        self.goal_weight = self.config.get('goal_weight', 1.0)
        self.safety_weight = self.config.get('safety_weight', 10.0)
        self.control_weight = self.config.get('control_weight', 0.1)
        self.cbf_alpha = self.config.get('cbf_alpha', 1.0)
        
        # Create directories for logging
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.model_dir = os.path.join(self.log_dir, 'models')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Initialize optimizer if not provided
        if optimizer is None:
            params = list(self.policy_network.parameters())
            if self.cbf_network is not None:
                params += list(self.cbf_network.parameters())
                
            self.optimizer = optim.Adam(
                params,
                lr=self.config.get('learning_rate', 0.001)
            )
        else:
            self.optimizer = optimizer
        
        # Initialize learning rate scheduler if specified
        if self.config.get('use_lr_scheduler', False):
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('lr_step_size', 2000),
                gamma=self.config.get('lr_gamma', 0.5)
            )
        else:
            self.scheduler = None
    
    def initialize_scenario(self, batch_size: int = 1) -> MultiAgentState:
        """
        Initialize a new scenario with random initial states and goals.
        
        Args:
            batch_size: Number of parallel environments to initialize
            
        Returns:
            Environment state
        """
        # Use the environment's reset method to initialize states
        return self.env.reset(batch_size=batch_size, randomize=True)
    
    def train(self) -> None:
        """
        Main training loop implementing BPTT optimization.
        """
        print(f"Starting BPTT training with configuration:")
        print(f"  Run name: {self.run_name}")
        print(f"  Steps: {self.training_steps}")
        print(f"  Horizon: {self.horizon_length}")
        print(f"  Log dir: {self.log_dir}")
        
        # Initialize wandb in offline mode
        if WANDB_AVAILABLE:
            wandb.init(name=self.run_name, project='gcbf-bptt', dir=self.log_dir, config=self.config, mode="offline")
        
        start_time = time.time()
        
        pbar = tqdm(total=self.training_steps)
        for step in range(self.training_steps):
            # Train mode
            self.policy_network.train()
            if self.cbf_network is not None:
                self.cbf_network.train()
            
            # Zero gradients before each backpropagation pass
            self.optimizer.zero_grad()
            
            # Initialize scenario
            state = self.initialize_scenario()
            
            # BPTT Rollout
            trajectory_states = []
            trajectory_actions = []
            trajectory_rewards = []
            trajectory_costs = []
            safety_losses = []
            
            # Run forward simulation and collect trajectory data
            current_state = state
            for t in range(self.horizon_length):
                # Save current state
                trajectory_states.append(current_state)
                
                # Get observations from state
                observations = self.env.get_observation(current_state)
                
                # Get actions from policy network
                actions = self.policy_network(observations)
                # Store a detached copy for backprop later
                trajectory_actions.append(actions.clone())
                
                # Apply safety filter if CBF network is provided
                if self.cbf_network is not None:
                    cbf_values = self.cbf_network(observations)
                    
                    # Compute safety loss based on CBF values
                    # Negative values indicate unsafe states
                    safety_loss = torch.mean(torch.relu(-cbf_values))
                    safety_losses.append(safety_loss)
                
                # Take a step in the environment
                step_result = self.env.step(current_state, actions)
                next_state = step_result.next_state
                rewards = step_result.reward
                costs = step_result.cost
                
                # Save reward and cost (detach to prevent modification during backprop)
                trajectory_rewards.append(rewards.clone())
                trajectory_costs.append(costs.clone())
                
                # Update current state for next iteration (detach to prevent inplace modifications)
                current_state = next_state
            
            # Compute losses
            
            # Goal reaching loss (using rewards)
            if trajectory_rewards:
                stacked_rewards = torch.stack(trajectory_rewards)
                goal_loss = -torch.mean(stacked_rewards)
            else:
                # Fallback: Use distance to goal
                goal_distances = self.env.get_goal_distance(current_state)
                goal_loss = torch.mean(goal_distances)
            
            # Control effort loss
            stacked_actions = torch.stack(trajectory_actions)
            control_effort = torch.mean(stacked_actions ** 2)
            
            # Safety loss
            if safety_losses:
                stacked_safety = torch.stack(safety_losses)
                total_safety_loss = torch.mean(stacked_safety)
            else:
                # If no CBF network, use environment costs
                stacked_costs = torch.stack(trajectory_costs)
                total_safety_loss = torch.mean(stacked_costs)
            
            # Compute total loss as weighted sum
            total_loss = (
                self.goal_weight * goal_loss +
                self.safety_weight * total_safety_loss +
                self.control_weight * control_effort
            )
            
            # Backpropagate loss through the entire computation graph
            # Always use retain_graph=True for BPTT to prevent issues with the computation graph
            total_loss.backward(retain_graph=True)
            
            # Clip gradients to prevent exploding gradients
            parameters = list(self.policy_network.parameters())
            if self.cbf_network is not None:
                parameters += list(self.cbf_network.parameters())
                
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
            
            # Update parameters
            self.optimizer.step()
            
            # Update learning rate if scheduler is enabled
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Compute metrics for logging
            metrics = {
                "train/total_loss": total_loss.item(),
                "train/goal_loss": goal_loss.item(),
                "train/safety_loss": total_safety_loss.item(),
                "train/control_loss": control_effort.item(),
                "train/lr": self.optimizer.param_groups[0]['lr'],
                "step": step,
            }
            
            # Log metrics
            if WANDB_AVAILABLE:
                wandb.log(metrics)
            
            # Evaluation and model saving
            if (step + 1) % self.eval_interval == 0:
                eval_metrics = self.evaluate()
                if WANDB_AVAILABLE:
                    wandb.log(eval_metrics)
                
                # Print progress
                time_elapsed = time.time() - start_time
                print(f"\nStep {step+1}/{self.training_steps}, Time: {time_elapsed:.2f}s")
                print(f"  Total Loss: {total_loss.item():.4f}")
                print(f"  Goal Loss: {goal_loss.item():.4f}")
                print(f"  Safety Loss: {total_safety_loss.item():.4f}")
                print(f"  Evaluation Success Rate: {eval_metrics['eval/success_rate']:.2f}")
                print(f"  Evaluation Collision Rate: {eval_metrics['eval/collision_rate']:.2f}")
            
            # Save models
            if (step + 1) % self.save_interval == 0:
                self.save_models(step + 1)
            
            pbar.update(1)
        
        pbar.close()
        print("Training completed.")
        
        # Save final models
        self.save_models(self.training_steps)
        
        # Final evaluation
        final_metrics = self.evaluate(num_episodes=20)
        print("\nFinal Evaluation Results:")
        print(f"  Success Rate: {final_metrics['eval/success_rate']:.2f}")
        print(f"  Collision Rate: {final_metrics['eval/collision_rate']:.2f}")
        print(f"  Avg Goal Distance: {final_metrics['eval/avg_goal_distance']:.4f}")
        
        return final_metrics
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the current policy and CBF networks.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        success_count = 0
        collision_count = 0
        avg_goal_distance = 0
        avg_min_cbf = float('inf')
        
        # Set networks to evaluation mode
        self.policy_network.eval()
        if self.cbf_network is not None:
            self.cbf_network.eval()
        
        for _ in range(num_episodes):
            # Initialize scenario
            state = self.initialize_scenario()
            
            # Run episode without gradient tracking
            with torch.no_grad():
                # Reset environment
                current_state = state
                
                # Run forward simulation
                for _ in range(self.eval_horizon):
                    # Get observations
                    observations = self.env.get_observation(current_state)
                    
                    # Get CBF values if available
                    if self.cbf_network is not None:
                        cbf_values = self.cbf_network(observations)
                        min_cbf_val = cbf_values.min().item()
                        avg_min_cbf = min(avg_min_cbf, min_cbf_val)
                    
                    # Get actions from policy network
                    actions = self.policy_network(observations)
                    
                    # Step simulation
                    step_result = self.env.step(current_state, actions)
                    next_state = step_result.next_state
                    
                    # Check for collisions
                    if torch.any(step_result.cost > 0):
                        collision_count += 1
                        break
                    
                    # Update state
                    current_state = next_state
                
                # Check if goals are reached (use goal distance from environment)
                goal_distances = self.env.get_goal_distance(current_state)
                avg_distance = goal_distances.mean().item()
                avg_goal_distance += avg_distance
                
                # Count as success if all agents are close to their goals
                if torch.all(goal_distances < self.env.agent_radius * 2):
                    success_count += 1
        
        # Set networks back to training mode
        self.policy_network.train()
        if self.cbf_network is not None:
            self.cbf_network.train()
        
        # Compute average metrics
        success_rate = success_count / num_episodes
        collision_rate = collision_count / num_episodes
        avg_goal_distance /= num_episodes
        
        # Prepare evaluation metrics
        metrics = {
            "eval/success_rate": success_rate,
            "eval/collision_rate": collision_rate,
            "eval/avg_goal_distance": avg_goal_distance,
        }
        
        # Add CBF metrics if available
        if self.cbf_network is not None and avg_min_cbf != float('inf'):
            metrics["eval/avg_min_cbf"] = avg_min_cbf
        
        return metrics
    
    def save_models(self, step: int) -> None:
        """
        Save the policy and CBF network models.
        
        Args:
            step: Current training step
        """
        step_dir = os.path.join(self.model_dir, str(step))
        if not os.path.exists(step_dir):
            os.makedirs(step_dir)
        
        # Save policy network
        policy_path = os.path.join(step_dir, "policy.pt")
        torch.save(self.policy_network.state_dict(), policy_path)
        
        # Save CBF network if available
        if self.cbf_network is not None:
            cbf_path = os.path.join(step_dir, "cbf.pt")
            torch.save(self.cbf_network.state_dict(), cbf_path)
        
        # Save optimizer state
        optim_path = os.path.join(step_dir, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optim_path)
        
        # Save configuration
        config_path = os.path.join(step_dir, "config.pt")
        torch.save(self.config, config_path)
        
        print(f"Models saved at step {step}")
    
    def load_models(self, step: int) -> None:
        """
        Load the policy and CBF network models.
        
        Args:
            step: Training step to load from
        """
        step_dir = os.path.join(self.model_dir, str(step))
        
        if not os.path.exists(step_dir):
            raise FileNotFoundError(f"No saved models found at step {step}")
        
        # Load policy network
        policy_path = os.path.join(step_dir, "policy.pt")
        if os.path.exists(policy_path):
            self.policy_network.load_state_dict(torch.load(policy_path))
            print(f"Policy network loaded from {policy_path}")
        
        # Load CBF network if available
        if self.cbf_network is not None:
            cbf_path = os.path.join(step_dir, "cbf.pt")
            if os.path.exists(cbf_path):
                self.cbf_network.load_state_dict(torch.load(cbf_path))
                print(f"CBF network loaded from {cbf_path}")
        
        # Load optimizer state
        optim_path = os.path.join(step_dir, "optimizer.pt")
        if os.path.exists(optim_path):
            self.optimizer.load_state_dict(torch.load(optim_path))
            print(f"Optimizer state loaded from {optim_path}")
        
        print(f"Models loaded from step {step}") 