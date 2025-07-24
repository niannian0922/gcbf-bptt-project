import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm

from ..env.differentiable_simulator import (
    DifferentiableDoubleIntegrator, 
    DifferentiableAgentCollision,
    DifferentiableObstacleCollision,
    DifferentiableGraphBuilder
)

from .bptt_utils import (
    initialize_states_and_goals,
    build_graph_features,
    calculate_cbf_derivative,
    check_cbf_condition
)


class BPTTTrainer:
    """
    A trainer that implements Backpropagation Through Time (BPTT) for end-to-end optimization
    of both the policy and CBF networks through a differentiable physics simulator.
    
    This trainer eliminates the need for Q-learning, expert policies, and replay buffers
    by directly optimizing both networks using gradients through the simulator.
    """
    
    def __init__(
        self,
        policy_network,
        cbf_network,
        num_agents,
        area_size,
        log_dir,
        device,
        params
    ):
        self.policy_network = policy_network.to(device)
        self.cbf_network = cbf_network.to(device)
        self.num_agents = num_agents
        self.area_size = area_size
        self.log_dir = log_dir
        self.device = device
        self.params = params
        
        # Create directories for logging
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.model_dir = os.path.join(log_dir, 'models')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Initialize differentiable components
        self.simulator = DifferentiableDoubleIntegrator(
            dt=params['dt'],
            mass=params['mass']
        ).to(device)
        
        self.graph_builder = DifferentiableGraphBuilder(
            comm_radius=params['comm_radius']
        ).to(device)
        
        self.agent_collision = DifferentiableAgentCollision(
            agent_radius=params['car_radius']
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            list(self.policy_network.parameters()) + 
            list(self.cbf_network.parameters()),
            lr=params['learning_rate']
        )
        
        # Initialize learning rate scheduler if specified
        if params.get('use_lr_scheduler', False):
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=params['lr_step_size'],
                gamma=params['lr_gamma']
            )
        else:
            self.scheduler = None
        
        # Initialize parameters for running
        self.steps = params['training_steps']
        self.save_interval = params['save_interval']
        self.eval_interval = params['eval_interval']
        self.horizon_length = params['horizon_length']
        
        # Initialize loss weights
        self.goal_weight = params['goal_weight']
        self.safety_weight = params['safety_weight']  
        self.control_weight = params['control_weight']
        self.cbf_alpha = params['cbf_alpha']  # For the CBF condition: h_dot + alpha * h >= 0
    
    def initialize_scenario(self):
        """
        Initialize a scenario with random start positions and goal positions
        for all agents.
        
        Returns:
            - states: Tensor of shape [num_agents, 4] with initial states
            - goals: Tensor of shape [num_agents, 2] with goal positions
        """
        # Use the utility function to generate valid states and goals
        states, goals = initialize_states_and_goals(
            num_agents=self.num_agents,
            state_dim=4,  # x, y, vx, vy
            area_size=self.area_size,
            min_dist=4 * self.params['car_radius'],
            max_travel=None,
            device=self.device
        )
        
        return states, goals
    
    def train(self):
        """
        Main training loop implementing BPTT optimization.
        """
        print("Starting BPTT training...")
        wandb.init(name=self.params['run_name'], project='gcbf+', dir=self.log_dir)
        
        start_time = time.time()
        
        pbar = tqdm(total=self.steps)
        for step in range(self.steps):
            # Zero gradients before each backpropagation pass
            self.optimizer.zero_grad()
            
            # Initialize scenario
            states, goals = self.initialize_scenario()
            current_states = states.clone().requires_grad_()
            
            # BPTT Rollout
            trajectory_states = []
            trajectory_actions = []
            safety_losses = []
            
            # Track the CBF values over the trajectory (for logging)
            cbf_values = []
            collision_measures = []
            
            # Run forward simulation and collect trajectory data
            for t in range(self.params['horizon_length']):
                # Save current state
                trajectory_states.append(current_states)
                
                # Build graph features for the current state
                graph_features = build_graph_features(
                    states=current_states,
                    goals=goals,
                    sensing_radius=self.params['comm_radius'],
                    device=self.device
                )
                
                # Get actions from policy network
                actions = self.policy_network(graph_features)
                trajectory_actions.append(actions)
                
                # Compute CBF values for current state
                h_vals = self.cbf_network(graph_features)
                cbf_values.append(h_vals.mean().item())
                
                # Simulate forward to get the next state
                next_states = self.simulator(current_states, actions)
                
                # Build graph for next state
                next_graph_features = build_graph_features(
                    states=next_states,
                    goals=goals,
                    sensing_radius=self.params['comm_radius'],
                    device=self.device
                )
                
                # Compute CBF values for next state
                h_vals_next = self.cbf_network(next_graph_features)
                
                # Calculate CBF derivative approximation
                h_dot_approx = calculate_cbf_derivative(h_vals, h_vals_next, self.params['dt'])
                
                # Check CBF condition and compute safety loss
                cbf_violation = check_cbf_condition(h_vals, h_dot_approx, self.cbf_alpha)
                safety_loss_t = torch.mean(cbf_violation)  # Mean over all agents
                safety_losses.append(safety_loss_t)
                
                # Compute collision measure (for logging)
                collision_measure = self.agent_collision(next_states[:, :2])
                collision_measures.append(collision_measure.mean().item())
                
                # Update current state for next iteration
                current_states = next_states
            
            # Stack trajectory data for loss computation
            trajectory_states = torch.stack(trajectory_states)
            trajectory_actions = torch.stack(trajectory_actions)
            
            # Compute goal reaching loss (distance to goal at final state)
            final_positions = current_states[:, :2]
            goal_loss = torch.mean((final_positions - goals) ** 2)
            
            # Compute control effort loss
            control_effort = torch.mean(trajectory_actions ** 2)
            
            # Compute total safety loss
            safety_losses = torch.stack(safety_losses)
            total_safety_loss = torch.sum(safety_losses)
            
            # Compute total loss as weighted sum
            total_loss = (
                self.goal_weight * goal_loss +
                self.safety_weight * total_safety_loss +
                self.control_weight * control_effort
            )
            
            # Backpropagate loss through the entire computation graph
            total_loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_network.parameters()) + list(self.cbf_network.parameters()),
                self.params['max_grad_norm']
            )
            
            # Update parameters
            self.optimizer.step()
            
            # Update learning rate if scheduler is enabled
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics for wandb
            metrics = {
                "train/total_loss": total_loss.item(),
                "train/goal_loss": goal_loss.item(),
                "train/safety_loss": total_safety_loss.item(),
                "train/control_loss": control_effort.item(),
                "train/mean_cbf_value": np.mean(cbf_values),
                "train/mean_collision": np.mean(collision_measures),
                "train/lr": self.optimizer.param_groups[0]['lr'],
                "step": step,
            }
            wandb.log(metrics)
            
            # Evaluation and model saving
            if (step + 1) % self.eval_interval == 0:
                eval_metrics = self.evaluate()
                wandb.log(eval_metrics)
                
                # Print progress
                time_elapsed = time.time() - start_time
                print(f"Step {step+1}/{self.steps}, Time: {time_elapsed:.2f}s")
                print(f"  Total Loss: {total_loss.item():.4f}")
                print(f"  Goal Loss: {goal_loss.item():.4f}")
                print(f"  Safety Loss: {total_safety_loss.item():.4f}")
                print(f"  Control Loss: {control_effort.item():.4f}")
                print(f"  Mean CBF Value: {np.mean(cbf_values):.4f}")
                print(f"  Mean Collision: {np.mean(collision_measures):.4f}")
                print(f"  Evaluation Success Rate: {eval_metrics['eval/success_rate']:.2f}")
                print(f"  Evaluation Collision Rate: {eval_metrics['eval/collision_rate']:.2f}")
            
            # Save models
            if (step + 1) % self.save_interval == 0:
                self.save_models(step + 1)
            
            pbar.update(1)
        
        pbar.close()
        print("Training completed.")
        
        # Save final models
        self.save_models(self.steps)
    
    def evaluate(self, num_episodes=10):
        """
        Evaluate the current policy and CBF networks.
        """
        success_count = 0
        collision_count = 0
        avg_goal_distance = 0
        avg_min_cbf = 0
        
        # Set networks to evaluation mode
        self.policy_network.eval()
        self.cbf_network.eval()
        
        for _ in range(num_episodes):
            # Initialize scenario
            states, goals = self.initialize_scenario()
            
            # Run episode without gradient tracking
            with torch.no_grad():
                min_cbf = float('inf')
                
                # Run forward simulation
                for _ in range(self.params['eval_horizon']):
                    # Build graph features
                    graph_features = build_graph_features(
                        states=states,
                        goals=goals,
                        sensing_radius=self.params['comm_radius'],
                        device=self.device
                    )
                    
                    # Get CBF values
                    h_vals = self.cbf_network(graph_features)
                    min_cbf_val = h_vals.min().item()
                    min_cbf = min(min_cbf, min_cbf_val)
                    
                    # Get actions from policy network
                    actions = self.policy_network(graph_features)
                    
                    # Step simulation
                    states = self.simulator(states, actions)
                    
                    # Check for collisions
                    collision_measure = self.agent_collision(states[:, :2])
                    if collision_measure.max().item() > 0:
                        collision_count += 1
                        break
                
                # Check if goals are reached
                final_distances = torch.norm(states[:, :2] - goals, dim=1)
                avg_distance = final_distances.mean().item()
                avg_goal_distance += avg_distance
                
                # Count as success if all agents are close enough to their goals
                if torch.all(final_distances < 2 * self.params['car_radius']):
                    success_count += 1
                
                # Track minimum CBF value
                avg_min_cbf += min_cbf
        
        # Set networks back to training mode
        self.policy_network.train()
        self.cbf_network.train()
        
        # Compute average metrics
        success_rate = success_count / num_episodes
        collision_rate = collision_count / num_episodes
        avg_goal_distance /= num_episodes
        avg_min_cbf /= num_episodes
        
        return {
            "eval/success_rate": success_rate,
            "eval/collision_rate": collision_rate,
            "eval/avg_goal_distance": avg_goal_distance,
            "eval/avg_min_cbf": avg_min_cbf,
        }
    
    def save_models(self, step):
        """
        Save the policy and CBF network models.
        """
        step_dir = os.path.join(self.model_dir, str(step))
        if not os.path.exists(step_dir):
            os.makedirs(step_dir)
        
        # Save policy network
        policy_path = os.path.join(step_dir, "policy.pt")
        torch.save(self.policy_network.state_dict(), policy_path)
        
        # Save CBF network
        cbf_path = os.path.join(step_dir, "cbf.pt")
        torch.save(self.cbf_network.state_dict(), cbf_path)
        
        print(f"Models saved at step {step}")
    
    def load_models(self, step):
        """
        Load the policy and CBF network models.
        """
        step_dir = os.path.join(self.model_dir, str(step))
        
        # Load policy network
        policy_path = os.path.join(step_dir, "policy.pt")
        self.policy_network.load_state_dict(torch.load(policy_path, map_location=self.device))
        
        # Load CBF network
        cbf_path = os.path.join(step_dir, "cbf.pt")
        self.cbf_network.load_state_dict(torch.load(cbf_path, map_location=self.device))
        
        print(f"Models loaded from step {step}") 