#!/usr/bin/env python3

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class BottleneckRegion:
    """Define a bottleneck region for metric calculation"""
    x_min: float
    x_max: float
    y_min: float
    y_max: float

@dataclass
class BottleneckMetrics:
    """Comprehensive metrics for bottleneck scenario analysis"""
    throughput: float                    # Agents passing through per second
    velocity_fluctuation: float          # Std dev of speeds in bottleneck
    total_waiting_time: float           # Total time spent waiting (low speed)
    avg_bottleneck_time: float          # Average time spent in bottleneck region
    coordination_efficiency: float       # How efficiently agents coordinate
    collision_rate: float               # Collision rate in bottleneck
    completion_rate: float              # Percentage of agents reaching goals

class BottleneckAnalyzer:
    """Analyzer for bottleneck-specific multi-agent coordination metrics"""
    
    def __init__(self, config: Dict):
        """
        Initialize bottleneck analyzer.
        
        Args:
            config: Configuration dict with bottleneck_metrics section
        """
        bottleneck_config = config.get('bottleneck_metrics', {})
        
        # Bottleneck region definition
        region_config = bottleneck_config.get('bottleneck_region', {})
        self.bottleneck_region = BottleneckRegion(
            x_min=region_config.get('x_min', 0.9),
            x_max=region_config.get('x_max', 1.1),
            y_min=region_config.get('y_min', 0.8),
            y_max=region_config.get('y_max', 1.2)
        )
        
        # Analysis parameters
        self.throughput_window = bottleneck_config.get('throughput_window', 1.0)
        self.velocity_threshold = bottleneck_config.get('velocity_threshold', 0.1)
        self.dt = config.get('env', {}).get('dt', 0.03)
        
        # Storage for trajectory analysis
        self.trajectory_data = []
        
    def reset(self):
        """Reset analyzer for new episode"""
        self.trajectory_data = []
    
    def update(self, positions: torch.Tensor, velocities: torch.Tensor, 
               goals: torch.Tensor, time_step: int):
        """
        Update analyzer with new trajectory data.
        
        Args:
            positions: Agent positions [batch_size, num_agents, 2]
            velocities: Agent velocities [batch_size, num_agents, 2] 
            goals: Agent goals [batch_size, num_agents, 2]
            time_step: Current simulation time step
        """
        batch_size, num_agents, _ = positions.shape
        
        # Convert to numpy for easier processing
        pos_np = positions.detach().cpu().numpy()
        vel_np = velocities.detach().cpu().numpy()
        goals_np = goals.detach().cpu().numpy()
        
        # Store trajectory data
        step_data = {
            'time_step': time_step,
            'positions': pos_np,
            'velocities': vel_np,
            'goals': goals_np,
            'time': time_step * self.dt
        }
        
        self.trajectory_data.append(step_data)
    
    def is_in_bottleneck(self, position: np.ndarray) -> bool:
        """Check if position is within bottleneck region"""
        x, y = position[0], position[1]
        return (self.bottleneck_region.x_min <= x <= self.bottleneck_region.x_max and
                self.bottleneck_region.y_min <= y <= self.bottleneck_region.y_max)
    
    def compute_throughput(self) -> float:
        """
        Compute throughput: number of agents passing through bottleneck per second.
        
        Returns:
            Throughput rate (agents per second)
        """
        if len(self.trajectory_data) < 2:
            return 0.0
            
        # Track agents that have passed through the bottleneck
        total_passages = 0
        num_agents = self.trajectory_data[0]['positions'].shape[1]
        
        # For each agent, check if they passed through the bottleneck
        for agent_idx in range(num_agents):
            in_bottleneck_history = []
            
            for step_data in self.trajectory_data:
                position = step_data['positions'][0, agent_idx]  # Assume batch_size=1
                in_bottleneck = self.is_in_bottleneck(position)
                in_bottleneck_history.append(in_bottleneck)
            
            # Count transitions from outside -> inside -> outside as passages
            was_outside = not in_bottleneck_history[0]
            entered_bottleneck = False
            
            for in_bottleneck in in_bottleneck_history[1:]:
                if was_outside and in_bottleneck:
                    entered_bottleneck = True
                elif entered_bottleneck and not in_bottleneck:
                    total_passages += 1
                    entered_bottleneck = False
                    was_outside = True
                else:
                    was_outside = not in_bottleneck
        
        # Calculate rate
        total_time = len(self.trajectory_data) * self.dt
        throughput = total_passages / total_time if total_time > 0 else 0.0
        
        return throughput
    
    def compute_velocity_fluctuation(self) -> float:
        """
        Compute velocity fluctuation: standard deviation of speeds in bottleneck.
        
        Returns:
            Standard deviation of velocities in bottleneck region
        """
        bottleneck_speeds = []
        
        for step_data in self.trajectory_data:
            positions = step_data['positions'][0]  # Assume batch_size=1
            velocities = step_data['velocities'][0]
            
            for agent_idx in range(positions.shape[0]):
                position = positions[agent_idx]
                velocity = velocities[agent_idx]
                
                if self.is_in_bottleneck(position):
                    speed = np.linalg.norm(velocity)
                    bottleneck_speeds.append(speed)
        
        if len(bottleneck_speeds) == 0:
            return 0.0
            
        return float(np.std(bottleneck_speeds))
    
    def compute_waiting_time(self) -> float:
        """
        Compute total waiting time: time spent with speed below threshold near bottleneck.
        
        Returns:
            Total waiting time in seconds
        """
        total_waiting_time = 0.0
        
        for step_data in self.trajectory_data:
            positions = step_data['positions'][0]  # Assume batch_size=1
            velocities = step_data['velocities'][0]
            
            for agent_idx in range(positions.shape[0]):
                position = positions[agent_idx]
                velocity = velocities[agent_idx]
                speed = np.linalg.norm(velocity)
                
                # Check if agent is near bottleneck and moving slowly
                near_bottleneck = self.is_near_bottleneck(position)
                if near_bottleneck and speed < self.velocity_threshold:
                    total_waiting_time += self.dt
        
        return total_waiting_time
    
    def is_near_bottleneck(self, position: np.ndarray, margin: float = 0.2) -> bool:
        """Check if position is near the bottleneck entrance"""
        x, y = position[0], position[1]
        expanded_region = BottleneckRegion(
            x_min=self.bottleneck_region.x_min - margin,
            x_max=self.bottleneck_region.x_max + margin,
            y_min=self.bottleneck_region.y_min - margin,
            y_max=self.bottleneck_region.y_max + margin
        )
        return (expanded_region.x_min <= x <= expanded_region.x_max and
                expanded_region.y_min <= y <= expanded_region.y_max)
    
    def compute_avg_bottleneck_time(self) -> float:
        """
        Compute average time each agent spends in bottleneck region.
        
        Returns:
            Average bottleneck residence time in seconds
        """
        if len(self.trajectory_data) < 2:
            return 0.0
            
        num_agents = self.trajectory_data[0]['positions'].shape[1]
        total_bottleneck_time = 0.0
        agents_in_bottleneck = 0
        
        for agent_idx in range(num_agents):
            bottleneck_time = 0.0
            
            for step_data in self.trajectory_data:
                position = step_data['positions'][0, agent_idx]
                if self.is_in_bottleneck(position):
                    bottleneck_time += self.dt
            
            if bottleneck_time > 0:
                total_bottleneck_time += bottleneck_time
                agents_in_bottleneck += 1
        
        return total_bottleneck_time / agents_in_bottleneck if agents_in_bottleneck > 0 else 0.0
    
    def compute_coordination_efficiency(self) -> float:
        """
        Compute coordination efficiency based on collision avoidance and smooth flow.
        
        Returns:
            Coordination efficiency score (0-1, higher is better)
        """
        if len(self.trajectory_data) < 2:
            return 0.0
        
        # Calculate efficiency based on:
        # 1. Low velocity variance (smooth flow)
        # 2. High throughput
        # 3. Low waiting time
        
        velocity_fluctuation = self.compute_velocity_fluctuation()
        throughput = self.compute_throughput()
        waiting_time = self.compute_waiting_time()
        
        # Normalize metrics (higher efficiency = lower fluctuation and waiting, higher throughput)
        max_throughput = 2.0  # Expected max throughput for normalization
        max_waiting = 10.0    # Expected max waiting time
        max_fluctuation = 1.0 # Expected max velocity fluctuation
        
        throughput_score = min(throughput / max_throughput, 1.0)
        waiting_score = max(0, 1.0 - waiting_time / max_waiting)
        fluctuation_score = max(0, 1.0 - velocity_fluctuation / max_fluctuation)
        
        # Weighted combination
        efficiency = 0.4 * throughput_score + 0.3 * waiting_score + 0.3 * fluctuation_score
        
        return efficiency
    
    def compute_collision_rate(self, agent_radius: float = 0.05) -> float:
        """
        Compute collision rate in bottleneck region.
        
        Args:
            agent_radius: Radius of each agent for collision detection
            
        Returns:
            Collision rate (collisions per agent per second)
        """
        total_collisions = 0
        total_agent_time_in_bottleneck = 0.0
        
        for step_data in self.trajectory_data:
            positions = step_data['positions'][0]  # Assume batch_size=1
            num_agents = positions.shape[0]
            
            agents_in_bottleneck = []
            for agent_idx in range(num_agents):
                if self.is_in_bottleneck(positions[agent_idx]):
                    agents_in_bottleneck.append(agent_idx)
                    total_agent_time_in_bottleneck += self.dt
            
            # Check for collisions among agents in bottleneck
            for i, agent_i in enumerate(agents_in_bottleneck):
                for agent_j in agents_in_bottleneck[i+1:]:
                    pos_i = positions[agent_i]
                    pos_j = positions[agent_j]
                    distance = np.linalg.norm(pos_i - pos_j)
                    
                    if distance < 2 * agent_radius:
                        total_collisions += 1
        
        # Rate per agent per second
        collision_rate = total_collisions / total_agent_time_in_bottleneck if total_agent_time_in_bottleneck > 0 else 0.0
        
        return collision_rate
    
    def compute_completion_rate(self, goal_threshold: float = 0.1) -> float:
        """
        Compute percentage of agents that reach their goals.
        
        Args:
            goal_threshold: Distance threshold for considering goal reached
            
        Returns:
            Completion rate (0-1)
        """
        if len(self.trajectory_data) == 0:
            return 0.0
            
        final_step = self.trajectory_data[-1]
        positions = final_step['positions'][0]  # Assume batch_size=1
        goals = final_step['goals'][0]
        num_agents = positions.shape[0]
        
        agents_at_goal = 0
        for agent_idx in range(num_agents):
            distance_to_goal = np.linalg.norm(positions[agent_idx] - goals[agent_idx])
            if distance_to_goal < goal_threshold:
                agents_at_goal += 1
        
        return agents_at_goal / num_agents
    
    def analyze(self, agent_radius: float = 0.05) -> BottleneckMetrics:
        """
        Perform complete bottleneck analysis.
        
        Args:
            agent_radius: Radius of agents for collision detection
            
        Returns:
            BottleneckMetrics object with all computed metrics
        """
        metrics = BottleneckMetrics(
            throughput=self.compute_throughput(),
            velocity_fluctuation=self.compute_velocity_fluctuation(),
            total_waiting_time=self.compute_waiting_time(),
            avg_bottleneck_time=self.compute_avg_bottleneck_time(),
            coordination_efficiency=self.compute_coordination_efficiency(),
            collision_rate=self.compute_collision_rate(agent_radius),
            completion_rate=self.compute_completion_rate()
        )
        
        return metrics 