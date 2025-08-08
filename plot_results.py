#!/usr/bin/env python3
"""
Episode Results Plotting Script

This standalone script loads episode data from .npz files and generates
comprehensive visualizations for multi-agent behavior analysis.

Usage:
    python plot_results.py <episode_file.npz>
    
Example:
    python plot_results.py episode_logs/episode_001_20250806_120000.npz
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys
import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class EpisodePlotter:
    """Comprehensive plotter for multi-agent episode data."""
    
    def __init__(self, data_path: str):
        """
        Initialize plotter with episode data.
        
        Args:
            data_path: Path to the .npz episode data file
        """
        self.data_path = data_path
        self.data = self._load_data(data_path)
        self.n_agents = self.data.get('n_agents', 0)
        self.total_steps = self.data.get('total_steps', 0)
        self.final_status = self.data.get('final_status', 'UNKNOWN')
        self.safety_radius = self.data.get('safety_radius', 0.2)
        
        # Agent colors for consistent visualization
        self.agent_colors = plt.cm.tab10(np.linspace(0, 1, max(10, self.n_agents)))
        
        print(f"Loaded episode data:")
        print(f"   File: {data_path}")
        print(f"   Agents: {self.n_agents}, Steps: {self.total_steps}")
        print(f"   Status: {self.final_status}")
    
    def _load_data(self, data_path: str) -> Dict[str, Any]:
        """Load and validate episode data."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Episode data file not found: {data_path}")
        
        try:
            data = dict(np.load(data_path, allow_pickle=True))
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load episode data: {e}")
    
    def plot_3d_trajectories(self, show_obstacles: bool = True, 
                           show_goals: bool = True, 
                           show_start_end: bool = True) -> plt.Figure:
        """
        Generate 3D trajectory plot showing agent paths through space.
        
        Args:
            show_obstacles: Whether to display obstacles
            show_goals: Whether to display goal positions
            show_start_end: Whether to highlight start/end points
            
        Returns:
            matplotlib Figure object
        """
        if 'positions' not in self.data:
            raise ValueError("No position data found in episode file")
        
        positions = self.data['positions']  # [timesteps, batch, n_agents, pos_dim]
        batch_size = positions.shape[1]
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectories for each agent in the first batch
        for agent_idx in range(self.n_agents):
            agent_pos = positions[:, 0, agent_idx, :]  # [timesteps, pos_dim]
            
            if agent_pos.shape[1] == 2:
                # 2D environment - add z=0
                x, y = agent_pos[:, 0], agent_pos[:, 1]
                z = np.zeros_like(x)
            else:
                # 3D environment
                x, y, z = agent_pos[:, 0], agent_pos[:, 1], agent_pos[:, 2]
            
            color = self.agent_colors[agent_idx]
            
            # Plot trajectory
            ax.plot(x, y, z, color=color, linewidth=2, alpha=0.8, 
                   label=f'Agent {agent_idx + 1}')
            
            if show_start_end:
                # Mark start point
                ax.scatter([x[0]], [y[0]], [z[0]], color=color, s=100, 
                          marker='o', edgecolors='black', linewidth=2, alpha=0.9)
                
                # Mark end point
                ax.scatter([x[-1]], [y[-1]], [z[-1]], color=color, s=100, 
                          marker='s', edgecolors='black', linewidth=2, alpha=0.9)
        
        # Add obstacles if available
        if show_obstacles and 'obstacles' in self.data and self.data['obstacles'] is not None:
            try:
                obstacles_data = self.data['obstacles']
                
                # 🔧 ROBUSTNESS FIX: Handle dynamic obstacle data format
                if len(obstacles_data.shape) == 3:
                    # New format: [timesteps, n_obstacles, pos_dim+1] - Use final timestep
                    obstacles = obstacles_data[-1]  # Use last timestep obstacles
                    print(f"Using dynamic obstacle data (final timestep): {obstacles.shape}")
                elif len(obstacles_data.shape) == 2:
                    # Legacy format: [n_obstacles, pos_dim+1]
                    obstacles = obstacles_data
                    print(f"Using static obstacle data: {obstacles.shape}")
                else:
                    print(f"Warning: Unexpected obstacle data shape: {obstacles_data.shape}")
                    obstacles = None
                
                if obstacles is not None:
                    # Filter out dummy obstacles (those with zero radius or far-away position)
                    valid_obstacles = []
                    for obs in obstacles:
                        if (len(obs) >= 3 and obs[2] > 0 and 
                            abs(obs[0]) < 50 and abs(obs[1]) < 50):  # Valid obstacle
                            valid_obstacles.append(obs)
                    
                    print(f"Found {len(valid_obstacles)} valid obstacles for 3D plot")
                    
                    for obs in valid_obstacles:
                        x_obs, y_obs, radius = obs[0], obs[1], obs[2]
                        z_obs = 0.0  # Place obstacles at z=0
                        
                        # Draw obstacle as a cylinder
                        theta = np.linspace(0, 2*np.pi, 20)
                        x_circle = x_obs + radius * np.cos(theta)
                        y_circle = y_obs + radius * np.sin(theta)
                        z_bottom = np.full_like(x_circle, z_obs)
                        z_top = np.full_like(x_circle, z_obs + 0.5)  # Height of 0.5
                        
                        # Plot cylinder sides
                        for i in range(len(theta)):
                            ax.plot([x_circle[i], x_circle[i]], 
                                   [y_circle[i], y_circle[i]], 
                                   [z_bottom[i], z_top[i]], 
                                   color='red', alpha=0.3)
                        
                        # Plot top and bottom circles
                        ax.plot(x_circle, y_circle, z_bottom, color='red', linewidth=2)
                        ax.plot(x_circle, y_circle, z_top, color='red', linewidth=2)
                        
            except Exception as e:
                print(f"Warning: Failed to plot obstacles in 3D trajectory: {e}")
                # Continue without obstacles
        
        # Add goals if available
        if show_goals and 'goals' in self.data and self.data['goals'] is not None:
            goals = self.data['goals']  # [batch, n_agents, pos_dim]
            
            for agent_idx in range(self.n_agents):
                goal_pos = goals[0, agent_idx, :]  # First batch
                
                if len(goal_pos) == 2:
                    x_goal, y_goal, z_goal = goal_pos[0], goal_pos[1], 0.0
                else:
                    x_goal, y_goal, z_goal = goal_pos[0], goal_pos[1], goal_pos[2]
                
                color = self.agent_colors[agent_idx]
                ax.scatter([x_goal], [y_goal], [z_goal], color=color, s=200, 
                          marker='*', edgecolors='black', linewidth=2, alpha=0.9)
        
        # Formatting
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(f'3D Agent Trajectories - {self.final_status}\n'
                    f'Episode: {self.data.get("episode_id", "Unknown")}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio
        max_range = np.array([positions[:, 0, :, 0].max() - positions[:, 0, :, 0].min(),
                             positions[:, 0, :, 1].max() - positions[:, 0, :, 1].min()]).max() / 2.0
        mid_x = (positions[:, 0, :, 0].max() + positions[:, 0, :, 0].min()) * 0.5
        mid_y = (positions[:, 0, :, 1].max() + positions[:, 0, :, 1].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(-0.1, 1.0)
        
        plt.tight_layout()
        return fig
    
    def plot_safety_distances(self, show_safety_threshold: bool = True) -> plt.Figure:
        """
        Generate safety distance plot over time.
        
        Args:
            show_safety_threshold: Whether to show the safety radius threshold
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Check for minimum distances data
        if 'min_distances' in self.data and self.data['min_distances'] is not None:
            min_distances = self.data['min_distances']  # [timesteps, batch, n_agents]
            timesteps = np.arange(len(min_distances))
            
            for agent_idx in range(self.n_agents):
                agent_distances = min_distances[:, 0, agent_idx]  # First batch
                color = self.agent_colors[agent_idx]
                
                ax.plot(timesteps, agent_distances, color=color, linewidth=2, 
                       label=f'Agent {agent_idx + 1}', alpha=0.8)
        
        # If no min_distances, compute from positions and obstacles
        elif 'positions' in self.data and 'obstacles' in self.data:
            try:
                positions = self.data['positions']  # [timesteps, batch, n_agents, pos_dim]
                obstacles_data = self.data['obstacles']
                timesteps = np.arange(len(positions))
                
                print("Computing minimum distances from position data...")
                
                for agent_idx in range(self.n_agents):
                    min_dists = []
                    
                    for t in range(len(positions)):
                        agent_pos = positions[t, 0, agent_idx, :2]  # First batch, x,y only
                        
                        # 🔧 ROBUSTNESS FIX: Handle dynamic obstacle data
                        if obstacles_data is not None:
                            if len(obstacles_data.shape) == 3:
                                # Dynamic obstacles: [timesteps, n_obstacles, pos_dim+1]
                                if t < len(obstacles_data):
                                    current_obstacles = obstacles_data[t]
                                else:
                                    current_obstacles = obstacles_data[-1]  # Use last timestep
                            else:
                                # Static obstacles: [n_obstacles, pos_dim+1]  
                                current_obstacles = obstacles_data
                            
                            # Filter valid obstacles
                            valid_obstacles = []
                            for obs in current_obstacles:
                                if (len(obs) >= 3 and obs[2] > 0 and 
                                    abs(obs[0]) < 50 and abs(obs[1]) < 50):
                                    valid_obstacles.append(obs)
                            
                            if len(valid_obstacles) > 0:
                                valid_obstacles = np.array(valid_obstacles)
                                obs_positions = valid_obstacles[:, :2]  # x, y positions
                                obs_radii = valid_obstacles[:, 2]       # radii
                                
                                distances = np.linalg.norm(obs_positions - agent_pos, axis=1) - obs_radii
                                min_dist = np.min(distances)
                            else:
                                min_dist = float('inf')  # No valid obstacles
                        else:
                            min_dist = float('inf')  # No obstacles
                        
                        min_dists.append(min_dist)
                    
                    color = self.agent_colors[agent_idx]
                    ax.plot(timesteps, min_dists, color=color, linewidth=2,
                           label=f'Agent {agent_idx + 1}', alpha=0.8)
                           
            except Exception as e:
                print(f"Warning: Failed to compute minimum distances: {e}")
                ax.text(0.5, 0.5, 'Distance data unavailable', 
                       transform=ax.transAxes, ha='center', va='center')
        
        else:
            ax.text(0.5, 0.5, 'No distance data available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=16)
            ax.set_title('Safety Distance Analysis - No Data')
            return fig
        
        # Add safety threshold line
        if show_safety_threshold and self.safety_radius > 0:
            ax.axhline(y=self.safety_radius, color='red', linestyle='--', linewidth=2,
                      alpha=0.7, label=f'Safety Radius ({self.safety_radius:.2f})')
        
        # Formatting
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Minimum Safety Distance')
        ax.set_title(f'Minimum Safety Distance Over Time - {self.final_status}\n'
                    f'Episode: {self.data.get("episode_id", "Unknown")}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Highlight collision zones
        ax.axhspan(-1, self.safety_radius if self.safety_radius > 0 else 0.2, 
                  alpha=0.2, color='red', label='Collision Zone')
        
        plt.tight_layout()
        return fig
    
    def plot_cbf_analysis(self) -> Optional[plt.Figure]:
        """
        Generate CBF (Control Barrier Function) analysis plots.
        
        Returns:
            matplotlib Figure object or None if no CBF data
        """
        if 'h_values' not in self.data or self.data['h_values'] is None:
            print("⚠️  No CBF h_values data found")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        h_values = self.data['h_values']        # [timesteps, batch, n_agents, 1]
        alpha_values = self.data.get('alpha_values', None)  # [timesteps, batch, n_agents, 1]
        timesteps = np.arange(len(h_values))
        
        # Plot 1: CBF h-values
        for agent_idx in range(self.n_agents):
            agent_h = h_values[:, 0, agent_idx, 0]  # First batch
            color = self.agent_colors[agent_idx]
            
            ax1.plot(timesteps, agent_h, color=color, linewidth=2,
                    label=f'Agent {agent_idx + 1}', alpha=0.8)
        
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label='Safety Boundary (h=0)')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('CBF h-value')
        ax1.set_title('Control Barrier Function Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Alpha values (if available)
        if alpha_values is not None:
            for agent_idx in range(self.n_agents):
                agent_alpha = alpha_values[:, 0, agent_idx, 0]  # First batch
                color = self.agent_colors[agent_idx]
                
                ax2.plot(timesteps, agent_alpha, color=color, linewidth=2,
                        label=f'Agent {agent_idx + 1}', alpha=0.8)
            
            ax2.set_xlabel('Timestep')
            ax2.set_ylabel('Alpha Value')
            ax2.set_title('CBF Alpha Parameters (Safety Aggressiveness)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Alpha values recorded', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=14)
            ax2.set_title('CBF Alpha Parameters - No Data')
        
        plt.tight_layout()
        return fig
    
    def plot_comprehensive_analysis(self) -> plt.Figure:
        """
        Generate a comprehensive 4-panel analysis plot.
        
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Panel 1: 2D Trajectory Overview (top-left)
        ax1 = plt.subplot(2, 2, 1)
        if 'positions' in self.data:
            positions = self.data['positions'][:, 0, :, :2]  # [timesteps, n_agents, 2]
            
            for agent_idx in range(self.n_agents):
                agent_pos = positions[:, agent_idx, :]
                color = self.agent_colors[agent_idx]
                
                ax1.plot(agent_pos[:, 0], agent_pos[:, 1], color=color, linewidth=2,
                        label=f'Agent {agent_idx + 1}', alpha=0.8)
                
                # Start and end points
                ax1.scatter(agent_pos[0, 0], agent_pos[0, 1], color=color, s=100,
                           marker='o', edgecolors='black', linewidth=2)
                ax1.scatter(agent_pos[-1, 0], agent_pos[-1, 1], color=color, s=100,
                           marker='s', edgecolors='black', linewidth=2)
            
            # Add obstacles
            if 'obstacles' in self.data and self.data['obstacles'] is not None:
                try:
                    obstacles_data = self.data['obstacles']
                    
                    # 🔧 ROBUSTNESS FIX: Handle dynamic obstacle data format
                    if len(obstacles_data.shape) == 3:
                        # Dynamic format: Use final timestep for overview
                        obstacles = obstacles_data[-1]
                    else:
                        # Static format
                        obstacles = obstacles_data
                    
                    # Filter and plot valid obstacles
                    for obs in obstacles:
                        if (len(obs) >= 3 and obs[2] > 0 and 
                            abs(obs[0]) < 50 and abs(obs[1]) < 50):  # Valid obstacle
                            circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.3)
                            ax1.add_patch(circle)
                            
                except Exception as e:
                    print(f"Warning: Failed to plot obstacles in comprehensive view: {e}")
            
            # Add goals
            if 'goals' in self.data and self.data['goals'] is not None:
                goals = self.data['goals'][0, :, :2]  # First batch
                for agent_idx, goal in enumerate(goals):
                    color = self.agent_colors[agent_idx]
                    ax1.scatter(goal[0], goal[1], color=color, s=200, marker='*',
                               edgecolors='black', linewidth=2)
        
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('2D Trajectory Overview')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Panel 2: Safety Distances (top-right)
        ax2 = plt.subplot(2, 2, 2)
        if 'min_distances' in self.data and self.data['min_distances'] is not None:
            min_distances = self.data['min_distances'][:, 0, :]  # [timesteps, n_agents]
            timesteps = np.arange(len(min_distances))
            
            for agent_idx in range(self.n_agents):
                color = self.agent_colors[agent_idx]
                ax2.plot(timesteps, min_distances[:, agent_idx], color=color, linewidth=2,
                        label=f'Agent {agent_idx + 1}', alpha=0.8)
            
            ax2.axhline(y=self.safety_radius, color='red', linestyle='--', linewidth=2,
                       alpha=0.7, label=f'Safety Radius')
        
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Min Distance')
        ax2.set_title('Safety Distances')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Goal Distances (bottom-left)
        ax3 = plt.subplot(2, 2, 3)
        if 'goal_distances' in self.data and self.data['goal_distances'] is not None:
            goal_distances = self.data['goal_distances'][:, 0, :]  # [timesteps, n_agents]
            timesteps = np.arange(len(goal_distances))
            
            for agent_idx in range(self.n_agents):
                color = self.agent_colors[agent_idx]
                ax3.plot(timesteps, goal_distances[:, agent_idx], color=color, linewidth=2,
                        label=f'Agent {agent_idx + 1}', alpha=0.8)
        
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Goal Distance')
        ax3.set_title('Goal Distances')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Action Magnitudes (bottom-right)
        ax4 = plt.subplot(2, 2, 4)
        if 'actions' in self.data:
            actions = self.data['actions'][:, 0, :, :]  # [timesteps, n_agents, action_dim]
            action_magnitudes = np.linalg.norm(actions, axis=-1)  # [timesteps, n_agents]
            timesteps = np.arange(len(action_magnitudes))
            
            for agent_idx in range(self.n_agents):
                color = self.agent_colors[agent_idx]
                ax4.plot(timesteps, action_magnitudes[:, agent_idx], color=color, linewidth=2,
                        label=f'Agent {agent_idx + 1}', alpha=0.8)
        
        ax4.set_xlabel('Timestep')
        ax4.set_ylabel('Action Magnitude')
        ax4.set_title('Action Magnitudes')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Comprehensive Episode Analysis - {self.final_status}\n'
                    f'Episode: {self.data.get("episode_id", "Unknown")}', fontsize=16)
        plt.tight_layout()
        return fig
    
    def save_all_plots(self, output_dir: str = "episode_plots") -> None:
        """
        Generate and save all available plots.
        
        Args:
            output_dir: Directory to save plot images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        episode_id = self.data.get('episode_id', 'unknown_episode')
        
        print(f"Generating plots for episode: {episode_id}")
        
        # 3D Trajectories
        try:
            fig_3d = self.plot_3d_trajectories()
            fig_3d.savefig(os.path.join(output_dir, f"{episode_id}_3d_trajectories.png"), 
                          dpi=300, bbox_inches='tight')
            plt.close(fig_3d)
            print("   3D trajectories plot saved")
        except Exception as e:
            print(f"   Failed to generate 3D trajectories plot: {e}")
        
        # Safety Distances
        try:
            fig_safety = self.plot_safety_distances()
            fig_safety.savefig(os.path.join(output_dir, f"{episode_id}_safety_distances.png"), 
                              dpi=300, bbox_inches='tight')
            plt.close(fig_safety)
            print("   Safety distances plot saved")
        except Exception as e:
            print(f"   Failed to generate safety distances plot: {e}")
        
        # CBF Analysis
        try:
            fig_cbf = self.plot_cbf_analysis()
            if fig_cbf is not None:
                fig_cbf.savefig(os.path.join(output_dir, f"{episode_id}_cbf_analysis.png"), 
                               dpi=300, bbox_inches='tight')
                plt.close(fig_cbf)
                print("   CBF analysis plot saved")
        except Exception as e:
            print(f"   Failed to generate CBF analysis plot: {e}")
        
        # Comprehensive Analysis
        try:
            fig_comp = self.plot_comprehensive_analysis()
            fig_comp.savefig(os.path.join(output_dir, f"{episode_id}_comprehensive.png"), 
                            dpi=300, bbox_inches='tight')
            plt.close(fig_comp)
            print("   Comprehensive analysis plot saved")
        except Exception as e:
            print(f"   Failed to generate comprehensive analysis plot: {e}")
        
        print(f"All plots saved to: {output_dir}")
    
    def generate_performance_dashboard(self, kpi_results: Dict[str, float], 
                                     best_episode_data: Dict[str, Any] = None,
                                     output_path: str = "performance_dashboard.png") -> plt.Figure:
        """
        🏆 **NEW: 生成冠军级别的性能仪表盘**
        
        Args:
            kpi_results: 包含所有KPI指标的字典
            best_episode_data: 最佳episode的详细数据 (可选)
            output_path: 输出图片路径
            
        Returns:
            matplotlib Figure object
        """
        # 创建2x2的图表布局
        fig = plt.figure(figsize=(20, 16))
        
        # 左侧：KPI文本展示
        ax_left = plt.subplot(1, 2, 1)
        ax_left.axis('off')  # 关闭坐标轴
        
        # 🏆 格式化KPI文本展示
        kpi_text = self._format_kpi_text(kpi_results)
        ax_left.text(0.05, 0.95, kpi_text, transform=ax_left.transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        ax_left.set_title("🏆 冠军评估体系 - KPI总览", fontsize=16, fontweight='bold', pad=20)
        
        # 右上：轨迹图
        ax_top_right = plt.subplot(2, 2, 2)
        if best_episode_data:
            self._plot_dashboard_trajectory(ax_top_right, best_episode_data)
        else:
            ax_top_right.text(0.5, 0.5, '最佳Episode数据不可用', 
                            transform=ax_top_right.transAxes, ha='center', va='center')
        ax_top_right.set_title("🥇 冠军Episode - 轨迹图", fontsize=14, fontweight='bold')
        
        # 右下：安全距离图
        ax_bottom_right = plt.subplot(2, 2, 4)
        if best_episode_data:
            self._plot_dashboard_safety(ax_bottom_right, best_episode_data)
        else:
            ax_bottom_right.text(0.5, 0.5, '安全距离数据不可用', 
                               transform=ax_bottom_right.transAxes, ha='center', va='center')
        ax_bottom_right.set_title("🛡️ 冠军Episode - 安全距离", fontsize=14, fontweight='bold')
        
        # 整体标题
        success_rate = kpi_results.get('champion/success_rate', 0)
        completion_time = kpi_results.get('champion/best_completion_time', 0)
        robustness = kpi_results.get('champion/robustness_score', 0)
        
        fig.suptitle(f'🏆 冠军模型性能仪表盘 | 成功率: {success_rate:.1%} | 最佳时间: {completion_time:.0f}步 | 鲁棒性: {robustness:.3f}', 
                    fontsize=18, fontweight='bold', y=0.96)
        
        plt.tight_layout()
        
        # 保存仪表盘
        try:
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"🏆 性能仪表盘已保存: {output_path}")
        except Exception as e:
            print(f"Warning: Failed to save dashboard: {e}")
        
        return fig
    
    def _format_kpi_text(self, kpi_results: Dict[str, float]) -> str:
        """格式化KPI结果为可读的文本."""
        lines = []
        lines.append("🏆 ═══ 冠军评估体系报告 ═══")
        lines.append("")
        
        # 基础性能指标
        lines.append("📊 核心性能指标:")
        lines.append(f"  ✅ 成功率: {kpi_results.get('champion/success_rate', 0):.1%}")
        lines.append(f"  ❌ 碰撞率: {kpi_results.get('champion/collision_rate', 0):.1%}")
        lines.append(f"  ⏰ 超时率: {kpi_results.get('champion/timeout_rate', 0):.1%}")
        lines.append(f"  🛡️ 鲁棒性得分: {kpi_results.get('champion/robustness_score', 0):.3f}")
        lines.append("")
        
        # 成功案例统计
        if 'champion/avg_completion_time_success' in kpi_results:
            lines.append("🎯 成功案例统计:")
            avg_time = kpi_results['champion/avg_completion_time_success']
            std_time = kpi_results.get('champion/std_completion_time_success', 0)
            lines.append(f"  ⏱️ 平均完成时间: {avg_time:.1f} ± {std_time:.1f} 步")
            lines.append(f"  🚀 最快完成时间: {kpi_results.get('champion/min_completion_time', 0):.0f} 步")
            lines.append(f"  🐌 最慢完成时间: {kpi_results.get('champion/max_completion_time', 0):.0f} 步")
            
            if 'champion/avg_jerk_success' in kpi_results:
                avg_jerk = kpi_results['champion/avg_jerk_success']
                std_jerk = kpi_results.get('champion/std_jerk_success', 0)
                lines.append(f"  📈 平均抖动: {avg_jerk:.4f} ± {std_jerk:.4f}")
                
            if 'champion/avg_min_safety_distance_success' in kpi_results:
                avg_safety = kpi_results['champion/avg_min_safety_distance_success']
                std_safety = kpi_results.get('champion/std_min_safety_distance_success', 0)
                lines.append(f"  🛡️ 平均安全距离: {avg_safety:.3f} ± {std_safety:.3f}")
            lines.append("")
        
        # 冠军episode信息
        if 'champion/best_episode_file' in kpi_results:
            lines.append("🥇 冠军Episode:")
            best_file = kpi_results['champion/best_episode_file']
            if isinstance(best_file, str):
                lines.append(f"  📁 文件: {os.path.basename(best_file)}")
            lines.append(f"  ⏱️ 完成时间: {kpi_results.get('champion/best_completion_time', 0):.0f} 步")
            lines.append(f"  📈 抖动值: {kpi_results.get('champion/best_episode_jerk', 0):.4f}")
            safety_val = kpi_results.get('champion/best_episode_safety', float('inf'))
            if safety_val != float('inf'):
                lines.append(f"  🛡️ 安全距离: {safety_val:.3f}")
            lines.append("")
        
        # 附加统计
        lines.append("📈 附加统计:")
        lines.append(f"  📊 平均Episode长度: {kpi_results.get('champion/avg_episode_length', 0):.1f} 步")
        if 'champion/avg_safety_violations_success' in kpi_results:
            violations = kpi_results['champion/avg_safety_violations_success']
            lines.append(f"  ⚠️ 平均安全违规: {violations:.1f} 次")
        
        lines.append("")
        lines.append("🏆 ═══════════════════════════")
        
        return '\n'.join(lines)
    
    def _plot_dashboard_trajectory(self, ax: plt.Axes, episode_data: Dict[str, Any]) -> None:
        """在仪表盘中绘制最佳episode的轨迹."""
        try:
            if 'positions' in episode_data and episode_data['positions'] is not None:
                positions = episode_data['positions']
                if len(positions.shape) == 3:  # [timesteps, agents, pos_dim]
                    # 2D轨迹图
                    for agent_idx in range(positions.shape[1]):
                        agent_pos = positions[:, agent_idx, :2]  # x, y only
                        color = plt.cm.tab10(agent_idx)
                        
                        # 绘制轨迹
                        ax.plot(agent_pos[:, 0], agent_pos[:, 1], color=color, 
                               linewidth=2, label=f'Agent {agent_idx + 1}', alpha=0.8)
                        
                        # 起点和终点
                        ax.scatter(agent_pos[0, 0], agent_pos[0, 1], color=color, 
                                 s=100, marker='o', edgecolors='black', linewidth=2)
                        ax.scatter(agent_pos[-1, 0], agent_pos[-1, 1], color=color, 
                                 s=100, marker='s', edgecolors='black', linewidth=2)
                    
                    # 添加障碍物
                    if 'obstacles' in episode_data and episode_data['obstacles'] is not None:
                        obstacles = episode_data['obstacles']
                        if len(obstacles.shape) == 3:
                            obstacles = obstacles[-1]  # 使用最后时间步
                        
                        for obs in obstacles:
                            if len(obs) >= 3 and obs[2] > 0:  # 有效障碍物
                                circle = plt.Circle((obs[0], obs[1]), obs[2], 
                                                  color='red', alpha=0.3)
                                ax.add_patch(circle)
                    
                    # 添加目标
                    if 'goals' in episode_data and episode_data['goals'] is not None:
                        goals = episode_data['goals']
                        if len(goals.shape) == 2:  # [agents, pos_dim]
                            for agent_idx, goal in enumerate(goals):
                                color = plt.cm.tab10(agent_idx)
                                ax.scatter(goal[0], goal[1], color=color, s=200, 
                                         marker='*', edgecolors='black', linewidth=2)
                    
                    ax.set_xlabel('X Position')
                    ax.set_ylabel('Y Position')
                    ax.grid(True, alpha=0.3)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.set_aspect('equal', adjustable='box')
                    
        except Exception as e:
            ax.text(0.5, 0.5, f'轨迹数据解析失败: {str(e)[:50]}...', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_dashboard_safety(self, ax: plt.Axes, episode_data: Dict[str, Any]) -> None:
        """在仪表盘中绘制最佳episode的安全距离."""
        try:
            if 'min_distances' in episode_data and episode_data['min_distances'] is not None:
                min_distances = episode_data['min_distances']
                timesteps = np.arange(len(min_distances))
                
                if len(min_distances.shape) == 2:  # [timesteps, agents]
                    for agent_idx in range(min_distances.shape[1]):
                        agent_distances = min_distances[:, agent_idx]
                        color = plt.cm.tab10(agent_idx)
                        ax.plot(timesteps, agent_distances, color=color, linewidth=2,
                               label=f'Agent {agent_idx + 1}', alpha=0.8)
                
                # 安全阈值线
                safety_radius = episode_data.get('safety_radius', 0.2)
                ax.axhline(y=safety_radius, color='red', linestyle='--', 
                          alpha=0.7, label=f'Safety Threshold ({safety_radius})')
                ax.axhline(y=0, color='red', linestyle='-', alpha=0.9, 
                          linewidth=2, label='Collision Boundary')
                
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Min Distance to Obstacles')
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # 设置y轴范围
                min_val = np.min(min_distances) if len(min_distances) > 0 else 0
                max_val = max(np.max(min_distances) if len(min_distances) > 0 else 1, safety_radius * 2)
                ax.set_ylim(min(min_val - 0.1, -0.1), max_val + 0.1)
                
            else:
                ax.text(0.5, 0.5, '安全距离数据不可用', 
                       transform=ax.transAxes, ha='center', va='center')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'安全数据解析失败: {str(e)[:50]}...', 
                   transform=ax.transAxes, ha='center', va='center')
    

def main():
    """Main entry point for the plotting script."""
    parser = argparse.ArgumentParser(description="Plot multi-agent episode results")
    parser.add_argument("episode_file", type=str, 
                       help="Path to the episode .npz data file")
    parser.add_argument("--output-dir", type=str, default="episode_plots",
                       help="Directory to save plot images (default: episode_plots)")
    parser.add_argument("--save-plots", action="store_true",
                       help="Save plots to files instead of displaying")
    parser.add_argument("--plot-type", type=str, choices=["3d", "safety", "cbf", "comprehensive", "all"],
                       default="all", help="Type of plot to generate (default: all)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.episode_file):
        print(f"Error: Episode file not found: {args.episode_file}")
        sys.exit(1)
    
    try:
        # Create plotter
        plotter = EpisodePlotter(args.episode_file)
        
        if args.save_plots:
            # Save all plots to files
            plotter.save_all_plots(args.output_dir)
        else:
            # Display plots interactively
            if args.plot_type in ["3d", "all"]:
                fig_3d = plotter.plot_3d_trajectories()
                plt.show()
            
            if args.plot_type in ["safety", "all"]:
                fig_safety = plotter.plot_safety_distances()
                plt.show()
            
            if args.plot_type in ["cbf", "all"]:
                fig_cbf = plotter.plot_cbf_analysis()
                if fig_cbf is not None:
                    plt.show()
            
            if args.plot_type in ["comprehensive", "all"]:
                fig_comp = plotter.plot_comprehensive_analysis()
                plt.show()
    
    except Exception as e:
        print(f"Error processing episode data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
