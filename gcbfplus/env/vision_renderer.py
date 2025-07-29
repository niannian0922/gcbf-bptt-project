"""
Simplified Vision Renderer for Multi-Agent Navigation
This module provides a lightweight alternative to PyTorch3D for generating depth images.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Any
import math


class SimpleDepthRenderer(nn.Module):
    """
    A simplified depth renderer that creates depth images from 2D agent/obstacle positions.
    This is a lightweight alternative to PyTorch3D for our navigation task.
    """
    
    def __init__(self, config: Dict):
        super(SimpleDepthRenderer, self).__init__()
        
        self.image_size = config.get('image_size', 64)
        self.camera_fov = config.get('camera_fov', 90.0)  # degrees
        self.camera_range = config.get('camera_range', 3.0)
        self.agent_radius = config.get('agent_radius', 0.05)
        self.obstacle_base_height = config.get('obstacle_base_height', 0.5)
        
        # Convert FOV to radians
        self.fov_rad = math.radians(self.camera_fov)
        
        # Create coordinate grids for the image
        self.register_buffer('pixel_coords', self._create_pixel_coordinates())
        
    def _create_pixel_coordinates(self) -> torch.Tensor:
        """Create normalized pixel coordinates for the depth image."""
        # Create pixel coordinates [-1, 1] for the image
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, self.image_size),
            torch.linspace(-1, 1, self.image_size),
            indexing='ij'
        )
        
        # Stack to create [H, W, 2] coordinate grid
        coords = torch.stack([x_coords, y_coords], dim=2)
        return coords
    
    def render_depth_from_position(
        self, 
        agent_pos: torch.Tensor,      # [2] - current agent position
        goal_pos: torch.Tensor,       # [2] - agent's goal position  
        other_agents: torch.Tensor,   # [N-1, 2] - other agent positions
        obstacles: Optional[torch.Tensor] = None  # [M, 3] - obstacle positions and radii
    ) -> torch.Tensor:
        """
        Render a depth image from the agent's perspective.
        
        Args:
            agent_pos: Current agent position [x, y]
            goal_pos: Agent's goal position [x, y] 
            other_agents: Positions of other agents [N-1, 2]
            obstacles: Obstacle positions and radii [M, 3] where each row is [x, y, radius]
            
        Returns:
            Depth image [1, H, W] with values in [0, 1]
        """
        device = agent_pos.device
        
        # Calculate viewing direction (towards goal)
        view_direction = goal_pos - agent_pos
        view_direction = view_direction / (torch.norm(view_direction) + 1e-8)
        
        # Create rotation matrix to align view with y-axis
        cos_theta = view_direction[1]  # y component
        sin_theta = view_direction[0]  # x component
        
        rotation_matrix = torch.tensor([
            [cos_theta, sin_theta],
            [-sin_theta, cos_theta]
        ], device=device, dtype=torch.float32)
        
        # Initialize depth image (far distance = 1.0, near = 0.0)
        depth_image = torch.ones(self.image_size, self.image_size, device=device)
        
        # Render other agents
        if other_agents.numel() > 0:
            depth_image = self._render_objects(
                depth_image, agent_pos, rotation_matrix, other_agents, 
                self.agent_radius, height=0.2
            )
        
        # Render obstacles
        if obstacles is not None and obstacles.numel() > 0:
            obstacle_positions = obstacles[:, :2]  # [M, 2]
            obstacle_radii = obstacles[:, 2]       # [M]
            
            for i in range(obstacles.shape[0]):
                single_obstacle = obstacle_positions[i:i+1]  # [1, 2]
                obstacle_radius = obstacle_radii[i].item()
                depth_image = self._render_objects(
                    depth_image, agent_pos, rotation_matrix, single_obstacle,
                    obstacle_radius, height=self.obstacle_base_height
                )
        
        return depth_image.unsqueeze(0)  # Add channel dimension [1, H, W]
    
    def _render_objects(
        self,
        depth_image: torch.Tensor,
        agent_pos: torch.Tensor,
        rotation_matrix: torch.Tensor,
        object_positions: torch.Tensor,
        object_radius: float,
        height: float
    ) -> torch.Tensor:
        """
        Render objects (agents or obstacles) into the depth image.
        
        Args:
            depth_image: Current depth image [H, W]
            agent_pos: Viewing agent position [2]
            rotation_matrix: 2D rotation matrix [2, 2]
            object_positions: Object positions to render [N, 2]
            object_radius: Radius of objects to render
            height: Height of objects (affects depth intensity)
            
        Returns:
            Updated depth image [H, W]
        """
        device = depth_image.device
        
        for obj_pos in object_positions:
            # Transform object position to camera coordinate system
            relative_pos = obj_pos - agent_pos  # [2]
            
            # Skip if object is behind the agent (negative y in camera coords)
            if torch.dot(relative_pos, rotation_matrix[1, :]) < 0:
                continue
                
            # Apply rotation to get position in camera coordinates
            cam_pos = torch.matmul(rotation_matrix, relative_pos)  # [2]
            
            # Project to image coordinates
            # Map from camera coords to image pixel coordinates
            distance = torch.norm(cam_pos)
            if distance > self.camera_range:
                continue
                
            # Calculate angular position
            angle_x = torch.atan2(cam_pos[0], cam_pos[1])  # horizontal angle
            
            # Check if object is within field of view
            if abs(angle_x) > self.fov_rad / 2:
                continue
            
            # Map angle to pixel coordinates
            pixel_x = (angle_x / (self.fov_rad / 2)) * 0.5  # normalize to [-0.5, 0.5]
            pixel_x_int = int((pixel_x + 0.5) * self.image_size)  # map to [0, image_size]
            
            # Calculate object size in pixels based on distance
            angular_size = object_radius / (distance + 1e-8)
            pixel_radius = int(angular_size * self.image_size * 0.5)
            pixel_radius = max(1, min(pixel_radius, self.image_size // 4))
            
            # Calculate depth value (closer objects have smaller depth values)
            depth_value = min(distance / self.camera_range, 1.0)
            # Modulate depth by object height for visual distinction
            depth_value = depth_value * (1.0 - height * 0.3)
            depth_value = max(0.0, depth_value)
            
            # Render circular object
            for dy in range(-pixel_radius, pixel_radius + 1):
                for dx in range(-pixel_radius, pixel_radius + 1):
                    if dx*dx + dy*dy <= pixel_radius*pixel_radius:
                        py = self.image_size // 2 + dy
                        px = pixel_x_int + dx
                        
                        if 0 <= py < self.image_size and 0 <= px < self.image_size:
                            # Update depth if this object is closer
                            depth_image[py, px] = min(depth_image[py, px], depth_value)
        
        return depth_image
    
    def add_noise_and_realism(self, depth_image: torch.Tensor, noise_level: float = 0.02) -> torch.Tensor:
        """
        Add noise and realistic effects to the depth image.
        
        Args:
            depth_image: Clean depth image [1, H, W]
            noise_level: Amount of noise to add
            
        Returns:
            Noisy depth image [1, H, W]
        """
        # Add Gaussian noise
        noise = torch.randn_like(depth_image) * noise_level
        noisy_depth = torch.clamp(depth_image + noise, 0.0, 1.0)
        
        # Add slight blur to simulate sensor limitations
        # Simple box filter
        kernel_size = 3
        padding = kernel_size // 2
        
        # Create simple averaging kernel
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=depth_image.device) / (kernel_size * kernel_size)
        
        # Apply convolution for blur effect
        blurred = torch.nn.functional.conv2d(
            noisy_depth.unsqueeze(0), kernel, padding=padding
        ).squeeze(0)
        
        return blurred


def create_simple_renderer(config: Dict) -> SimpleDepthRenderer:
    """
    Factory function to create a simple depth renderer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured SimpleDepthRenderer instance
    """
    return SimpleDepthRenderer(config) 