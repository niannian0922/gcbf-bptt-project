#!/usr/bin/env python3
"""
智能模型加載器 - 根據實際權重完全重建網絡架構
"""

import torch
import torch.nn as nn
import yaml
import os
from gcbfplus.env import DoubleIntegratorEnv
from gcbfplus.policy import BPTTPolicy, create_policy_from_config
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def analyze_model_architecture(policy_path):
    """
    分析模型權重來推斷完整的網絡架構
    """
    print(f"🔍 分析模型架構: {policy_path}")
    
    state_dict = torch.load(policy_path, map_location='cpu', weights_only=True)
    
    print(f"📋 模型權重鍵:")
    for key, tensor in state_dict.items():
        print(f"  {key}: {tensor.shape}")
    
    # 分析perception層
    perception_input_dim = None
    perception_output_dim = None
    perception_layers = []
    
    if 'perception.mlp.0.weight' in state_dict:
        perception_input_dim = state_dict['perception.mlp.0.weight'].shape[1]
        perception_output_dim = state_dict['perception.mlp.0.weight'].shape[0]
        
    # 找到所有perception層
    for key in state_dict.keys():
        if key.startswith('perception.mlp.') and key.endswith('.weight'):
            layer_num = int(key.split('.')[2])
            out_features = state_dict[key].shape[0]
            perception_layers.append((layer_num, out_features))
    
    perception_layers.sort()
    perception_hidden_dims = [dim for _, dim in perception_layers[:-1]]  # 除了最後一層
    
    # 分析memory層
    memory_hidden_dim = None
    if 'memory.gru.weight_hh_l0' in state_dict:
        memory_hidden_dim = state_dict['memory.gru.weight_hh_l0'].shape[1]
    
    # 分析policy_head層
    policy_head_layers = []
    action_layers = []
    alpha_layers = []
    
    # 找action_layers
    for key in state_dict.keys():
        if key.startswith('policy_head.action_layers.') and key.endswith('.weight'):
            # 從 'policy_head.action_layers.N.weight' 中提取 N
            parts = key.split('.')
            if len(parts) >= 4 and parts[3].isdigit():
                layer_num = int(parts[3])
                out_features = state_dict[key].shape[0]
                action_layers.append((layer_num, out_features))
    
    action_layers.sort()
    
    # 找alpha_network
    for key in state_dict.keys():
        if key.startswith('policy_head.alpha_network.') and key.endswith('.weight'):
            # 從 'policy_head.alpha_network.N.weight' 中提取 N
            parts = key.split('.')
            if len(parts) >= 4 and parts[3].isdigit():
                layer_num = int(parts[3])
                out_features = state_dict[key].shape[0]
                alpha_layers.append((layer_num, out_features))
    
    alpha_layers.sort()
    
    # 構建完整配置
    config = {
        'perception': {
            'use_vision': False,
            'input_dim': perception_input_dim,
            'output_dim': perception_output_dim,
            'hidden_dims': perception_hidden_dims,
            'activation': 'relu'
        },
        'memory': {
            'hidden_dim': memory_hidden_dim,
            'num_layers': 1
        },
        'policy_head': {
            'output_dim': action_layers[-1][1] if action_layers else 2,  # 最後一層是輸出維度
            'hidden_dims': [dim for _, dim in action_layers[:-1]] if len(action_layers) > 1 else [perception_output_dim],
            'activation': 'relu',
            'predict_alpha': True,
            'alpha_hidden_dims': [dim for _, dim in alpha_layers[:-1]] if len(alpha_layers) > 1 else [perception_output_dim]
        }
    }
    
    print(f"✅ 推斷的網絡架構:")
    print(f"  Perception: 輸入={perception_input_dim}, 輸出={perception_output_dim}, 隱藏層={perception_hidden_dims}")
    print(f"  Memory: 隱藏維度={memory_hidden_dim}")
    print(f"  Action層: {[dim for _, dim in action_layers]}")
    print(f"  Alpha層: {[dim for _, dim in alpha_layers]}")
    
    return config


def create_exact_model(model_path, device='cpu'):
    """
    根據實際模型權重創建完全匹配的網絡
    """
    print(f"🎯 創建精確匹配的模型")
    
    # 分析架構
    policy_config = analyze_model_architecture(model_path)
    
    # 創建網絡
    policy_network = create_policy_from_config(policy_config)
    policy_network = policy_network.to(device)
    
    # 加載權重
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    try:
        policy_network.load_state_dict(state_dict, strict=False)
        print(f"✅ 模型加載成功 (strict=False)")
    except Exception as e:
        print(f"❌ 模型加載失败: {e}")
        return None, None
    
    return policy_network, policy_config


def create_simple_visualization(env, policy_network, device, num_steps=100):
    """
    創建簡單的可視化
    """
    print(f"🎬 開始簡單可視化 ({num_steps} 步)")
    
    policy_network.eval()
    
    # 初始化環境
    state = env.reset()
    
    trajectory_positions = []
    
    with torch.no_grad():
        for step in range(num_steps):
            # 記錄位置
            current_positions = state.positions[0].cpu().numpy()
            trajectory_positions.append(current_positions.copy())
            
            # 獲取觀測
            observations = env.get_observations(state).to(device)
            
            # 策略推理
            try:
                policy_output = policy_network(observations, state)
                actions = policy_output.actions
                alphas = getattr(policy_output, 'alphas', torch.ones_like(actions[:, :, :1]) * 0.5)
                
                # 檢查動作
                action_magnitude = torch.norm(actions, dim=-1).mean().item()
                if step % 20 == 0:
                    print(f"步驟 {step}: 動作強度={action_magnitude:.6f}")
                
                # 環境步進
                step_result = env.step(state, actions, alphas)
                state = step_result.next_state
                
            except Exception as e:
                print(f"❌ 步驟 {step} 失敗: {e}")
                break
    
    # 創建動畫
    print(f"🎨 創建動畫...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('🎯 統一代碼路徑 - 真實模型可視化', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加障碍物
    obstacles = [
        {'pos': [0.0, -0.8], 'radius': 0.4},
        {'pos': [0.0, 0.8], 'radius': 0.4}
    ]
    
    for obs in obstacles:
        circle = plt.Circle(obs['pos'], obs['radius'], color='red', alpha=0.8)
        ax.add_patch(circle)
    
    # 起始和目標區域
    start_zone = plt.Rectangle((-2.5, -1.5), 1.0, 3.0, fill=False, 
                              edgecolor='green', linestyle='--', linewidth=3, 
                              alpha=0.9, label='起始區域')
    ax.add_patch(start_zone)
    
    target_zone = plt.Rectangle((1.5, -1.5), 1.0, 3.0, fill=False, 
                               edgecolor='blue', linestyle='--', linewidth=3, 
                               alpha=0.9, label='目標區域')
    ax.add_patch(target_zone)
    
    # 智能體顏色
    num_agents = len(trajectory_positions[0])
    colors = ['#FF4444', '#44FF44', '#4444FF', '#FFAA44', '#FF44AA', '#44AAFF'][:num_agents]
    
    # 軌跡線和點
    trail_lines = []
    agent_dots = []
    
    for i in range(num_agents):
        line, = ax.plot([], [], '-', color=colors[i], alpha=0.8, linewidth=2,
                       label=f'智能體{i+1}' if i < 3 else "")
        trail_lines.append(line)
        
        dot, = ax.plot([], [], 'o', color=colors[i], markersize=12, 
                      markeredgecolor='black', markeredgewidth=2, zorder=5)
        agent_dots.append(dot)
    
    ax.legend()
    
    # 添加信息文本
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame):
        if frame >= len(trajectory_positions):
            return trail_lines + agent_dots + [info_text]
        
        current_pos = trajectory_positions[frame]
        
        # 更新軌跡和智能體位置
        for i in range(num_agents):
            trail_x = [pos[i, 0] for pos in trajectory_positions[:frame+1]]
            trail_y = [pos[i, 1] for pos in trajectory_positions[:frame+1]]
            trail_lines[i].set_data(trail_x, trail_y)
            
            agent_dots[i].set_data([current_pos[i, 0]], [current_pos[i, 1]])
        
        # 計算總位移
        if frame > 0:
            initial_pos = trajectory_positions[0]
            total_displacement = np.mean([
                np.linalg.norm(current_pos[i] - initial_pos[i]) 
                for i in range(num_agents)
            ])
        else:
            total_displacement = 0
        
        info_text.set_text(f'步驟: {frame}\n智能體數: {num_agents}\n平均位移: {total_displacement:.3f}')
        
        return trail_lines + agent_dots + [info_text]
    
    anim = FuncAnimation(fig, animate, frames=len(trajectory_positions),
                        interval=100, blit=False, repeat=True)
    
    return anim, trajectory_positions


def main():
    """
    主函數
    """
    print(f"🎯 智能模型加載器")
    print(f"=" * 60)
    
    # 設置
    model_dir = 'logs/full_collaboration_training'
    device = torch.device('cpu')
    
    # 找到最新模型
    models_dir = os.path.join(model_dir, 'models')
    steps = [int(d) for d in os.listdir(models_dir) if d.isdigit()]
    latest_step = max(steps)
    
    policy_path = os.path.join(model_dir, 'models', str(latest_step), 'policy.pt')
    print(f"📁 使用模型: {policy_path}")
    
    # 創建精確模型
    policy_network, policy_config = create_exact_model(policy_path, device)
    
    if policy_network is None:
        print(f"❌ 模型創建失敗")
        return
    
    # 創建環境
    env_config = {
        'area_size': 3.0,
        'car_radius': 0.15,
        'comm_radius': 1.0,
        'dt': 0.05,
        'mass': 0.1,
        'max_force': 1.0,
        'max_steps': 80,
        'name': 'DoubleIntegrator',
        'num_agents': 6,
        'obstacles': {
            'enabled': True,
            'bottleneck': True,
            'positions': [[0.0, -0.8], [0.0, 0.8]],
            'radii': [0.4, 0.4]
        }
    }
    
    env = DoubleIntegratorEnv(env_config)
    env = env.to(device)
    
    print(f"🌍 環境創建成功")
    print(f"📏 觀測形狀: {env.observation_shape}")
    print(f"📏 動作形狀: {env.action_shape}")
    
    # 創建可視化
    anim, trajectory = create_simple_visualization(env, policy_network, device, 120)
    
    # 保存動畫
    output_path = 'SMART_FINAL_COLLABORATION_RESULT.mp4'
    try:
        print(f"💾 保存動畫: {output_path}")
        anim.save(output_path, writer='ffmpeg', fps=10, dpi=150)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ 保存成功!")
        print(f"📁 文件: {output_path}")
        print(f"📊 大小: {file_size:.2f}MB")
        
        # 統計分析
        if trajectory:
            initial_pos = trajectory[0]
            final_pos = trajectory[-1]
            total_displacement = np.mean([
                np.linalg.norm(final_pos[i] - initial_pos[i]) 
                for i in range(len(initial_pos))
            ])
            print(f"📈 平均總位移: {total_displacement:.4f}")
            
            if total_displacement < 0.01:
                print(f"⚠️ 警告: 智能體幾乎靜止，可能存在模型問題")
            else:
                print(f"✅ 智能體正常移動")
        
    except Exception as e:
        print(f"❌ 保存失敗: {e}")
        # 嘗試保存為GIF
        try:
            gif_path = 'SMART_FINAL_COLLABORATION_RESULT.gif'
            anim.save(gif_path, writer='pillow', fps=8)
            print(f"✅ 已保存為GIF: {gif_path}")
        except Exception as e2:
            print(f"❌ GIF保存也失敗: {e2}")
    
    plt.close()


if __name__ == '__main__':
    main()
 
"""
智能模型加載器 - 根據實際權重完全重建網絡架構
"""

import torch
import torch.nn as nn
import yaml
import os
from gcbfplus.env import DoubleIntegratorEnv
from gcbfplus.policy import BPTTPolicy, create_policy_from_config
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def analyze_model_architecture(policy_path):
    """
    分析模型權重來推斷完整的網絡架構
    """
    print(f"🔍 分析模型架構: {policy_path}")
    
    state_dict = torch.load(policy_path, map_location='cpu', weights_only=True)
    
    print(f"📋 模型權重鍵:")
    for key, tensor in state_dict.items():
        print(f"  {key}: {tensor.shape}")
    
    # 分析perception層
    perception_input_dim = None
    perception_output_dim = None
    perception_layers = []
    
    if 'perception.mlp.0.weight' in state_dict:
        perception_input_dim = state_dict['perception.mlp.0.weight'].shape[1]
        perception_output_dim = state_dict['perception.mlp.0.weight'].shape[0]
        
    # 找到所有perception層
    for key in state_dict.keys():
        if key.startswith('perception.mlp.') and key.endswith('.weight'):
            layer_num = int(key.split('.')[2])
            out_features = state_dict[key].shape[0]
            perception_layers.append((layer_num, out_features))
    
    perception_layers.sort()
    perception_hidden_dims = [dim for _, dim in perception_layers[:-1]]  # 除了最後一層
    
    # 分析memory層
    memory_hidden_dim = None
    if 'memory.gru.weight_hh_l0' in state_dict:
        memory_hidden_dim = state_dict['memory.gru.weight_hh_l0'].shape[1]
    
    # 分析policy_head層
    policy_head_layers = []
    action_layers = []
    alpha_layers = []
    
    # 找action_layers
    for key in state_dict.keys():
        if key.startswith('policy_head.action_layers.') and key.endswith('.weight'):
            # 從 'policy_head.action_layers.N.weight' 中提取 N
            parts = key.split('.')
            if len(parts) >= 4 and parts[3].isdigit():
                layer_num = int(parts[3])
                out_features = state_dict[key].shape[0]
                action_layers.append((layer_num, out_features))
    
    action_layers.sort()
    
    # 找alpha_network
    for key in state_dict.keys():
        if key.startswith('policy_head.alpha_network.') and key.endswith('.weight'):
            # 從 'policy_head.alpha_network.N.weight' 中提取 N
            parts = key.split('.')
            if len(parts) >= 4 and parts[3].isdigit():
                layer_num = int(parts[3])
                out_features = state_dict[key].shape[0]
                alpha_layers.append((layer_num, out_features))
    
    alpha_layers.sort()
    
    # 構建完整配置
    config = {
        'perception': {
            'use_vision': False,
            'input_dim': perception_input_dim,
            'output_dim': perception_output_dim,
            'hidden_dims': perception_hidden_dims,
            'activation': 'relu'
        },
        'memory': {
            'hidden_dim': memory_hidden_dim,
            'num_layers': 1
        },
        'policy_head': {
            'output_dim': action_layers[-1][1] if action_layers else 2,  # 最後一層是輸出維度
            'hidden_dims': [dim for _, dim in action_layers[:-1]] if len(action_layers) > 1 else [perception_output_dim],
            'activation': 'relu',
            'predict_alpha': True,
            'alpha_hidden_dims': [dim for _, dim in alpha_layers[:-1]] if len(alpha_layers) > 1 else [perception_output_dim]
        }
    }
    
    print(f"✅ 推斷的網絡架構:")
    print(f"  Perception: 輸入={perception_input_dim}, 輸出={perception_output_dim}, 隱藏層={perception_hidden_dims}")
    print(f"  Memory: 隱藏維度={memory_hidden_dim}")
    print(f"  Action層: {[dim for _, dim in action_layers]}")
    print(f"  Alpha層: {[dim for _, dim in alpha_layers]}")
    
    return config


def create_exact_model(model_path, device='cpu'):
    """
    根據實際模型權重創建完全匹配的網絡
    """
    print(f"🎯 創建精確匹配的模型")
    
    # 分析架構
    policy_config = analyze_model_architecture(model_path)
    
    # 創建網絡
    policy_network = create_policy_from_config(policy_config)
    policy_network = policy_network.to(device)
    
    # 加載權重
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    try:
        policy_network.load_state_dict(state_dict, strict=False)
        print(f"✅ 模型加載成功 (strict=False)")
    except Exception as e:
        print(f"❌ 模型加載失败: {e}")
        return None, None
    
    return policy_network, policy_config


def create_simple_visualization(env, policy_network, device, num_steps=100):
    """
    創建簡單的可視化
    """
    print(f"🎬 開始簡單可視化 ({num_steps} 步)")
    
    policy_network.eval()
    
    # 初始化環境
    state = env.reset()
    
    trajectory_positions = []
    
    with torch.no_grad():
        for step in range(num_steps):
            # 記錄位置
            current_positions = state.positions[0].cpu().numpy()
            trajectory_positions.append(current_positions.copy())
            
            # 獲取觀測
            observations = env.get_observations(state).to(device)
            
            # 策略推理
            try:
                policy_output = policy_network(observations, state)
                actions = policy_output.actions
                alphas = getattr(policy_output, 'alphas', torch.ones_like(actions[:, :, :1]) * 0.5)
                
                # 檢查動作
                action_magnitude = torch.norm(actions, dim=-1).mean().item()
                if step % 20 == 0:
                    print(f"步驟 {step}: 動作強度={action_magnitude:.6f}")
                
                # 環境步進
                step_result = env.step(state, actions, alphas)
                state = step_result.next_state
                
            except Exception as e:
                print(f"❌ 步驟 {step} 失敗: {e}")
                break
    
    # 創建動畫
    print(f"🎨 創建動畫...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('🎯 統一代碼路徑 - 真實模型可視化', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加障碍物
    obstacles = [
        {'pos': [0.0, -0.8], 'radius': 0.4},
        {'pos': [0.0, 0.8], 'radius': 0.4}
    ]
    
    for obs in obstacles:
        circle = plt.Circle(obs['pos'], obs['radius'], color='red', alpha=0.8)
        ax.add_patch(circle)
    
    # 起始和目標區域
    start_zone = plt.Rectangle((-2.5, -1.5), 1.0, 3.0, fill=False, 
                              edgecolor='green', linestyle='--', linewidth=3, 
                              alpha=0.9, label='起始區域')
    ax.add_patch(start_zone)
    
    target_zone = plt.Rectangle((1.5, -1.5), 1.0, 3.0, fill=False, 
                               edgecolor='blue', linestyle='--', linewidth=3, 
                               alpha=0.9, label='目標區域')
    ax.add_patch(target_zone)
    
    # 智能體顏色
    num_agents = len(trajectory_positions[0])
    colors = ['#FF4444', '#44FF44', '#4444FF', '#FFAA44', '#FF44AA', '#44AAFF'][:num_agents]
    
    # 軌跡線和點
    trail_lines = []
    agent_dots = []
    
    for i in range(num_agents):
        line, = ax.plot([], [], '-', color=colors[i], alpha=0.8, linewidth=2,
                       label=f'智能體{i+1}' if i < 3 else "")
        trail_lines.append(line)
        
        dot, = ax.plot([], [], 'o', color=colors[i], markersize=12, 
                      markeredgecolor='black', markeredgewidth=2, zorder=5)
        agent_dots.append(dot)
    
    ax.legend()
    
    # 添加信息文本
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame):
        if frame >= len(trajectory_positions):
            return trail_lines + agent_dots + [info_text]
        
        current_pos = trajectory_positions[frame]
        
        # 更新軌跡和智能體位置
        for i in range(num_agents):
            trail_x = [pos[i, 0] for pos in trajectory_positions[:frame+1]]
            trail_y = [pos[i, 1] for pos in trajectory_positions[:frame+1]]
            trail_lines[i].set_data(trail_x, trail_y)
            
            agent_dots[i].set_data([current_pos[i, 0]], [current_pos[i, 1]])
        
        # 計算總位移
        if frame > 0:
            initial_pos = trajectory_positions[0]
            total_displacement = np.mean([
                np.linalg.norm(current_pos[i] - initial_pos[i]) 
                for i in range(num_agents)
            ])
        else:
            total_displacement = 0
        
        info_text.set_text(f'步驟: {frame}\n智能體數: {num_agents}\n平均位移: {total_displacement:.3f}')
        
        return trail_lines + agent_dots + [info_text]
    
    anim = FuncAnimation(fig, animate, frames=len(trajectory_positions),
                        interval=100, blit=False, repeat=True)
    
    return anim, trajectory_positions


def main():
    """
    主函數
    """
    print(f"🎯 智能模型加載器")
    print(f"=" * 60)
    
    # 設置
    model_dir = 'logs/full_collaboration_training'
    device = torch.device('cpu')
    
    # 找到最新模型
    models_dir = os.path.join(model_dir, 'models')
    steps = [int(d) for d in os.listdir(models_dir) if d.isdigit()]
    latest_step = max(steps)
    
    policy_path = os.path.join(model_dir, 'models', str(latest_step), 'policy.pt')
    print(f"📁 使用模型: {policy_path}")
    
    # 創建精確模型
    policy_network, policy_config = create_exact_model(policy_path, device)
    
    if policy_network is None:
        print(f"❌ 模型創建失敗")
        return
    
    # 創建環境
    env_config = {
        'area_size': 3.0,
        'car_radius': 0.15,
        'comm_radius': 1.0,
        'dt': 0.05,
        'mass': 0.1,
        'max_force': 1.0,
        'max_steps': 80,
        'name': 'DoubleIntegrator',
        'num_agents': 6,
        'obstacles': {
            'enabled': True,
            'bottleneck': True,
            'positions': [[0.0, -0.8], [0.0, 0.8]],
            'radii': [0.4, 0.4]
        }
    }
    
    env = DoubleIntegratorEnv(env_config)
    env = env.to(device)
    
    print(f"🌍 環境創建成功")
    print(f"📏 觀測形狀: {env.observation_shape}")
    print(f"📏 動作形狀: {env.action_shape}")
    
    # 創建可視化
    anim, trajectory = create_simple_visualization(env, policy_network, device, 120)
    
    # 保存動畫
    output_path = 'SMART_FINAL_COLLABORATION_RESULT.mp4'
    try:
        print(f"💾 保存動畫: {output_path}")
        anim.save(output_path, writer='ffmpeg', fps=10, dpi=150)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ 保存成功!")
        print(f"📁 文件: {output_path}")
        print(f"📊 大小: {file_size:.2f}MB")
        
        # 統計分析
        if trajectory:
            initial_pos = trajectory[0]
            final_pos = trajectory[-1]
            total_displacement = np.mean([
                np.linalg.norm(final_pos[i] - initial_pos[i]) 
                for i in range(len(initial_pos))
            ])
            print(f"📈 平均總位移: {total_displacement:.4f}")
            
            if total_displacement < 0.01:
                print(f"⚠️ 警告: 智能體幾乎靜止，可能存在模型問題")
            else:
                print(f"✅ 智能體正常移動")
        
    except Exception as e:
        print(f"❌ 保存失敗: {e}")
        # 嘗試保存為GIF
        try:
            gif_path = 'SMART_FINAL_COLLABORATION_RESULT.gif'
            anim.save(gif_path, writer='pillow', fps=8)
            print(f"✅ 已保存為GIF: {gif_path}")
        except Exception as e2:
            print(f"❌ GIF保存也失敗: {e2}")
    
    plt.close()


if __name__ == '__main__':
    main()
 
 
 
 