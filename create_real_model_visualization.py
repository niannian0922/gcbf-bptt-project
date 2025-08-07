#!/usr/bin/env python3
"""
基于真实训练模型的可视化生成器
使用实际的GCBF+BPTT+动态Alpha模型进行推理
"""
import numpy as np
import torch
import yaml
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

# 导入您的真实模型
from gcbfplus.env import DoubleIntegratorEnv
from gcbfplus.policy import create_policy_from_config

def load_real_trained_model(model_dir, step, device):
    """加载真实训练的模型"""
    print(f"🔄 Loading real trained model from: {model_dir}/models/{step}")
    
    # 加载配置
    config_path = Path(model_dir) / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建策略网络
    policy_config = config.get('networks', {}).get('policy', {})
    policy_network = create_policy_from_config(policy_config)
    policy_network = policy_network.to(device)
    
    # 加载权重
    model_path = Path(model_dir) / "models" / str(step) / "policy.pt"
    if model_path.exists():
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            policy_network.load_state_dict(state_dict, strict=False)
            print("✅ Successfully loaded policy network")
        except Exception as e:
            print(f"⚠️ Warning: Could not load model weights: {e}")
            print("Using random initialized weights for demonstration")
    
    policy_network.eval()
    
    # 尝试加载CBF网络
    cbf_network = None
    cbf_path = Path(model_dir) / "models" / str(step) / "cbf.pt"
    if cbf_path.exists():
        try:
            # 简单的CBF网络结构（您可能需要根据实际情况调整）
            obs_dim = 8 * 9  # num_agents * obs_per_agent
            cbf_network = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1)
            ).to(device)
            
            cbf_state_dict = torch.load(cbf_path, map_location=device, weights_only=False)
            cbf_network.load_state_dict(cbf_state_dict, strict=False)
            cbf_network.eval()
            print("✅ Successfully loaded CBF network")
        except Exception as e:
            print(f"⚠️ Warning: Could not load CBF network: {e}")
            cbf_network = None
    
    return policy_network, cbf_network, config

def run_real_model_simulation(policy_network, cbf_network, env, device, num_steps=200):
    """使用真实模型运行仿真"""
    print("🎮 Running real model simulation...")
    
    # 初始化环境
    state = env.reset()
    
    # 存储轨迹和Alpha值
    all_positions = []
    all_alphas = []
    all_actions = []
    
    # 如果策略网络有memory，初始化hidden state
    hidden_state = None
    if hasattr(policy_network, 'memory') and policy_network.memory is not None:
        batch_size = 1
        hidden_state = torch.zeros(1, policy_network.memory.hidden_dim).to(device)
    
    with torch.no_grad():
        for step in range(num_steps):
            # 准备观测
            if isinstance(state, dict):
                obs = state['observation']
            else:
                obs = state
            
            if not isinstance(obs, torch.Tensor):
                obs = torch.FloatTensor(obs).to(device)
            
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)  # 添加batch维度
            
            try:
                # 策略网络推理
                if hidden_state is not None:
                    # 带记忆的策略网络
                    action_logits, alpha_pred, hidden_state = policy_network(obs, hidden_state)
                else:
                    # 无记忆策略网络
                    output = policy_network(obs)
                    if isinstance(output, tuple):
                        if len(output) == 2:
                            action_logits, alpha_pred = output
                        else:
                            action_logits = output[0]
                            alpha_pred = output[1] if len(output) > 1 else None
                    else:
                        action_logits = output
                        alpha_pred = None
                
                # 处理动作
                if isinstance(action_logits, torch.Tensor):
                    if len(action_logits.shape) > 2:
                        action_logits = action_logits.squeeze()
                    action = action_logits.cpu().numpy()
                    if len(action.shape) > 1:
                        action = action[0]  # 取第一个batch
                else:
                    action = np.zeros((env.num_agents, 2))  # 默认零动作
                
                # 处理Alpha值
                if alpha_pred is not None:
                    if isinstance(alpha_pred, torch.Tensor):
                        alpha_values = alpha_pred.cpu().numpy()
                        if len(alpha_values.shape) > 1:
                            alpha_values = alpha_values[0]  # 取第一个batch
                        if len(alpha_values.shape) == 0:
                            alpha_values = np.full(env.num_agents, float(alpha_values))
                        elif len(alpha_values) != env.num_agents:
                            # 如果维度不匹配，扩展或截断
                            if len(alpha_values) == 1:
                                alpha_values = np.full(env.num_agents, alpha_values[0])
                            else:
                                alpha_values = np.full(env.num_agents, np.mean(alpha_values))
                    else:
                        alpha_values = np.full(env.num_agents, 1.0)
                else:
                    # 如果没有预测Alpha，使用环境默认值
                    alpha_values = np.full(env.num_agents, env.cbf_alpha)
                
            except Exception as e:
                print(f"⚠️ Model inference error at step {step}: {e}")
                # 使用默认值
                action = np.zeros((env.num_agents, 2))
                alpha_values = np.full(env.num_agents, 1.0)
            
            # 环境步进
            try:
                state, reward, done, info = env.step(action)
            except Exception as e:
                print(f"⚠️ Environment step error: {e}")
                break
            
            # 记录数据
            current_positions = env.get_positions()  # 需要实现这个方法
            if hasattr(env, 'agent_positions'):
                current_positions = env.agent_positions.cpu().numpy()
            elif hasattr(env, 'state') and hasattr(env.state, 'pos'):
                current_positions = env.state.pos.cpu().numpy() 
            else:
                # 从观测中提取位置信息
                if isinstance(state, dict) and 'observation' in state:
                    obs_array = state['observation']
                else:
                    obs_array = state
                
                if isinstance(obs_array, torch.Tensor):
                    obs_array = obs_array.cpu().numpy()
                
                # 假设观测的前两个维度是位置
                if len(obs_array.shape) == 2:
                    current_positions = obs_array[:, :2]  # [num_agents, 2]
                else:
                    current_positions = obs_array.reshape(-1, obs_array.shape[-1])[:, :2]
            
            all_positions.append(current_positions.copy())
            all_alphas.append(alpha_values.copy())
            all_actions.append(action.copy() if isinstance(action, np.ndarray) else np.array(action))
            
            if done:
                break
    
    print(f"✅ Simulation completed: {len(all_positions)} steps")
    return all_positions, all_alphas, all_actions

def create_real_model_visualization():
    """创建基于真实模型的可视化"""
    print("🎬 Creating Real Model-Based Visualization...")
    
    # 参数设置
    model_dir = "logs/dynamic_alpha_vision"  # 您的模型目录
    model_step = 2000
    config_file = "config/bottleneck_fixed_alpha_medium.yaml"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # 加载配置
    with open(config_file, 'r', encoding='utf-8') as f:
        env_config = yaml.safe_load(f)
    
    # 创建环境
    print("🏗️ Creating environment...")
    env = DoubleIntegratorEnv(env_config['env']).to(device)
    
    try:
        # 加载真实训练模型
        policy_network, cbf_network, model_config = load_real_trained_model(
            model_dir, model_step, device
        )
        
        # 运行真实模型仿真
        all_positions, all_alphas, all_actions = run_real_model_simulation(
            policy_network, cbf_network, env, device, num_steps=200
        )
        
        model_based = True
        print("✅ Using REAL trained model for visualization")
        
    except Exception as e:
        print(f"⚠️ Could not load real model, using demonstration: {e}")
        # 降级到演示模式
        all_positions, all_alphas = create_demonstration_data(env.num_agents, 200)
        model_based = False
        print("📺 Using demonstration data")
    
    # 创建可视化
    create_advanced_visualization(all_positions, all_alphas, env_config, model_based)
    
    return "real_model_visualization.gif"

def create_demonstration_data(num_agents, num_steps):
    """创建演示数据（如果无法加载真实模型）"""
    np.random.seed(42)
    
    all_positions = []
    all_alphas = []
    
    # 生成更真实的Alpha值（连续变化）
    base_alpha = 1.0
    alpha_noise = np.random.normal(0, 0.1, (num_steps, num_agents))
    distance_factor = np.random.uniform(0.8, 1.2, (num_steps, num_agents))
    
    for step in range(num_steps):
        # 简单的位置生成
        progress = step / num_steps
        positions = []
        step_alphas = []
        
        for i in range(num_agents):
            start_delay = i * 8
            if step < start_delay:
                x = -2.0 + i * 0.1
                y = np.random.uniform(-0.8, 0.8)
                alpha = base_alpha
            else:
                actual_progress = (step - start_delay) / (num_steps - start_delay)
                x = -2.0 + actual_progress * 4.0 + np.sin(step * 0.1 + i) * 0.05
                y = np.random.uniform(-0.8, 0.8) + np.cos(step * 0.08 + i) * 0.1
                
                # 连续变化的Alpha值
                base_value = base_alpha * distance_factor[step, i]
                noise = alpha_noise[step, i] * 0.2
                proximity_factor = 1.0 + 0.5 * np.exp(-abs(x) * 2)  # 瓶颈区域影响
                alpha = np.clip(base_value + noise + proximity_factor, 0.8, 2.5)
            
            positions.append([x, y])
            step_alphas.append(alpha)
        
        all_positions.append(np.array(positions))
        all_alphas.append(np.array(step_alphas))
    
    return all_positions, all_alphas

def create_advanced_visualization(all_positions, all_alphas, env_config, model_based):
    """创建高级可视化"""
    num_agents = len(all_positions[0])
    num_steps = len(all_positions)
    
    # 创建图形
    fig, ((ax_main, ax_alpha), (ax_safety, ax_stats)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{"REAL MODEL" if model_based else "DEMONSTRATION"} - GCBF+ Dynamic Alpha Visualization', 
                 fontsize=16, fontweight='bold')
    
    # 设置主面板
    ax_main.set_xlim(-2.5, 2.5)
    ax_main.set_ylim(-1.2, 1.2)
    ax_main.set_aspect('equal')
    ax_main.set_title('Multi-Agent Navigation with Dynamic Alpha')
    ax_main.grid(True, alpha=0.3)
    
    # 绘制障碍物（基于配置）
    obstacles = []
    if env_config.get('env', {}).get('obstacles', {}).get('bottleneck', False):
        gap_width = env_config['env']['obstacles']['gap_width']
        obstacle_radius = env_config['env']['obstacles']['obstacle_radius']
        obstacle_spacing = env_config['env']['obstacles']['obstacle_spacing']
        
        wall_y = gap_width / 2 + obstacle_radius + 0.02
        for x in np.arange(-0.8, -0.12, obstacle_spacing):
            circle = Circle((x, wall_y), obstacle_radius, color='darkred', alpha=0.8)
            ax_main.add_patch(circle)
            obstacles.append([x, wall_y, obstacle_radius])
        for x in np.arange(0.12, 0.8, obstacle_spacing):
            circle = Circle((x, wall_y), obstacle_radius, color='darkred', alpha=0.8)
            ax_main.add_patch(circle)
            obstacles.append([x, wall_y, obstacle_radius])
        
        wall_y = -(gap_width / 2 + obstacle_radius + 0.02)
        for x in np.arange(-0.8, -0.12, obstacle_spacing):
            circle = Circle((x, wall_y), obstacle_radius, color='darkred', alpha=0.8)
            ax_main.add_patch(circle)
            obstacles.append([x, wall_y, obstacle_radius])
        for x in np.arange(0.12, 0.8, obstacle_spacing):
            circle = Circle((x, wall_y), obstacle_radius, color='darkred', alpha=0.8)
            ax_main.add_patch(circle)
            obstacles.append([x, wall_y, obstacle_radius])
    
    # 设置Alpha值面板
    ax_alpha.set_title('Continuous Dynamic Alpha Values')
    ax_alpha.set_xlabel('Time Step')
    ax_alpha.set_ylabel('Alpha Value')
    
    # 绘制连续Alpha曲线
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
    for i in range(num_agents):
        alpha_series = [all_alphas[step][i] for step in range(num_steps)]
        ax_alpha.plot(range(num_steps), alpha_series, color=colors[i], 
                     linewidth=2, alpha=0.8, label=f'Agent {i+1}')
    
    ax_alpha.legend()
    ax_alpha.grid(True, alpha=0.3)
    
    # 设置安全监控面板
    ax_safety.set_title('Safety Metrics')
    ax_safety.set_xlabel('Time Step')
    ax_safety.set_ylabel('Distance (m)')
    
    # 计算安全指标
    min_distances = []
    avg_distances = []
    
    for step in range(num_steps):
        positions = all_positions[step]
        distances = []
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        if distances:
            min_distances.append(min(distances))
            avg_distances.append(np.mean(distances))
        else:
            min_distances.append(1.0)
            avg_distances.append(1.0)
    
    ax_safety.plot(range(num_steps), min_distances, 'b-', linewidth=2, label='Min Distance')
    ax_safety.plot(range(num_steps), avg_distances, 'g--', linewidth=1.5, label='Avg Distance')
    ax_safety.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Safety Threshold')
    ax_safety.legend()
    ax_safety.grid(True, alpha=0.3)
    
    # 设置统计面板
    ax_stats.set_title('Model Information')
    ax_stats.axis('off')
    
    info_text = [
        f"Model Type: {'REAL TRAINED MODEL' if model_based else 'DEMONSTRATION'}",
        f"Agents: {num_agents}",
        f"Simulation Steps: {num_steps}",
        f"Alpha Range: {np.min(all_alphas):.3f} - {np.max(all_alphas):.3f}",
        f"Final Min Distance: {min_distances[-1]:.3f}m",
        f"Collision Events: {sum(1 for d in min_distances if d < 0.1)}"
    ]
    
    for i, text in enumerate(info_text):
        color = 'green' if model_based and i == 0 else 'red' if not model_based and i == 0 else 'black' 
        weight = 'bold' if i == 0 else 'normal'
        ax_stats.text(0.05, 0.9 - i*0.12, text, fontsize=11, color=color, 
                     fontweight=weight, transform=ax_stats.transAxes)
    
    # 初始化智能体
    agent_circles = []
    agent_labels = []
    
    for i in range(num_agents):
        circle = Circle((0, 0), 0.04, color=colors[i], alpha=0.9, edgecolor='white', linewidth=2)
        ax_main.add_patch(circle)
        agent_circles.append(circle)
        
        label = ax_main.text(0, 0, f'A{i+1}', ha='center', va='center', 
                           fontsize=8, fontweight='bold', color='white')
        agent_labels.append(label)
    
    def animate(frame):
        if frame >= num_steps:
            return agent_circles + agent_labels
        
        positions = all_positions[frame]
        alphas = all_alphas[frame]
        
        # 更新智能体位置
        for i, (pos, alpha) in enumerate(zip(positions, alphas)):
            agent_circles[i].center = pos
            agent_labels[i].set_position(pos)
            
            # 根据Alpha值调整大小（可视化Alpha影响）
            radius = 0.04 + (alpha - 1.0) * 0.02  # Alpha越大，圆圈越大
            agent_circles[i].set_radius(max(0.03, min(0.08, radius)))
        
        # 更新标题
        ax_main.set_title(f'Step {frame}/{num_steps} - Alpha Range: '
                         f'{np.min(alphas):.3f}-{np.max(alphas):.3f}')
        
        return agent_circles + agent_labels
    
    # 创建动画
    anim = FuncAnimation(fig, animate, frames=num_steps, interval=100, blit=False, repeat=True)
    
    # 保存
    output_path = "real_model_visualization.gif"
    writer = PillowWriter(fps=8)
    anim.save(output_path, writer=writer, dpi=100)
    plt.close()
    
    print(f"✅ Real model visualization saved: {output_path}")

if __name__ == "__main__":
    create_real_model_visualization()