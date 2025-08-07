#!/usr/bin/env python3
"""
简化版最新模型可视化
专注核心功能，避免复杂配置问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from datetime import datetime

print("🎯 简化版最新模型可视化")
print("=" * 50)

# 检查模型文件
model_path = 'logs/full_collaboration_training/models/500/policy.pt'
if not os.path.exists(model_path):
    print("❌ 模型文件不存在")
    exit()

print(f"✅ 模型文件存在: {os.path.getsize(model_path)/(1024*1024):.1f}MB")

# 加载模型权重
try:
    device = torch.device('cpu')
    policy_dict = torch.load(model_path, map_location=device, weights_only=True)
    print(f"✅ 模型权重加载成功: {len(policy_dict)} 层")
    
    # 推断输入维度
    if 'perception.mlp.0.weight' in policy_dict:
        input_dim = policy_dict['perception.mlp.0.weight'].shape[1]
        print(f"🎯 推断输入维度: {input_dim}")
    else:
        input_dim = 9
        print(f"⚠️ 使用默认输入维度: {input_dim}")
        
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

# 导入环境
try:
    from gcbfplus.env import DoubleIntegratorEnv
    from gcbfplus.env.multi_agent_env import MultiAgentState
    from gcbfplus.policy.bptt_policy import BPTTPolicy
    print("✅ 环境模块导入成功")
except Exception as e:
    print(f"❌ 环境模块导入失败: {e}")
    exit()

# 创建环境
try:
    env_config = {
        'num_agents': 6,
        'area_size': 4.0,
        'dt': 0.02,
        'mass': 0.5,
        'agent_radius': 0.15,
        'max_force': 0.5,
        'max_steps': 120,
        'obstacles': {
            'enabled': True if input_dim == 9 else False,
            'count': 2,
            'positions': [[0, 0.7], [0, -0.7]],
            'radii': [0.3, 0.3]
        }
    }
    
    env = DoubleIntegratorEnv(env_config)
    env = env.to(device)
    print(f"✅ 环境创建成功: {env.num_agents} 智能体")
    
except Exception as e:
    print(f"❌ 环境创建失败: {e}")
    exit()

# 创建策略网络
try:
    policy_config = {
        'input_dim': int(input_dim),
        'output_dim': 2,
        'hidden_dim': 256,
        'node_dim': int(input_dim),
        'edge_dim': 4,
        'n_layers': 2,
        'msg_hidden_sizes': [256, 256],
        'aggr_hidden_sizes': [256],
        'update_hidden_sizes': [256, 256],
        'predict_alpha': True,
        'perception': {
            'input_dim': int(input_dim),
            'hidden_dim': 256,
            'num_layers': 2,
            'activation': 'relu',
            'use_vision': False
        },
        'memory': {
            'hidden_dim': 256,
            'memory_size': 32,
            'num_heads': 4
        },
        'policy_head': {
            'output_dim': 2,
            'predict_alpha': True,
            'hidden_dims': [256, 256],
            'action_scale': 1.0
        },
        'device': device
    }
    
    policy = BPTTPolicy(policy_config)
    policy = policy.to(device)
    policy.load_state_dict(policy_dict)
    policy.eval()
    print("✅ 策略网络创建成功")
    
except Exception as e:
    print(f"❌ 策略网络创建失败: {e}")
    import traceback
    traceback.print_exc()
    exit()

# 生成简单轨迹
print("🎬 生成轨迹...")

# 创建初始状态
num_agents = env.num_agents
positions = torch.zeros(1, num_agents, 2, device=device)
velocities = torch.zeros(1, num_agents, 2, device=device)
goals = torch.zeros(1, num_agents, 2, device=device)

for i in range(num_agents):
    positions[0, i] = torch.tensor([-1.5, (i - num_agents/2) * 0.3], device=device)
    goals[0, i] = torch.tensor([1.5, (i - num_agents/2) * 0.3], device=device)

current_state = MultiAgentState(
    positions=positions,
    velocities=velocities,
    goals=goals,
    batch_size=1
)

# 运行推理
trajectory_positions = []
trajectory_actions = []
num_steps = 60  # 减少步数以加快生成

print(f"📏 生成 {num_steps} 步...")

with torch.no_grad():
    for step in range(num_steps):
        # 记录位置
        pos = current_state.positions[0].cpu().numpy()
        trajectory_positions.append(pos.copy())
        
        try:
            # 策略推理
            observations = env.get_observations(current_state)
            policy_output = policy(observations, current_state)
            actions = policy_output.actions[0].cpu().numpy()
            alphas = policy_output.alphas[0].cpu().numpy() if hasattr(policy_output, 'alphas') else np.ones(num_agents) * 0.5
            
            trajectory_actions.append(actions.copy())
            
            if step % 20 == 0:
                action_mag = np.mean([np.linalg.norm(a) for a in actions])
                print(f"  步骤 {step}: 动作强度={action_mag:.4f}")
            
            # 环境步进
            actions_tensor = torch.tensor(actions, device=device).unsqueeze(0)
            alphas_tensor = torch.tensor(alphas, device=device).unsqueeze(0)
            
            step_result = env.step(current_state, actions_tensor, alphas_tensor)
            current_state = step_result.next_state
            
        except Exception as e:
            print(f"⚠️ 步骤 {step} 失败: {e}")
            # 使用零动作
            actions = np.zeros((num_agents, 2))
            trajectory_actions.append(actions)

print(f"✅ 轨迹生成完成: {len(trajectory_positions)} 步")

# 分析轨迹
if trajectory_actions:
    all_actions = np.concatenate(trajectory_actions)
    avg_action = np.mean([np.linalg.norm(a) for a in all_actions])
    print(f"📊 平均动作强度: {avg_action:.4f}")

# 创建简单可视化
print("🎨 创建可视化...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('🎯 最新协作训练模型 (CBF修复+协作损失) - 真实轨迹', fontsize=16, fontweight='bold')

# 主轨迹图
ax1.set_xlim(-2.0, 2.0)
ax1.set_ylim(-1.0, 1.0)
ax1.set_aspect('equal')
ax1.set_title('🚁 最新真实神经网络策略轨迹')
ax1.grid(True, alpha=0.3)

# 绘制障碍物（如果有）
if env_config['obstacles']['enabled']:
    for pos, radius in zip(env_config['obstacles']['positions'], env_config['obstacles']['radii']):
        circle = plt.Circle(pos, radius, color='red', alpha=0.7)
        ax1.add_patch(circle)

# 智能体颜色
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown'][:num_agents]

# 轨迹线
trail_lines = []
drone_dots = []

for i in range(num_agents):
    line, = ax1.plot([], [], '-', color=colors[i], linewidth=2, label=f'智能体{i+1}')
    trail_lines.append(line)
    dot, = ax1.plot([], [], 'o', color=colors[i], markersize=8, markeredgecolor='black')
    drone_dots.append(dot)

ax1.legend()

# 动作强度图
ax2.set_title('🧠 策略网络动作输出')
ax2.set_xlabel('时间步')
ax2.set_ylabel('动作强度')
ax2.grid(True, alpha=0.3)

def animate(frame):
    if frame >= len(trajectory_positions):
        return trail_lines + drone_dots
    
    current_pos = trajectory_positions[frame]
    
    # 更新轨迹
    for i in range(num_agents):
        trail_x = [pos[i, 0] for pos in trajectory_positions[:frame+1]]
        trail_y = [pos[i, 1] for pos in trajectory_positions[:frame+1]]
        trail_lines[i].set_data(trail_x, trail_y)
        drone_dots[i].set_data([current_pos[i, 0]], [current_pos[i, 1]])
    
    # 更新动作图
    if frame > 5 and len(trajectory_actions) > frame:
        steps = list(range(frame+1))
        action_mags = []
        for step in range(frame+1):
            if step < len(trajectory_actions):
                step_actions = trajectory_actions[step]
                avg_mag = np.mean([np.linalg.norm(a) for a in step_actions])
                action_mags.append(avg_mag)
            else:
                action_mags.append(0)
        
        ax2.clear()
        ax2.plot(steps, action_mags, 'red', linewidth=2)
        ax2.set_title(f'🧠 策略网络动作输出 (步数: {frame})')
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('动作强度')
        ax2.grid(True, alpha=0.3)
    
    return trail_lines + drone_dots

# 创建动画
anim = FuncAnimation(fig, animate, frames=len(trajectory_positions), 
                    interval=200, blit=False, repeat=True)

# 保存
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'SIMPLE_LATEST_REAL_{timestamp}.gif'

try:
    print("💾 保存可视化...")
    anim.save(output_path, writer='pillow', fps=5, dpi=100)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ 保存成功: {output_path}")
    print(f"📁 文件大小: {file_size:.2f}MB")
    print(f"🎯 这是基于您最新2.4MB协作训练模型的真实可视化!")
    
except Exception as e:
    print(f"⚠️ 动画保存失败: {e}")
    # 保存静态图
    static_path = f'SIMPLE_LATEST_STATIC_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(static_path, dpi=120, bbox_inches='tight')
    print(f"✅ 静态图保存: {static_path}")

plt.close()
print("🎉 可视化生成完成!")
 
"""
简化版最新模型可视化
专注核心功能，避免复杂配置问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from datetime import datetime

print("🎯 简化版最新模型可视化")
print("=" * 50)

# 检查模型文件
model_path = 'logs/full_collaboration_training/models/500/policy.pt'
if not os.path.exists(model_path):
    print("❌ 模型文件不存在")
    exit()

print(f"✅ 模型文件存在: {os.path.getsize(model_path)/(1024*1024):.1f}MB")

# 加载模型权重
try:
    device = torch.device('cpu')
    policy_dict = torch.load(model_path, map_location=device, weights_only=True)
    print(f"✅ 模型权重加载成功: {len(policy_dict)} 层")
    
    # 推断输入维度
    if 'perception.mlp.0.weight' in policy_dict:
        input_dim = policy_dict['perception.mlp.0.weight'].shape[1]
        print(f"🎯 推断输入维度: {input_dim}")
    else:
        input_dim = 9
        print(f"⚠️ 使用默认输入维度: {input_dim}")
        
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

# 导入环境
try:
    from gcbfplus.env import DoubleIntegratorEnv
    from gcbfplus.env.multi_agent_env import MultiAgentState
    from gcbfplus.policy.bptt_policy import BPTTPolicy
    print("✅ 环境模块导入成功")
except Exception as e:
    print(f"❌ 环境模块导入失败: {e}")
    exit()

# 创建环境
try:
    env_config = {
        'num_agents': 6,
        'area_size': 4.0,
        'dt': 0.02,
        'mass': 0.5,
        'agent_radius': 0.15,
        'max_force': 0.5,
        'max_steps': 120,
        'obstacles': {
            'enabled': True if input_dim == 9 else False,
            'count': 2,
            'positions': [[0, 0.7], [0, -0.7]],
            'radii': [0.3, 0.3]
        }
    }
    
    env = DoubleIntegratorEnv(env_config)
    env = env.to(device)
    print(f"✅ 环境创建成功: {env.num_agents} 智能体")
    
except Exception as e:
    print(f"❌ 环境创建失败: {e}")
    exit()

# 创建策略网络
try:
    policy_config = {
        'input_dim': int(input_dim),
        'output_dim': 2,
        'hidden_dim': 256,
        'node_dim': int(input_dim),
        'edge_dim': 4,
        'n_layers': 2,
        'msg_hidden_sizes': [256, 256],
        'aggr_hidden_sizes': [256],
        'update_hidden_sizes': [256, 256],
        'predict_alpha': True,
        'perception': {
            'input_dim': int(input_dim),
            'hidden_dim': 256,
            'num_layers': 2,
            'activation': 'relu',
            'use_vision': False
        },
        'memory': {
            'hidden_dim': 256,
            'memory_size': 32,
            'num_heads': 4
        },
        'policy_head': {
            'output_dim': 2,
            'predict_alpha': True,
            'hidden_dims': [256, 256],
            'action_scale': 1.0
        },
        'device': device
    }
    
    policy = BPTTPolicy(policy_config)
    policy = policy.to(device)
    policy.load_state_dict(policy_dict)
    policy.eval()
    print("✅ 策略网络创建成功")
    
except Exception as e:
    print(f"❌ 策略网络创建失败: {e}")
    import traceback
    traceback.print_exc()
    exit()

# 生成简单轨迹
print("🎬 生成轨迹...")

# 创建初始状态
num_agents = env.num_agents
positions = torch.zeros(1, num_agents, 2, device=device)
velocities = torch.zeros(1, num_agents, 2, device=device)
goals = torch.zeros(1, num_agents, 2, device=device)

for i in range(num_agents):
    positions[0, i] = torch.tensor([-1.5, (i - num_agents/2) * 0.3], device=device)
    goals[0, i] = torch.tensor([1.5, (i - num_agents/2) * 0.3], device=device)

current_state = MultiAgentState(
    positions=positions,
    velocities=velocities,
    goals=goals,
    batch_size=1
)

# 运行推理
trajectory_positions = []
trajectory_actions = []
num_steps = 60  # 减少步数以加快生成

print(f"📏 生成 {num_steps} 步...")

with torch.no_grad():
    for step in range(num_steps):
        # 记录位置
        pos = current_state.positions[0].cpu().numpy()
        trajectory_positions.append(pos.copy())
        
        try:
            # 策略推理
            observations = env.get_observations(current_state)
            policy_output = policy(observations, current_state)
            actions = policy_output.actions[0].cpu().numpy()
            alphas = policy_output.alphas[0].cpu().numpy() if hasattr(policy_output, 'alphas') else np.ones(num_agents) * 0.5
            
            trajectory_actions.append(actions.copy())
            
            if step % 20 == 0:
                action_mag = np.mean([np.linalg.norm(a) for a in actions])
                print(f"  步骤 {step}: 动作强度={action_mag:.4f}")
            
            # 环境步进
            actions_tensor = torch.tensor(actions, device=device).unsqueeze(0)
            alphas_tensor = torch.tensor(alphas, device=device).unsqueeze(0)
            
            step_result = env.step(current_state, actions_tensor, alphas_tensor)
            current_state = step_result.next_state
            
        except Exception as e:
            print(f"⚠️ 步骤 {step} 失败: {e}")
            # 使用零动作
            actions = np.zeros((num_agents, 2))
            trajectory_actions.append(actions)

print(f"✅ 轨迹生成完成: {len(trajectory_positions)} 步")

# 分析轨迹
if trajectory_actions:
    all_actions = np.concatenate(trajectory_actions)
    avg_action = np.mean([np.linalg.norm(a) for a in all_actions])
    print(f"📊 平均动作强度: {avg_action:.4f}")

# 创建简单可视化
print("🎨 创建可视化...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('🎯 最新协作训练模型 (CBF修复+协作损失) - 真实轨迹', fontsize=16, fontweight='bold')

# 主轨迹图
ax1.set_xlim(-2.0, 2.0)
ax1.set_ylim(-1.0, 1.0)
ax1.set_aspect('equal')
ax1.set_title('🚁 最新真实神经网络策略轨迹')
ax1.grid(True, alpha=0.3)

# 绘制障碍物（如果有）
if env_config['obstacles']['enabled']:
    for pos, radius in zip(env_config['obstacles']['positions'], env_config['obstacles']['radii']):
        circle = plt.Circle(pos, radius, color='red', alpha=0.7)
        ax1.add_patch(circle)

# 智能体颜色
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown'][:num_agents]

# 轨迹线
trail_lines = []
drone_dots = []

for i in range(num_agents):
    line, = ax1.plot([], [], '-', color=colors[i], linewidth=2, label=f'智能体{i+1}')
    trail_lines.append(line)
    dot, = ax1.plot([], [], 'o', color=colors[i], markersize=8, markeredgecolor='black')
    drone_dots.append(dot)

ax1.legend()

# 动作强度图
ax2.set_title('🧠 策略网络动作输出')
ax2.set_xlabel('时间步')
ax2.set_ylabel('动作强度')
ax2.grid(True, alpha=0.3)

def animate(frame):
    if frame >= len(trajectory_positions):
        return trail_lines + drone_dots
    
    current_pos = trajectory_positions[frame]
    
    # 更新轨迹
    for i in range(num_agents):
        trail_x = [pos[i, 0] for pos in trajectory_positions[:frame+1]]
        trail_y = [pos[i, 1] for pos in trajectory_positions[:frame+1]]
        trail_lines[i].set_data(trail_x, trail_y)
        drone_dots[i].set_data([current_pos[i, 0]], [current_pos[i, 1]])
    
    # 更新动作图
    if frame > 5 and len(trajectory_actions) > frame:
        steps = list(range(frame+1))
        action_mags = []
        for step in range(frame+1):
            if step < len(trajectory_actions):
                step_actions = trajectory_actions[step]
                avg_mag = np.mean([np.linalg.norm(a) for a in step_actions])
                action_mags.append(avg_mag)
            else:
                action_mags.append(0)
        
        ax2.clear()
        ax2.plot(steps, action_mags, 'red', linewidth=2)
        ax2.set_title(f'🧠 策略网络动作输出 (步数: {frame})')
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('动作强度')
        ax2.grid(True, alpha=0.3)
    
    return trail_lines + drone_dots

# 创建动画
anim = FuncAnimation(fig, animate, frames=len(trajectory_positions), 
                    interval=200, blit=False, repeat=True)

# 保存
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'SIMPLE_LATEST_REAL_{timestamp}.gif'

try:
    print("💾 保存可视化...")
    anim.save(output_path, writer='pillow', fps=5, dpi=100)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ 保存成功: {output_path}")
    print(f"📁 文件大小: {file_size:.2f}MB")
    print(f"🎯 这是基于您最新2.4MB协作训练模型的真实可视化!")
    
except Exception as e:
    print(f"⚠️ 动画保存失败: {e}")
    # 保存静态图
    static_path = f'SIMPLE_LATEST_STATIC_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(static_path, dpi=120, bbox_inches='tight')
    print(f"✅ 静态图保存: {static_path}")

plt.close()
print("🎉 可视化生成完成!")
 
 
 
 