#!/usr/bin/env python3
"""
🔧 修复边界可视化
解决无人机跑出画面的问题
确保智能体在可视范围内运动
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import yaml
import os
from datetime import datetime

def create_bounded_visualization():
    """创建边界限制的可视化"""
    print("🔧 修复边界可视化问题")
    print("=" * 50)
    print("🎯 目标: 确保无人机在画面内移动")
    print("🚁 内容: 无人机编队协作绕过障碍物")
    print("=" * 50)
    
    # 修复后的配置
    config = {
        'env': {
            'name': 'DoubleIntegrator',
            'num_agents': 6,
            'area_size': 4.0,  # 增大区域
            'dt': 0.02,  # 减小时间步长，增加稳定性
            'mass': 0.5,  # 增大质量，减少加速度
            'agent_radius': 0.15,
            'comm_radius': 1.0,
            'max_force': 0.5,  # 减小最大力
            'max_steps': 150,
            'social_radius': 0.4,
            'obstacles': {
                'enabled': True,
                'count': 2,
                'positions': [[0, 0.6], [0, -0.6]],
                'radii': [0.25, 0.25]
            }
        }
    }
    
    try:
        # 导入
        from gcbfplus.env import DoubleIntegratorEnv
        from gcbfplus.env.multi_agent_env import MultiAgentState
        from gcbfplus.policy.bptt_policy import create_policy_from_config
        
        print("✅ 导入成功")
        
        # 创建环境
        device = torch.device('cpu')
        env = DoubleIntegratorEnv(config['env'])
        env = env.to(device)
        
        print(f"✅ 环境创建成功")
        print(f"   📊 观测维度: {env.observation_shape}")
        print(f"   🎯 动作维度: {env.action_shape}")
        print(f"   ⚡ 最大力: {env.max_force}")
        print(f"   ⏰ 时间步长: {env.dt}")
        
        # 创建策略网络配置（降低动作缩放）
        policy_config = {
            'type': 'bptt',
            'hidden_dim': 128,  # 减小网络
            'input_dim': 6,
            'node_dim': 6,
            'edge_dim': 4,
            'n_layers': 1,  # 减少层数
            'msg_hidden_sizes': [128],
            'aggr_hidden_sizes': [128],
            'update_hidden_sizes': [128],
            'predict_alpha': True,
            'perception': {
                'input_dim': 6,
                'hidden_dim': 128,
                'num_layers': 1,  # 减少层数
                'activation': 'relu',
                'use_vision': False
            },
            'memory': {
                'hidden_dim': 128,
                'memory_size': 16,
                'num_heads': 2
            },
            'policy_head': {
                'output_dim': 2,
                'predict_alpha': True,
                'hidden_dims': [64],  # 减小输出层
                'action_scale': 0.2  # 大幅减小动作缩放
            }
        }
        
        # 创建策略网络
        policy = create_policy_from_config(policy_config)
        policy.eval()
        
        print("✅ 策略网络创建成功（降低动作缩放）")
        
        # 创建稳定的初始场景
        initial_state = create_stable_scenario(device, config['env']['num_agents'])
        
        print("✅ 稳定场景创建成功")
        
        # 运行稳定模拟
        trajectory_data = run_stable_simulation(env, policy, initial_state, config)
        
        print(f"✅ 稳定模拟完成: {len(trajectory_data['positions'])} 步")
        
        # 生成有界可视化
        output_file = create_bounded_animation(trajectory_data, config)
        
        print(f"🎉 有界可视化完成: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_stable_scenario(device, num_agents):
    """创建稳定的初始场景"""
    from gcbfplus.env.multi_agent_env import MultiAgentState
    
    # 更紧凑的初始位置
    positions = torch.zeros(1, num_agents, 2, device=device)
    velocities = torch.zeros(1, num_agents, 2, device=device)
    goals = torch.zeros(1, num_agents, 2, device=device)
    
    # 左侧紧凑编队
    formation_x = -1.5
    target_x = 1.5
    
    for i in range(num_agents):
        # 简单的2x3网格排列
        row = i // 3
        col = i % 3
        
        positions[0, i] = torch.tensor([
            formation_x + col * 0.2,
            (row - 0.5) * 0.4
        ], device=device)
        
        # 对应的目标位置
        goals[0, i] = torch.tensor([
            target_x + col * 0.2,
            (row - 0.5) * 0.4
        ], device=device)
    
    print(f"   📍 初始位置范围: x=[{positions[0, :, 0].min():.2f}, {positions[0, :, 0].max():.2f}]")
    print(f"   🎯 目标位置范围: x=[{goals[0, :, 0].min():.2f}, {goals[0, :, 0].max():.2f}]")
    
    return MultiAgentState(
        positions=positions,
        velocities=velocities,
        goals=goals,
        batch_size=1
    )

def run_stable_simulation(env, policy, initial_state, config):
    """运行稳定模拟"""
    num_steps = 150
    max_position = 3.0  # 位置边界
    max_velocity = 2.0  # 速度边界
    max_action = 0.3    # 动作边界
    
    trajectory_data = {
        'positions': [],
        'velocities': [],
        'actions': [],
        'goal_distances': [],
        'bounded_info': [],
        'config': config
    }
    
    current_state = initial_state
    
    with torch.no_grad():
        for step in range(num_steps):
            positions = current_state.positions[0].cpu().numpy()
            velocities = current_state.velocities[0].cpu().numpy()
            goal_positions = current_state.goals[0].cpu().numpy()
            
            trajectory_data['positions'].append(positions.copy())
            trajectory_data['velocities'].append(velocities.copy())
            
            # 获取策略动作
            try:
                observations = env.get_observation(current_state)
                actions, alphas = policy(observations)
                
                # 严格限制动作
                actions = torch.clamp(actions, -max_action, max_action)
                
            except Exception as e:
                print(f"   ⚠️ 策略失败，使用目标导向动作")
                # 安全的目标导向动作
                actions = torch.zeros(1, len(positions), 2, device=current_state.positions.device)
                for i in range(len(positions)):
                    direction = goal_positions[i] - positions[i]
                    distance = np.linalg.norm(direction)
                    if distance > 0.1:
                        # 限制动作大小
                        normalized_direction = direction / distance
                        action_magnitude = min(distance * 0.5, max_action)
                        actions[0, i] = torch.tensor(normalized_direction * action_magnitude)
            
            # 记录动作
            trajectory_data['actions'].append(actions[0].cpu().numpy())
            
            # 计算目标距离
            goal_distances = [np.linalg.norm(positions[i] - goal_positions[i]) 
                            for i in range(len(positions))]
            trajectory_data['goal_distances'].append(goal_distances)
            
            # 边界检查信息
            pos_bounds = {
                'min_x': positions[:, 0].min(),
                'max_x': positions[:, 0].max(),
                'min_y': positions[:, 1].min(),
                'max_y': positions[:, 1].max()
            }
            
            vel_magnitudes = [np.linalg.norm(vel) for vel in velocities]
            action_magnitudes = [np.linalg.norm(action) for action in actions[0].cpu().numpy()]
            
            bounded_info = {
                'step': step,
                'position_bounds': pos_bounds,
                'max_velocity': max(vel_magnitudes),
                'max_action': max(action_magnitudes),
                'avg_goal_distance': np.mean(goal_distances),
                'out_of_bounds': any(abs(p) > max_position for pos in positions for p in pos)
            }
            trajectory_data['bounded_info'].append(bounded_info)
            
            # 环境步进
            try:
                step_result = env.step(current_state, actions, alphas)
                next_state = step_result.next_state
                
                # 强制边界限制
                next_positions = next_state.positions.clone()
                next_velocities = next_state.velocities.clone()
                
                # 位置边界
                next_positions = torch.clamp(next_positions, -max_position, max_position)
                
                # 速度边界
                for i in range(next_velocities.shape[1]):
                    vel_magnitude = torch.norm(next_velocities[0, i])
                    if vel_magnitude > max_velocity:
                        next_velocities[0, i] = next_velocities[0, i] / vel_magnitude * max_velocity
                
                # 更新状态
                next_state = MultiAgentState(
                    positions=next_positions,
                    velocities=next_velocities,
                    goals=next_state.goals,
                    batch_size=next_state.batch_size
                )
                
                current_state = next_state
                
                # 显示进度
                if step % 30 == 0:
                    print(f"   步骤 {step:3d}: 位置范围=[{pos_bounds['min_x']:.2f}, {pos_bounds['max_x']:.2f}], "
                          f"目标距离={bounded_info['avg_goal_distance']:.3f}")
                
                # 检查完成
                if bounded_info['avg_goal_distance'] < 0.4:
                    print(f"   🎯 到达目标! (步数: {step+1})")
                    break
                    
            except Exception as e:
                print(f"   ⚠️ 环境步进失败: {e}")
                break
    
    return trajectory_data

def create_bounded_animation(trajectory_data, config):
    """创建有界动画"""
    positions_history = trajectory_data['positions']
    if not positions_history:
        return None
    
    num_agents = len(positions_history[0])
    num_steps = len(positions_history)
    obstacles = config['env']['obstacles']
    
    # 计算实际位置范围
    all_positions = np.concatenate(positions_history, axis=0)
    min_x, max_x = all_positions[:, 0].min() - 0.5, all_positions[:, 0].max() + 0.5
    min_y, max_y = all_positions[:, 1].min() - 0.5, all_positions[:, 1].max() + 0.5
    
    print(f"   📏 实际位置范围: x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}]")
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🚁 修复版无人机编队协作 (确保在画面内)', fontsize=16, fontweight='bold')
    
    # 主轨迹图 - 使用计算出的范围
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_y, max_y)
    ax1.set_aspect('equal')
    ax1.set_title('🚁 无人机编队协作 (有界版本)')
    ax1.grid(True, alpha=0.3)
    
    # 绘制障碍物
    for i, (pos, radius) in enumerate(zip(obstacles['positions'], obstacles['radii'])):
        circle = plt.Circle(pos, radius, color='red', alpha=0.8, 
                          label='障碍物' if i == 0 else "")
        ax1.add_patch(circle)
    
    # 绘制边界
    boundary = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                           fill=False, edgecolor='gray', linestyle='--', linewidth=2, 
                           label='可视边界')
    ax1.add_patch(boundary)
    
    # 无人机颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 初始化
    trail_lines = []
    drone_dots = []
    goal_markers = []
    
    for i in range(num_agents):
        # 轨迹
        line, = ax1.plot([], [], '-', color=colors[i % len(colors)], alpha=0.7, linewidth=2)
        trail_lines.append(line)
        
        # 无人机
        drone, = ax1.plot([], [], 'o', color=colors[i % len(colors)], markersize=12, 
                         markeredgecolor='black', markeredgewidth=2)
        drone_dots.append(drone)
        
        # 目标
        goal, = ax1.plot([], [], 's', color=colors[i % len(colors)], markersize=8, alpha=0.7)
        goal_markers.append(goal)
    
    ax1.legend()
    
    # 位置边界监控
    ax2.set_title('📍 位置边界监控')
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('位置')
    ax2.grid(True, alpha=0.3)
    
    # 速度和动作监控
    ax3.set_title('⚡ 速度与动作监控')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('幅度')
    ax3.grid(True, alpha=0.3)
    
    # 目标距离
    ax4.set_title('🎯 目标距离变化')
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('距离')
    ax4.grid(True, alpha=0.3)
    
    def animate(frame):
        if frame >= num_steps:
            return trail_lines + drone_dots
        
        current_positions = positions_history[frame]
        
        # 更新无人机和轨迹
        for i, (line, drone, goal) in enumerate(zip(trail_lines, drone_dots, goal_markers)):
            if i < len(current_positions):
                # 轨迹
                trail_x = [pos[i, 0] for pos in positions_history[:frame+1]]
                trail_y = [pos[i, 1] for pos in positions_history[:frame+1]]
                line.set_data(trail_x, trail_y)
                
                # 无人机
                drone.set_data([current_positions[i, 0]], [current_positions[i, 1]])
                
                # 目标 (基于实际目标位置)
                if frame < len(trajectory_data['goal_distances']):
                    # 使用初始目标位置的估算
                    goal_x = 1.5 + (i % 3) * 0.2
                    goal_y = ((i // 3) - 0.5) * 0.4
                    goal.set_data([goal_x], [goal_y])
        
        # 更新监控图表
        if frame > 0 and trajectory_data['bounded_info']:
            steps = list(range(min(frame+1, len(trajectory_data['bounded_info']))))
            
            # 位置边界
            bounds_info = trajectory_data['bounded_info'][:frame+1]
            min_xs = [info['position_bounds']['min_x'] for info in bounds_info]
            max_xs = [info['position_bounds']['max_x'] for info in bounds_info]
            min_ys = [info['position_bounds']['min_y'] for info in bounds_info]
            max_ys = [info['position_bounds']['max_y'] for info in bounds_info]
            
            ax2.clear()
            ax2.plot(steps, min_xs, 'b-', label='最小X', alpha=0.7)
            ax2.plot(steps, max_xs, 'b--', label='最大X', alpha=0.7)
            ax2.plot(steps, min_ys, 'r-', label='最小Y', alpha=0.7)
            ax2.plot(steps, max_ys, 'r--', label='最大Y', alpha=0.7)
            ax2.axhline(y=3.0, color='gray', linestyle=':', alpha=0.5, label='边界')
            ax2.axhline(y=-3.0, color='gray', linestyle=':', alpha=0.5)
            ax2.set_title(f'📍 位置边界监控 (步数: {frame})')
            ax2.set_xlabel('时间步')
            ax2.set_ylabel('位置')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 速度和动作
            max_vels = [info['max_velocity'] for info in bounds_info]
            max_actions = [info['max_action'] for info in bounds_info]
            
            ax3.clear()
            ax3.plot(steps, max_vels, 'g-', linewidth=2, label='最大速度')
            ax3.plot(steps, max_actions, 'orange', linewidth=2, label='最大动作')
            ax3.set_title(f'⚡ 速度与动作监控 (步数: {frame})')
            ax3.set_xlabel('时间步')
            ax3.set_ylabel('幅度')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 目标距离
            if len(trajectory_data['goal_distances']) > frame:
                avg_goal_dists = [np.mean(dists) for dists in trajectory_data['goal_distances'][:frame+1]]
                ax4.clear()
                ax4.plot(steps, avg_goal_dists, 'purple', linewidth=2, label='平均目标距离')
                ax4.set_title(f'🎯 目标距离变化 (步数: {frame})')
                ax4.set_xlabel('时间步')
                ax4.set_ylabel('距离')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        return trail_lines + drone_dots
    
    # 创建动画
    anim = FuncAnimation(fig, animate, frames=num_steps, 
                        interval=150, blit=False, repeat=True)
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"BOUNDED_DRONE_FORMATION_{timestamp}.gif"
    
    try:
        print(f"💾 保存有界可视化...")
        anim.save(output_path, writer='pillow', fps=6, dpi=120)
        print(f"✅ 保存成功: {output_path}")
        
        # 最终分析
        final_info = trajectory_data['bounded_info'][-1] if trajectory_data['bounded_info'] else {}
        print(f"📊 最终状态:")
        print(f"   位置范围: x=[{final_info.get('position_bounds', {}).get('min_x', 0):.2f}, "
              f"{final_info.get('position_bounds', {}).get('max_x', 0):.2f}]")
        print(f"   最大速度: {final_info.get('max_velocity', 0):.3f}")
        print(f"   最大动作: {final_info.get('max_action', 0):.3f}")
        print(f"   目标距离: {final_info.get('avg_goal_distance', 0):.3f}")
        print(f"   超出边界: {'❌ 否' if not final_info.get('out_of_bounds', True) else '⚠️ 是'}")
        
    except Exception as e:
        print(f"⚠️ 保存失败: {e}")
        # 保存静态图
        static_path = f"BOUNDED_STATIC_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(static_path, dpi=150, bbox_inches='tight')
        print(f"✅ 静态图保存: {static_path}")
        output_path = static_path
    
    plt.close()
    return output_path

if __name__ == "__main__":
    print("🔧 修复边界可视化系统")
    print("解决无人机跑出画面的问题")
    print("确保智能体在可视范围内协作")
    print("=" * 60)
    
    success = create_bounded_visualization()
    
    if success:
        print(f"\n🎉 边界问题修复成功!")
        print(f"🚁 生成了确保在画面内的无人机编队协作可视化")
        print(f"📏 智能体现在会保持在可视范围内")
        print(f"📁 检查新的GIF文件")
    else:
        print(f"\n❌ 修复失败，需要进一步调试")
 
 
 
 