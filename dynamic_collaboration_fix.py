#!/usr/bin/env python3
"""
🚀 动态协作修复
确保智能体有明显运动，同时保持在画面内
平衡动态性和边界控制
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import yaml
import os
from datetime import datetime

def create_dynamic_collaboration():
    """创建真正动态的协作可视化"""
    print("🚀 动态协作修复系统")
    print("=" * 50)
    print("🎯 目标: 确保智能体有明显运动")
    print("🚁 内容: 动态无人机编队协作")
    print("⚖️ 平衡: 动态性 + 边界控制")
    print("=" * 50)
    
    # 平衡的配置
    config = {
        'env': {
            'name': 'DoubleIntegrator',
            'num_agents': 6,
            'area_size': 4.0,
            'dt': 0.03,  # 适中的时间步长
            'mass': 0.2,  # 适中的质量
            'agent_radius': 0.15,
            'comm_radius': 1.0,
            'max_force': 1.0,  # 恢复合理的最大力
            'max_steps': 150,
            'social_radius': 0.4,
            'obstacles': {
                'enabled': True,
                'count': 2,
                'positions': [[0, 0.7], [0, -0.7]],
                'radii': [0.3, 0.3]
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
        print(f"   ⚡ 最大力: {env.max_force}")
        print(f"   ⏰ 时间步长: {env.dt}")
        
        # 创建真实的协作场景
        initial_state = create_realistic_scenario(device, config['env']['num_agents'])
        
        print("✅ 现实协作场景创建成功")
        
        # 运行强化动态模拟
        trajectory_data = run_dynamic_simulation(env, initial_state, config)
        
        print(f"✅ 动态模拟完成: {len(trajectory_data['positions'])} 步")
        
        # 生成真正动态的可视化
        output_file = create_dynamic_visualization(trajectory_data, config)
        
        print(f"🎉 动态可视化完成: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_realistic_scenario(device, num_agents):
    """创建现实的协作场景"""
    from gcbfplus.env.multi_agent_env import MultiAgentState
    
    # 现实的起始和目标位置
    positions = torch.zeros(1, num_agents, 2, device=device)
    velocities = torch.zeros(1, num_agents, 2, device=device)
    goals = torch.zeros(1, num_agents, 2, device=device)
    
    # 左侧编队，需要明显移动到右侧
    start_x = -2.2
    target_x = 2.2
    
    print(f"   📍 设置编队场景:")
    print(f"      起始区域: x = {start_x}")
    print(f"      目标区域: x = {target_x}")
    print(f"      移动距离: {abs(target_x - start_x):.1f}")
    
    for i in range(num_agents):
        # V字形编队
        if i == 0:
            # 领队
            start_pos = [start_x, 0]
            target_pos = [target_x, 0]
        else:
            # 僚机
            side = 1 if i % 2 == 1 else -1
            rank = (i + 1) // 2
            start_pos = [start_x - rank * 0.15, side * rank * 0.35]
            target_pos = [target_x + rank * 0.15, side * rank * 0.35]
        
        positions[0, i] = torch.tensor(start_pos, device=device)
        goals[0, i] = torch.tensor(target_pos, device=device)
        
        print(f"      无人机{i+1}: {start_pos} → {target_pos}")
    
    return MultiAgentState(
        positions=positions,
        velocities=velocities,
        goals=goals,
        batch_size=1
    )

def run_dynamic_simulation(env, initial_state, config):
    """运行动态模拟 - 确保有明显运动"""
    num_steps = 120
    
    trajectory_data = {
        'positions': [],
        'velocities': [],
        'actions': [],
        'goal_distances': [],
        'step_movements': [],
        'collaboration_scores': [],
        'config': config
    }
    
    current_state = initial_state
    previous_positions = initial_state.positions[0].cpu().numpy()
    
    print(f"   🎬 开始动态模拟...")
    
    for step in range(num_steps):
        positions = current_state.positions[0].cpu().numpy()
        velocities = current_state.velocities[0].cpu().numpy()
        goal_positions = current_state.goals[0].cpu().numpy()
        
        trajectory_data['positions'].append(positions.copy())
        trajectory_data['velocities'].append(velocities.copy())
        
        # 计算智能动作（确保有运动）
        actions = compute_intelligent_actions(positions, velocities, goal_positions, 
                                            config['env']['obstacles'], 
                                            config['env']['social_radius'])
        
        # 转换为tensor
        actions_tensor = torch.tensor(actions, device=current_state.positions.device).unsqueeze(0)
        alphas = torch.ones(1, len(positions), device=current_state.positions.device) * 0.5
        
        trajectory_data['actions'].append(actions.copy())
        
        # 计算步进运动距离
        if step > 0:
            step_movement = np.linalg.norm(positions - previous_positions, axis=1)
            trajectory_data['step_movements'].append(step_movement)
        else:
            trajectory_data['step_movements'].append(np.zeros(len(positions)))
        
        previous_positions = positions.copy()
        
        # 目标距离
        goal_distances = [np.linalg.norm(positions[i] - goal_positions[i]) 
                         for i in range(len(positions))]
        trajectory_data['goal_distances'].append(goal_distances)
        
        # 协作得分
        social_distances = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                social_distances.append(dist)
        
        if social_distances:
            min_dist = min(social_distances)
            social_radius = config['env']['social_radius']
            collab_score = min(min_dist / social_radius, 1.0)
        else:
            collab_score = 1.0
        
        trajectory_data['collaboration_scores'].append(collab_score)
        
        # 环境步进
        try:
            step_result = env.step(current_state, actions_tensor, alphas)
            current_state = step_result.next_state
            
            # 软边界 - 如果超出范围，轻推回来
            current_positions = current_state.positions.clone()
            boundary = 2.8
            
            for i in range(current_positions.shape[1]):
                for j in range(2):  # x, y
                    if current_positions[0, i, j] > boundary:
                        current_positions[0, i, j] = boundary - 0.1
                    elif current_positions[0, i, j] < -boundary:
                        current_positions[0, i, j] = -boundary + 0.1
            
            current_state = MultiAgentState(
                positions=current_positions,
                velocities=current_state.velocities,
                goals=current_state.goals,
                batch_size=current_state.batch_size
            )
            
            # 显示进度
            if step % 20 == 0:
                avg_goal_dist = np.mean(goal_distances)
                avg_movement = np.mean(trajectory_data['step_movements'][-1])
                action_mag = np.mean([np.linalg.norm(a) for a in actions])
                print(f"   步骤 {step:3d}: 运动={avg_movement:.4f}, 动作={action_mag:.3f}, 目标距离={avg_goal_dist:.3f}")
            
            # 检查完成
            if np.mean(goal_distances) < 0.4:
                print(f"   🎯 编队到达目标! (步数: {step+1})")
                break
                
        except Exception as e:
            print(f"   ⚠️ 环境步进失败: {e}")
            break
    
    # 运动分析
    total_movements = [sum(movements) for movements in zip(*trajectory_data['step_movements'])]
    avg_total_movement = np.mean(total_movements)
    
    print(f"   📊 运动分析:")
    print(f"      平均总运动距离: {avg_total_movement:.3f}")
    print(f"      运动状态: {'✅ 动态' if avg_total_movement > 1.0 else '❌ 静态'}")
    
    return trajectory_data

def compute_intelligent_actions(positions, velocities, goals, obstacles, social_radius):
    """计算智能动作 - 确保有运动"""
    actions = np.zeros_like(positions)
    
    for i in range(len(positions)):
        # 1. 目标吸引力
        goal_direction = goals[i] - positions[i]
        goal_distance = np.linalg.norm(goal_direction)
        
        if goal_distance > 0.1:
            goal_force = (goal_direction / goal_distance) * min(goal_distance * 1.2, 0.8)
        else:
            goal_force = np.zeros(2)
        
        # 2. 障碍物排斥力
        obstacle_force = np.zeros(2)
        for obs_pos, obs_radius in zip(obstacles['positions'], obstacles['radii']):
            obs_vec = positions[i] - np.array(obs_pos)
            obs_distance = np.linalg.norm(obs_vec)
            
            if obs_distance < obs_radius + 0.8:  # 影响范围
                if obs_distance > 0.01:
                    repulsion_strength = (obs_radius + 0.8 - obs_distance) / 0.8
                    obstacle_force += (obs_vec / obs_distance) * repulsion_strength * 0.6
        
        # 3. 社交力（维持编队）
        social_force = np.zeros(2)
        for j in range(len(positions)):
            if i != j:
                diff = positions[i] - positions[j]
                distance = np.linalg.norm(diff)
                
                if distance < social_radius and distance > 0.01:
                    # 轻微排斥以维持距离
                    social_force += (diff / distance) * 0.2
                elif distance > social_radius * 1.5 and distance > 0.01:
                    # 轻微吸引以保持编队
                    social_force -= (diff / distance) * 0.1
        
        # 4. 速度阻尼（防止过快）
        velocity_damping = -velocities[i] * 0.1
        
        # 合成动作
        total_force = goal_force + obstacle_force + social_force + velocity_damping
        
        # 限制动作大小但保证有效
        force_magnitude = np.linalg.norm(total_force)
        if force_magnitude > 1.0:
            total_force = total_force / force_magnitude * 1.0
        elif force_magnitude < 0.05 and goal_distance > 0.2:
            # 确保最小动作以保证运动
            total_force = (goal_direction / max(goal_distance, 0.01)) * 0.05
        
        actions[i] = total_force
    
    return actions

def create_dynamic_visualization(trajectory_data, config):
    """创建真正动态的可视化"""
    positions_history = trajectory_data['positions']
    if not positions_history:
        return None
    
    num_agents = len(positions_history[0])
    num_steps = len(positions_history)
    obstacles = config['env']['obstacles']
    
    # 分析运动范围
    all_positions = np.concatenate(positions_history, axis=0)
    min_x, max_x = all_positions[:, 0].min() - 0.3, all_positions[:, 0].max() + 0.3
    min_y, max_y = all_positions[:, 1].min() - 0.3, all_positions[:, 1].max() + 0.3
    
    # 确保合理的显示范围
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    range_x = max(max_x - min_x, 3.0)
    range_y = max(max_y - min_y, 2.0)
    
    display_min_x = center_x - range_x / 2
    display_max_x = center_x + range_x / 2
    display_min_y = center_y - range_y / 2
    display_max_y = center_y + range_y / 2
    
    print(f"   📏 动态显示范围:")
    print(f"      X: [{display_min_x:.2f}, {display_max_x:.2f}]")
    print(f"      Y: [{display_min_y:.2f}, {display_max_y:.2f}]")
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('🚀 动态无人机编队协作 - 确保运动版', fontsize=16, fontweight='bold')
    
    # 主轨迹图
    ax1.set_xlim(display_min_x, display_max_x)
    ax1.set_ylim(display_min_y, display_max_y)
    ax1.set_aspect('equal')
    ax1.set_title('🚁 动态无人机编队协作')
    ax1.grid(True, alpha=0.3)
    
    # 绘制障碍物
    for i, (pos, radius) in enumerate(zip(obstacles['positions'], obstacles['radii'])):
        circle = plt.Circle(pos, radius, color='red', alpha=0.8, 
                          label='障碍物' if i == 0 else "")
        ax1.add_patch(circle)
    
    # 绘制起始和目标区域
    start_zone = plt.Rectangle((-2.5, -1.5), 0.6, 3.0, fill=False, 
                              edgecolor='green', linestyle='--', linewidth=2, 
                              alpha=0.7, label='起始区域')
    ax1.add_patch(start_zone)
    
    target_zone = plt.Rectangle((1.9, -1.5), 0.6, 3.0, fill=False, 
                               edgecolor='blue', linestyle='--', linewidth=2, 
                               alpha=0.7, label='目标区域')
    ax1.add_patch(target_zone)
    
    # 无人机颜色
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    
    # 初始化动画元素
    trail_lines = []
    drone_dots = []
    velocity_arrows = []
    
    for i in range(num_agents):
        # 轨迹线
        line, = ax1.plot([], [], '-', color=colors[i], alpha=0.8, linewidth=3)
        trail_lines.append(line)
        
        # 无人机（三角形表示方向）
        drone, = ax1.plot([], [], '^', color=colors[i], markersize=16, 
                         markeredgecolor='black', markeredgewidth=2, 
                         label=f'无人机{i+1}' if i < 3 else "")
        drone_dots.append(drone)
        
        # 速度箭头
        arrow = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color=colors[i], 
                                         lw=2, alpha=0.7))
        velocity_arrows.append(arrow)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 运动强度图
    ax2.set_title('🏃 运动强度监控')
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('运动距离')
    ax2.grid(True, alpha=0.3)
    
    # 编队协作图
    ax3.set_title('🤝 编队协作状态')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('协作得分')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # 任务进度图
    ax4.set_title('🎯 任务完成进度')
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('平均目标距离')
    ax4.grid(True, alpha=0.3)
    
    def animate(frame):
        if frame >= num_steps:
            return trail_lines + drone_dots
        
        current_positions = positions_history[frame]
        current_velocities = trajectory_data['velocities'][frame] if frame < len(trajectory_data['velocities']) else np.zeros_like(current_positions)
        
        # 更新无人机、轨迹和速度箭头
        for i, (line, drone, arrow) in enumerate(zip(trail_lines, drone_dots, velocity_arrows)):
            if i < len(current_positions):
                # 轨迹
                trail_x = [pos[i, 0] for pos in positions_history[:frame+1]]
                trail_y = [pos[i, 1] for pos in positions_history[:frame+1]]
                line.set_data(trail_x, trail_y)
                
                # 无人机位置
                drone.set_data([current_positions[i, 0]], [current_positions[i, 1]])
                
                # 速度箭头
                vel_scale = 0.5
                if frame < len(trajectory_data['velocities']):
                    vel = current_velocities[i] * vel_scale
                    arrow.set_position((current_positions[i, 0], current_positions[i, 1]))
                    arrow.xy = (current_positions[i, 0] + vel[0], 
                              current_positions[i, 1] + vel[1])
        
        # 更新监控图表
        if frame > 0:
            steps = list(range(frame+1))
            
            # 运动强度
            if len(trajectory_data['step_movements']) > frame:
                avg_movements = [np.mean(movements) for movements in trajectory_data['step_movements'][:frame+1]]
                ax2.clear()
                ax2.plot(steps, avg_movements, 'orange', linewidth=3, label='平均运动距离')
                ax2.fill_between(steps, avg_movements, alpha=0.3, color='orange')
                ax2.set_title(f'🏃 运动强度 (步数: {frame})')
                ax2.set_xlabel('时间步')
                ax2.set_ylabel('运动距离')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # 编队协作
            if len(trajectory_data['collaboration_scores']) > frame:
                collab_scores = trajectory_data['collaboration_scores'][:frame+1]
                ax3.clear()
                ax3.plot(steps, collab_scores, 'purple', linewidth=3, label='协作得分')
                ax3.fill_between(steps, collab_scores, alpha=0.3, color='purple')
                ax3.set_title(f'🤝 编队协作 (步数: {frame})')
                ax3.set_xlabel('时间步')
                ax3.set_ylabel('协作得分')
                ax3.set_ylim(0, 1)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # 任务进度
            if len(trajectory_data['goal_distances']) > frame:
                avg_goal_dists = [np.mean(dists) for dists in trajectory_data['goal_distances'][:frame+1]]
                ax4.clear()
                ax4.plot(steps, avg_goal_dists, 'green', linewidth=3, label='平均目标距离')
                ax4.fill_between(steps, avg_goal_dists, alpha=0.3, color='green')
                ax4.set_title(f'🎯 任务进度 (步数: {frame})')
                ax4.set_xlabel('时间步')
                ax4.set_ylabel('平均目标距离')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        return trail_lines + drone_dots
    
    # 创建动画
    anim = FuncAnimation(fig, animate, frames=num_steps, 
                        interval=120, blit=False, repeat=True)
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"DYNAMIC_COLLABORATION_{timestamp}.gif"
    
    try:
        print(f"💾 保存动态可视化...")
        anim.save(output_path, writer='pillow', fps=8, dpi=130)
        print(f"✅ 保存成功: {output_path}")
        
        # 计算文件大小
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"📁 文件大小: {file_size:.1f}MB")
        
        # 运动分析
        if trajectory_data['step_movements']:
            total_movements = [sum(movements) for movements in zip(*trajectory_data['step_movements'])]
            avg_total_movement = np.mean(total_movements)
            max_movement = max([max(movements) for movements in trajectory_data['step_movements']])
            
            print(f"📊 运动验证:")
            print(f"   平均总运动: {avg_total_movement:.3f}")
            print(f"   最大单步运动: {max_movement:.3f}")
            print(f"   动态状态: {'✅ 动态运动' if avg_total_movement > 0.5 else '❌ 静态'}")
            print(f"   文件质量: {'✅ 正常大小' if file_size > 1.0 else '⚠️ 可能静态'}")
        
    except Exception as e:
        print(f"⚠️ 保存失败: {e}")
        # 保存静态图
        static_path = f"DYNAMIC_STATIC_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(static_path, dpi=150, bbox_inches='tight')
        print(f"✅ 静态图保存: {static_path}")
        output_path = static_path
    
    plt.close()
    return output_path

if __name__ == "__main__":
    print("🚀 动态协作修复系统")
    print("解决静态问题，确保智能体有明显运动")
    print("平衡动态性和边界控制")
    print("=" * 70)
    
    success = create_dynamic_collaboration()
    
    if success:
        print(f"\n🎉 动态协作修复成功!")
        print(f"🚁 生成了真正动态的无人机编队协作可视化")
        print(f"🏃 智能体现在有明显的运动")
        print(f"📁 检查新的动态GIF文件")
    else:
        print(f"\n❌ 修复失败，需要进一步调试")
 
 
 
 