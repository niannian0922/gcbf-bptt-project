#!/usr/bin/env python3
"""
🎯 保证运动的可视化
确保智能体有明显、可见的运动
基于简单但有效的物理模拟
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime

def create_guaranteed_moving_visualization():
    """创建保证运动的可视化"""
    print("🎯 保证运动的可视化生成器")
    print("=" * 60)
    print("🚀 特点: 确保智能体有明显可见的运动")
    print("🚁 内容: 无人机编队协作绕过障碍物")
    print("✅ 保证: 100%动态，绝不静态")
    print("=" * 60)
    
    # 创建确保运动的场景
    print("🎬 创建保证运动的场景...")
    scenario_data = create_guaranteed_moving_scenario()
    
    # 生成保证运动的轨迹
    print("🏃 生成保证运动的轨迹...")
    trajectory_data = generate_guaranteed_movement(scenario_data)
    
    # 创建动态可视化
    print("🎨 创建动态可视化...")
    output_file = create_guaranteed_animation(trajectory_data)
    
    print(f"🎉 保证运动的可视化完成: {output_file}")
    return output_file

def create_guaranteed_moving_scenario():
    """创建确保运动的场景"""
    num_agents = 6
    
    # 明确的起始和目标位置，确保有大距离移动
    scenario = {
        'num_agents': num_agents,
        'start_positions': np.array([
            [-2.5, 0.0],    # 领队
            [-2.7, 0.4],    # 右翼
            [-2.7, -0.4],   # 左翼
            [-2.9, 0.8],    # 右后
            [-2.9, -0.8],   # 左后
            [-2.5, 0.0]     # 备用
        ][:num_agents]),
        'target_positions': np.array([
            [2.5, 0.0],     # 领队目标
            [2.3, 0.4],     # 右翼目标
            [2.3, -0.4],    # 左翼目标
            [2.1, 0.8],     # 右后目标
            [2.1, -0.8],    # 左后目标
            [2.5, 0.0]      # 备用目标
        ][:num_agents]),
        'obstacles': [
            {'position': [0, 0.7], 'radius': 0.3},
            {'position': [0, -0.7], 'radius': 0.3}
        ],
        'colors': ['#FF4444', '#44FF44', '#4444FF', '#FFAA44', '#FF44AA', '#44AAFF'][:num_agents]
    }
    
    # 验证移动距离
    distances = [np.linalg.norm(scenario['target_positions'][i] - scenario['start_positions'][i]) 
                for i in range(num_agents)]
    print(f"   📏 移动距离: 平均 {np.mean(distances):.2f}, 范围 [{min(distances):.2f}, {max(distances):.2f}]")
    
    return scenario

def generate_guaranteed_movement(scenario):
    """生成保证有运动的轨迹"""
    num_agents = scenario['num_agents']
    num_steps = 150
    dt = 0.05
    
    # 初始化
    positions = np.zeros((num_steps, num_agents, 2))
    velocities = np.zeros((num_steps, num_agents, 2))
    actions = np.zeros((num_steps, num_agents, 2))
    
    # 设置初始状态
    positions[0] = scenario['start_positions'].copy()
    velocities[0] = np.zeros((num_agents, 2))
    
    print(f"   🎬 模拟 {num_steps} 步运动...")
    
    for step in range(1, num_steps):
        for agent in range(num_agents):
            # 计算各种力
            
            # 1. 目标吸引力 (主要驱动力)
            goal_vec = scenario['target_positions'][agent] - positions[step-1, agent]
            goal_distance = np.linalg.norm(goal_vec)
            
            if goal_distance > 0.1:
                goal_force = (goal_vec / goal_distance) * min(goal_distance * 2.0, 1.5)
            else:
                goal_force = np.zeros(2)
            
            # 2. 障碍物避让力
            obstacle_force = np.zeros(2)
            for obs in scenario['obstacles']:
                obs_vec = positions[step-1, agent] - np.array(obs['position'])
                obs_distance = np.linalg.norm(obs_vec)
                
                danger_distance = obs['radius'] + 0.6
                if obs_distance < danger_distance and obs_distance > 0.01:
                    # 强烈的排斥力
                    repulsion_strength = (danger_distance - obs_distance) / danger_distance
                    obstacle_force += (obs_vec / obs_distance) * repulsion_strength * 2.0
            
            # 3. 智能体之间的避让力
            agent_force = np.zeros(2)
            for other in range(num_agents):
                if agent != other:
                    diff = positions[step-1, agent] - positions[step-1, other]
                    distance = np.linalg.norm(diff)
                    
                    min_distance = 0.3
                    if distance < min_distance and distance > 0.01:
                        agent_force += (diff / distance) * 0.5
            
            # 4. 编队保持力（轻微）
            formation_force = np.zeros(2)
            if agent > 0:  # 非领队保持与领队的相对位置
                leader_pos = positions[step-1, 0]
                desired_offset = scenario['start_positions'][agent] - scenario['start_positions'][0]
                desired_pos = leader_pos + desired_offset * 0.5  # 缩小编队
                formation_vec = desired_pos - positions[step-1, agent]
                formation_force = formation_vec * 0.3
            
            # 5. 阻尼力（防止振荡）
            damping_force = -velocities[step-1, agent] * 0.2
            
            # 合成总力
            total_force = goal_force + obstacle_force + agent_force + formation_force + damping_force
            
            # 限制最大力但保证最小运动
            force_magnitude = np.linalg.norm(total_force)
            max_force = 2.0
            min_force = 0.1  # 保证最小力以确保运动
            
            if force_magnitude > max_force:
                total_force = total_force / force_magnitude * max_force
            elif force_magnitude < min_force and goal_distance > 0.2:
                # 确保有最小的目标导向力
                total_force = (goal_vec / max(goal_distance, 0.01)) * min_force
            
            actions[step, agent] = total_force
            
            # 更新运动状态
            velocities[step, agent] = velocities[step-1, agent] + total_force * dt
            
            # 限制最大速度
            vel_magnitude = np.linalg.norm(velocities[step, agent])
            max_velocity = 3.0
            if vel_magnitude > max_velocity:
                velocities[step, agent] = velocities[step, agent] / vel_magnitude * max_velocity
            
            # 更新位置
            positions[step, agent] = positions[step-1, agent] + velocities[step, agent] * dt
            
            # 软边界（防止跑出画面但不停止运动）
            boundary = 3.0
            for dim in range(2):
                if positions[step, agent, dim] > boundary:
                    positions[step, agent, dim] = boundary
                    velocities[step, agent, dim] = min(velocities[step, agent, dim], 0)
                elif positions[step, agent, dim] < -boundary:
                    positions[step, agent, dim] = -boundary
                    velocities[step, agent, dim] = max(velocities[step, agent, dim], 0)
        
        # 显示进度
        if step % 30 == 0:
            # 计算运动统计
            step_movements = np.linalg.norm(positions[step] - positions[step-1], axis=1)
            avg_movement = np.mean(step_movements)
            max_movement = np.max(step_movements)
            
            # 计算目标距离
            goal_distances = [np.linalg.norm(positions[step, i] - scenario['target_positions'][i]) 
                            for i in range(num_agents)]
            avg_goal_distance = np.mean(goal_distances)
            
            print(f"      步骤 {step:3d}: 运动 平均={avg_movement:.4f} 最大={max_movement:.4f}, 目标距离={avg_goal_distance:.3f}")
    
    # 运动质量分析
    total_movements = []
    for step in range(1, num_steps):
        step_movements = np.linalg.norm(positions[step] - positions[step-1], axis=1)
        total_movements.append(step_movements)
    
    avg_total_movement = np.mean(total_movements)
    max_total_movement = np.max(total_movements)
    
    print(f"   📊 运动质量分析:")
    print(f"      平均运动: {avg_total_movement:.4f}")
    print(f"      最大运动: {max_total_movement:.4f}")
    print(f"      运动状态: {'✅ 高动态' if avg_total_movement > 0.05 else '❌ 低动态'}")
    
    return {
        'positions': positions,
        'velocities': velocities,
        'actions': actions,
        'scenario': scenario,
        'movement_stats': {
            'avg_movement': avg_total_movement,
            'max_movement': max_total_movement
        }
    }

def create_guaranteed_animation(trajectory_data):
    """创建保证动态的动画"""
    positions = trajectory_data['positions']
    velocities = trajectory_data['velocities']
    scenario = trajectory_data['scenario']
    
    num_steps, num_agents, _ = positions.shape
    
    print(f"   🎨 创建 {num_steps} 帧动画...")
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('🚀 保证运动版 - 无人机编队协作', fontsize=18, fontweight='bold')
    
    # 主轨迹图
    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(-2.0, 2.0)
    ax1.set_aspect('equal')
    ax1.set_title('🚁 无人机编队协作 (保证运动版)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 绘制障碍物
    for i, obs in enumerate(scenario['obstacles']):
        circle = plt.Circle(obs['position'], obs['radius'], color='red', alpha=0.8, 
                          label='障碍物' if i == 0 else "")
        ax1.add_patch(circle)
    
    # 绘制起始和目标区域
    start_zone = plt.Rectangle((-3.2, -1.5), 0.8, 3.0, fill=False, 
                              edgecolor='green', linestyle='--', linewidth=2, 
                              alpha=0.8, label='起始区域')
    ax1.add_patch(start_zone)
    
    target_zone = plt.Rectangle((2.4, -1.5), 0.8, 3.0, fill=False, 
                               edgecolor='blue', linestyle='--', linewidth=2, 
                               alpha=0.8, label='目标区域')
    ax1.add_patch(target_zone)
    
    # 绘制初始目标点
    for i in range(num_agents):
        ax1.plot(scenario['target_positions'][i, 0], scenario['target_positions'][i, 1], 
                's', color=scenario['colors'][i], markersize=10, alpha=0.6)
    
    # 初始化动画元素
    trail_lines = []
    drone_dots = []
    velocity_vectors = []
    
    for i in range(num_agents):
        # 轨迹线
        line, = ax1.plot([], [], '-', color=scenario['colors'][i], alpha=0.8, linewidth=3,
                        label=f'无人机{i+1}' if i < 3 else "")
        trail_lines.append(line)
        
        # 无人机点
        drone, = ax1.plot([], [], 'o', color=scenario['colors'][i], markersize=14, 
                         markeredgecolor='black', markeredgewidth=2, zorder=5)
        drone_dots.append(drone)
        
        # 速度向量
        vector = ax1.quiver([], [], [], [], color=scenario['colors'][i], alpha=0.7, 
                          scale=5, scale_units='xy', angles='xy')
        velocity_vectors.append(vector)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 实时运动监控
    ax2.set_title('🏃 实时运动监控', fontsize=12)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('运动速度')
    ax2.grid(True, alpha=0.3)
    
    # 位置追踪
    ax3.set_title('📍 位置追踪', fontsize=12)
    ax3.set_xlabel('X坐标')
    ax3.set_ylabel('Y坐标')
    ax3.grid(True, alpha=0.3)
    
    # 任务进度
    ax4.set_title('🎯 任务完成进度', fontsize=12)
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('平均目标距离')
    ax4.grid(True, alpha=0.3)
    
    def animate(frame):
        if frame >= num_steps:
            return trail_lines + drone_dots
        
        current_positions = positions[frame]
        current_velocities = velocities[frame]
        
        # 更新轨迹和无人机
        for i in range(num_agents):
            # 轨迹
            trail_x = positions[:frame+1, i, 0]
            trail_y = positions[:frame+1, i, 1]
            trail_lines[i].set_data(trail_x, trail_y)
            
            # 无人机位置
            drone_dots[i].set_data([current_positions[i, 0]], [current_positions[i, 1]])
            
            # 速度向量
            if frame > 0:
                vel_scale = 0.3
                velocity_vectors[i].set_offsets([[current_positions[i, 0], current_positions[i, 1]]])
                velocity_vectors[i].set_UVC([current_velocities[i, 0] * vel_scale], 
                                          [current_velocities[i, 1] * vel_scale])
        
        # 更新监控图表
        if frame > 5:  # 有足够数据后才显示
            steps = list(range(frame+1))
            
            # 实时运动监控
            movements = []
            for step in range(1, frame+1):
                step_movement = np.mean(np.linalg.norm(positions[step] - positions[step-1], axis=1))
                movements.append(step_movement)
            
            ax2.clear()
            ax2.plot(steps[1:], movements, 'red', linewidth=3, label='平均运动速度')
            ax2.fill_between(steps[1:], movements, alpha=0.3, color='red')
            ax2.set_title(f'🏃 实时运动监控 (步数: {frame})')
            ax2.set_xlabel('时间步')
            ax2.set_ylabel('运动速度')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 位置追踪
            ax3.clear()
            for i in range(num_agents):
                ax3.plot(positions[:frame+1, i, 0], positions[:frame+1, i, 1], 
                        color=scenario['colors'][i], alpha=0.7, linewidth=2)
                ax3.plot(current_positions[i, 0], current_positions[i, 1], 
                        'o', color=scenario['colors'][i], markersize=8)
            
            # 绘制障碍物
            for obs in scenario['obstacles']:
                circle = plt.Circle(obs['position'], obs['radius'], color='red', alpha=0.3)
                ax3.add_patch(circle)
            
            ax3.set_title(f'📍 位置追踪 (步数: {frame})')
            ax3.set_xlabel('X坐标')
            ax3.set_ylabel('Y坐标')
            ax3.set_xlim(-3.5, 3.5)
            ax3.set_ylim(-2.0, 2.0)
            ax3.grid(True, alpha=0.3)
            
            # 任务进度
            goal_distances = []
            for step in range(frame+1):
                step_goal_dists = [np.linalg.norm(positions[step, i] - scenario['target_positions'][i]) 
                                 for i in range(num_agents)]
                goal_distances.append(np.mean(step_goal_dists))
            
            ax4.clear()
            ax4.plot(steps, goal_distances, 'green', linewidth=3, label='平均目标距离')
            ax4.fill_between(steps, goal_distances, alpha=0.3, color='green')
            ax4.set_title(f'🎯 任务完成进度 (步数: {frame})')
            ax4.set_xlabel('时间步')
            ax4.set_ylabel('平均目标距离')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        return trail_lines + drone_dots
    
    # 创建动画
    print(f"   ⚡ 创建动画 (间隔: 100ms, FPS: 10)...")
    anim = FuncAnimation(fig, animate, frames=num_steps, 
                        interval=100, blit=False, repeat=True)
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"GUARANTEED_MOVING_{timestamp}.gif"
    
    try:
        print(f"💾 保存保证运动的可视化...")
        # 使用更高的FPS和DPI确保动态效果
        anim.save(output_path, writer='pillow', fps=10, dpi=150)
        
        # 检查结果
        import os
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        print(f"✅ 保存成功: {output_path}")
        print(f"📁 文件大小: {file_size:.2f}MB")
        
        # 验证动态性
        movement_stats = trajectory_data['movement_stats']
        print(f"📊 运动验证:")
        print(f"   平均运动: {movement_stats['avg_movement']:.4f}")
        print(f"   最大运动: {movement_stats['max_movement']:.4f}")
        
        if file_size > 0.5 and movement_stats['avg_movement'] > 0.02:
            print(f"✅ 验证通过: 确实是动态的!")
        else:
            print(f"⚠️ 可能仍有问题")
        
    except Exception as e:
        print(f"⚠️ 保存失败: {e}")
        # 保存静态图
        static_path = f"GUARANTEED_STATIC_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(static_path, dpi=150, bbox_inches='tight')
        print(f"✅ 静态图保存: {static_path}")
        output_path = static_path
    
    plt.close()
    return output_path

if __name__ == "__main__":
    print("🎯 保证运动的可视化系统")
    print("确保智能体有明显、可见的运动")
    print("=" * 70)
    
    output_file = create_guaranteed_moving_visualization()
    
    print(f"\n🎉 保证运动的可视化完成!")
    print(f"📁 输出文件: {output_file}")
    print(f"\n✅ 特点:")
    print(f"   🚀 保证有明显运动")
    print(f"   🚁 无人机编队协作")
    print(f"   🎯 从起点到目标的完整过程")
    print(f"   📊 实时运动监控")
    print(f"\n🎊 现在应该可以看到真正动态的协作!")
 
 
 
 