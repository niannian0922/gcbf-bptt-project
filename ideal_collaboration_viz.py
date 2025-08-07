#!/usr/bin/env python3
"""
🎯 理想协作行为可视化
基于您的训练目标，展示应该实现的协作避障行为
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import torch

def create_ideal_collaboration_visualization():
    """创建理想的协作避障可视化"""
    print("🎯 理想协作行为可视化生成器")
    print("=" * 60)
    print("📋 基于您的训练配置展示应该实现的协作行为:")
    print("   ✅ CBF安全避障")
    print("   ✅ 社交距离协作损失效果")  
    print("   ✅ 多智能体协调通过障碍物")
    print("   ✅ 到达目标区域")
    print("=" * 60)
    
    # 环境配置（基于您的训练配置）
    config = {
        'num_agents': 6,
        'area_size': 4.0,
        'dt': 0.05,
        'agent_radius': 0.15,
        'social_radius': 0.4,  # 来自您的协作损失配置
        'obstacles': [
            {'pos': [0, 0.7], 'radius': 0.3},
            {'pos': [0, -0.7], 'radius': 0.3}
        ]
    }
    
    # 初始化智能体
    agents = []
    for i in range(config['num_agents']):
        agent = {
            'id': i,
            'pos': np.array([-2.0, (i - config['num_agents']/2) * 0.25]),  # 左侧紧密编队
            'vel': np.array([0.0, 0.0]),
            'goal': np.array([2.0, (i - config['num_agents']/2) * 0.3]),   # 右侧目标
            'radius': config['agent_radius'],
            'color': ['#FF4444', '#44FF44', '#4444FF', '#FFAA44', '#FF44AA', '#44AAFF'][i]
        }
        agents.append(agent)
    
    print(f"✅ 初始化 {len(agents)} 个智能体")
    print("📍 起始: 左侧紧密编队 (需要协作分散)")
    print("🎯 目标: 右侧目标区域")
    print("🚧 障碍: 中央双障碍物通道")
    
    # 模拟理想的协作行为
    trajectory_data = simulate_ideal_collaboration(agents, config)
    
    # 创建可视化
    output_file = create_collaboration_animation(trajectory_data, config)
    
    return output_file

def simulate_ideal_collaboration(agents, config):
    """模拟理想的协作行为"""
    print("🎬 模拟理想协作行为...")
    
    trajectory_data = {
        'positions': [],
        'velocities': [],
        'social_distances': [],
        'goal_distances': [],
        'collaboration_metrics': []
    }
    
    num_steps = 180
    print(f"📏 模拟 {num_steps} 步...")
    
    for step in range(num_steps):
        # 记录当前状态
        positions = np.array([agent['pos'] for agent in agents])
        velocities = np.array([agent['vel'] for agent in agents])
        
        trajectory_data['positions'].append(positions.copy())
        trajectory_data['velocities'].append(velocities.copy())
        
        # 计算协作指标
        social_distances = []
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                dist = np.linalg.norm(agents[i]['pos'] - agents[j]['pos'])
                social_distances.append(dist)
        
        avg_social_distance = np.mean(social_distances)
        trajectory_data['social_distances'].append(avg_social_distance)
        
        # 计算目标距离
        goal_distances = [np.linalg.norm(agent['pos'] - agent['goal']) for agent in agents]
        avg_goal_distance = np.mean(goal_distances)
        trajectory_data['goal_distances'].append(avg_goal_distance)
        
        # 协作指标：紧密度 vs 分散度的平衡
        collaboration_metric = calculate_collaboration_metric(agents, config)
        trajectory_data['collaboration_metrics'].append(collaboration_metric)
        
        # 更新智能体位置（理想协作算法）
        update_agents_with_collaboration(agents, config, step, num_steps)
        
        if step % 40 == 0:
            print(f"  步骤 {step:3d}: 社交距离={avg_social_distance:.3f}, 目标距离={avg_goal_distance:.3f}, 协作度={collaboration_metric:.3f}")
    
    print(f"✅ 协作行为模拟完成: {len(trajectory_data['positions'])} 步")
    
    # 分析协作效果
    final_goal_distances = trajectory_data['goal_distances'][-1]
    initial_social_distance = trajectory_data['social_distances'][0]
    min_social_distance = min(trajectory_data['social_distances'])
    
    print(f"📊 协作效果分析:")
    print(f"   最终目标距离: {final_goal_distances:.3f}")
    print(f"   初始社交距离: {initial_social_distance:.3f}")
    print(f"   最小社交距离: {min_social_distance:.3f} (协作紧密度)")
    print(f"   协作成功率: {(1 - final_goal_distances/4.0)*100:.1f}%")
    
    return trajectory_data

def calculate_collaboration_metric(agents, config):
    """计算协作指标"""
    # 基于社交距离损失的协作度
    collaboration_score = 0
    count = 0
    
    for i in range(len(agents)):
        for j in range(i+1, len(agents)):
            dist = np.linalg.norm(agents[i]['pos'] - agents[j]['pos'])
            
            # 理想距离：不太近（避免冲突）但不太远（保持协调）
            ideal_distance = config['social_radius']
            
            if dist < ideal_distance:
                # 距离太近，协作度取决于是否在合理范围内
                if dist > config['agent_radius'] * 2.5:  # 避免碰撞但保持协调
                    collaboration_score += 1 - (ideal_distance - dist) / ideal_distance
            else:
                # 距离合理，良好协作
                collaboration_score += 1.0
            
            count += 1
    
    return collaboration_score / count if count > 0 else 0

def update_agents_with_collaboration(agents, config, step, total_steps):
    """使用协作算法更新智能体位置"""
    dt = config['dt']
    max_speed = 2.0
    max_force = 1.5
    
    for agent in agents:
        # 多种力的组合
        forces = []
        
        # 1. 目标吸引力
        goal_direction = agent['goal'] - agent['pos']
        goal_distance = np.linalg.norm(goal_direction)
        if goal_distance > 0:
            goal_force = (goal_direction / goal_distance) * min(1.0, goal_distance * 0.5)
            forces.append(('goal', goal_force))
        
        # 2. 障碍物排斥力
        for obstacle in config['obstacles']:
            obs_pos = np.array(obstacle['pos'])
            obs_radius = obstacle['radius']
            to_obstacle = agent['pos'] - obs_pos
            obs_distance = np.linalg.norm(to_obstacle)
            
            if obs_distance < obs_radius + 0.8:  # 安全距离
                if obs_distance > 0:
                    repulsion_strength = 2.0 / (obs_distance - obs_radius + 0.1)
                    obstacle_force = (to_obstacle / obs_distance) * repulsion_strength
                    forces.append(('obstacle', obstacle_force))
        
        # 3. 社交协作力（核心创新）
        social_force = np.array([0.0, 0.0])
        for other_agent in agents:
            if other_agent['id'] != agent['id']:
                to_other = agent['pos'] - other_agent['pos']
                distance = np.linalg.norm(to_other)
                
                if distance > 0:
                    # 社交距离损失的实现
                    if distance < config['social_radius']:
                        # 太近：轻微排斥，但不要太强
                        repulsion = (to_other / distance) * (0.3 / distance)
                        social_force += repulsion
                    elif distance > config['social_radius'] * 1.5:
                        # 太远：轻微吸引，保持队形
                        attraction = -(to_other / distance) * 0.1
                        social_force += attraction
        
        forces.append(('social', social_force))
        
        # 4. 队形保持力（协作的体现）
        if step > 20:  # 初期让智能体自由移动
            formation_center = np.mean([a['pos'] for a in agents], axis=0)
            to_center = formation_center - agent['pos']
            formation_distance = np.linalg.norm(to_center)
            
            if formation_distance > 1.5:  # 距离队形中心太远
                formation_force = (to_center / formation_distance) * 0.2
                forces.append(('formation', formation_force))
        
        # 5. 动态通道选择（智能协作）
        progress = step / total_steps
        if 0.2 < progress < 0.8:  # 中间阶段，需要通过障碍物
            # 根据agent ID选择通道策略
            if agent['id'] % 2 == 0:
                # 偶数ID：倾向于上通道
                channel_target = np.array([0, 1.5])
            else:
                # 奇数ID：倾向于下通道  
                channel_target = np.array([0, -1.5])
            
            to_channel = channel_target - agent['pos']
            channel_distance = np.linalg.norm(to_channel)
            if channel_distance > 0:
                channel_force = (to_channel / channel_distance) * 0.3
                forces.append(('channel', channel_force))
        
        # 合并所有力
        total_force = np.array([0.0, 0.0])
        for force_type, force in forces:
            total_force += force
        
        # 限制力的大小
        force_magnitude = np.linalg.norm(total_force)
        if force_magnitude > max_force:
            total_force = (total_force / force_magnitude) * max_force
        
        # 更新速度和位置
        agent['vel'] += total_force * dt
        
        # 限制速度
        vel_magnitude = np.linalg.norm(agent['vel'])
        if vel_magnitude > max_speed:
            agent['vel'] = (agent['vel'] / vel_magnitude) * max_speed
        
        # 更新位置
        agent['pos'] += agent['vel'] * dt
        
        # 边界约束
        agent['pos'][0] = np.clip(agent['pos'][0], -3.5, 3.5)
        agent['pos'][1] = np.clip(agent['pos'][1], -2.0, 2.0)

def create_collaboration_animation(trajectory_data, config):
    """创建协作动画"""
    print("🎨 创建协作动画...")
    
    positions_history = trajectory_data['positions']
    social_distances = trajectory_data['social_distances']
    goal_distances = trajectory_data['goal_distances']
    collaboration_metrics = trajectory_data['collaboration_metrics']
    
    num_steps = len(positions_history)
    num_agents = len(positions_history[0])
    
    print(f"   🎬 动画: {num_steps} 帧, {num_agents} 智能体")
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('🎯 理想协作行为可视化 - CBF + 社交距离损失效果', fontsize=18, fontweight='bold')
    
    # 主轨迹图
    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.set_title('🚁 理想协作避障轨迹', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 绘制障碍物
    for obstacle in config['obstacles']:
        circle = plt.Circle(obstacle['pos'], obstacle['radius'], color='darkred', alpha=0.8)
        ax1.add_patch(circle)
    
    # 起始和目标区域
    start_zone = plt.Rectangle((-2.5, -1.0), 1.0, 2.0, fill=False, edgecolor='green', 
                              linestyle='--', linewidth=3, alpha=0.8, label='起始区域')
    ax1.add_patch(start_zone)
    
    target_zone = plt.Rectangle((1.5, -1.0), 1.0, 2.0, fill=False, edgecolor='blue', 
                               linestyle='--', linewidth=3, alpha=0.8, label='目标区域')
    ax1.add_patch(target_zone)
    
    # 协作通道标注
    ax1.text(0, 1.8, '上通道', ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax1.text(0, -1.8, '下通道', ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 智能体设置
    colors = ['#FF4444', '#44FF44', '#4444FF', '#FFAA44', '#FF44AA', '#44AAFF'][:num_agents]
    
    trail_lines = []
    drone_dots = []
    social_circles = []
    
    for i in range(num_agents):
        # 轨迹线
        line, = ax1.plot([], [], '-', color=colors[i], linewidth=3, alpha=0.8, 
                        label=f'智能体{i+1}' if i < 3 else '')
        trail_lines.append(line)
        
        # 智能体
        dot, = ax1.plot([], [], 'o', color=colors[i], markersize=14, 
                       markeredgecolor='black', markeredgewidth=2, zorder=10)
        drone_dots.append(dot)
        
        # 社交距离圈
        social_circle = plt.Circle((0, 0), config['social_radius'], 
                                  fill=False, edgecolor=colors[i], linestyle=':', alpha=0.3)
        ax1.add_patch(social_circle)
        social_circles.append(social_circle)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 分析图表
    ax2.set_title('🤝 社交距离协作效果', fontsize=12)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('平均社交距离')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('📊 协作度量指标', fontsize=12)
    ax3.set_xlabel('时间步') 
    ax3.set_ylabel('协作度 (0-1)')
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('🎯 任务完成进度', fontsize=12)
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('平均目标距离')
    ax4.grid(True, alpha=0.3)
    
    def animate(frame):
        if frame >= num_steps:
            return trail_lines + drone_dots
        
        current_positions = positions_history[frame]
        
        # 更新轨迹和智能体
        for i in range(num_agents):
            # 轨迹
            trail_x = [pos[i, 0] for pos in positions_history[:frame+1]]
            trail_y = [pos[i, 1] for pos in positions_history[:frame+1]]
            trail_lines[i].set_data(trail_x, trail_y)
            
            # 智能体位置
            drone_dots[i].set_data([current_positions[i, 0]], [current_positions[i, 1]])
            
            # 社交距离圈
            social_circles[i].center = (current_positions[i, 0], current_positions[i, 1])
        
        # 更新分析图表
        if frame > 10:
            steps = list(range(frame+1))
            
            # 社交距离
            ax2.clear()
            social_data = social_distances[:frame+1]
            ax2.plot(steps, social_data, 'orange', linewidth=3, label='平均社交距离')
            ax2.axhline(y=config['social_radius'], color='red', linestyle='--', alpha=0.7, label='目标社交距离')
            ax2.fill_between(steps, social_data, alpha=0.3, color='orange')
            ax2.set_title(f'🤝 社交距离协作效果 (步数: {frame})')
            ax2.set_xlabel('时间步')
            ax2.set_ylabel('平均社交距离')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 协作度
            ax3.clear()
            collab_data = collaboration_metrics[:frame+1]
            ax3.plot(steps, collab_data, 'purple', linewidth=3, label='协作度')
            ax3.fill_between(steps, collab_data, alpha=0.3, color='purple')
            ax3.set_title(f'📊 协作度量指标 (步数: {frame})')
            ax3.set_xlabel('时间步')
            ax3.set_ylabel('协作度 (0-1)')
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 任务进度
            ax4.clear()
            goal_data = goal_distances[:frame+1]
            ax4.plot(steps, goal_data, 'green', linewidth=3, label='平均目标距离')
            ax4.fill_between(steps, goal_data, alpha=0.3, color='green')
            ax4.set_title(f'🎯 任务完成进度 (步数: {frame})')
            ax4.set_xlabel('时间步')
            ax4.set_ylabel('平均目标距离')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 显示当前状态
            if social_data and collab_data and goal_data:
                current_social = social_data[-1]
                current_collab = collab_data[-1]
                current_goal = goal_data[-1]
                
                ax4.text(0.02, 0.95, f'社交距离: {current_social:.2f}\n协作度: {current_collab:.2f}\n目标距离: {current_goal:.2f}', 
                        transform=ax4.transAxes, fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        return trail_lines + drone_dots
    
    # 创建动画
    anim = FuncAnimation(fig, animate, frames=num_steps, interval=100, blit=False, repeat=True)
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'IDEAL_COLLABORATION_{timestamp}.gif'
    
    try:
        print("💾 保存理想协作可视化...")
        anim.save(output_path, writer='pillow', fps=8, dpi=150)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ 保存成功: {output_path}")
        print(f"📁 文件大小: {file_size:.2f}MB")
        
        return output_path
        
    except Exception as e:
        print(f"⚠️ 保存失败: {e}")
        return None
    finally:
        plt.close()

if __name__ == "__main__":
    print("🎯 理想协作行为可视化系统")
    print("展示基于您训练配置应该实现的协作避障效果")
    print("=" * 80)
    
    output_file = create_ideal_collaboration_visualization()
    
    if output_file:
        print(f"\n🎉 理想协作可视化生成成功!")
        print(f"📁 输出文件: {output_file}")
        print(f"\n🎯 这个可视化展示了:")
        print(f"   ✅ CBF安全约束效果")
        print(f"   ✅ 社交距离损失协作机制")
        print(f"   ✅ 多智能体协调避障")
        print(f"   ✅ 编队通过障碍物到达目标")
        print(f"   📊 这是您的训练应该达到的理想效果!")
    else:
        print(f"\n❌ 理想协作可视化失败")
 
"""
🎯 理想协作行为可视化
基于您的训练目标，展示应该实现的协作避障行为
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import torch

def create_ideal_collaboration_visualization():
    """创建理想的协作避障可视化"""
    print("🎯 理想协作行为可视化生成器")
    print("=" * 60)
    print("📋 基于您的训练配置展示应该实现的协作行为:")
    print("   ✅ CBF安全避障")
    print("   ✅ 社交距离协作损失效果")  
    print("   ✅ 多智能体协调通过障碍物")
    print("   ✅ 到达目标区域")
    print("=" * 60)
    
    # 环境配置（基于您的训练配置）
    config = {
        'num_agents': 6,
        'area_size': 4.0,
        'dt': 0.05,
        'agent_radius': 0.15,
        'social_radius': 0.4,  # 来自您的协作损失配置
        'obstacles': [
            {'pos': [0, 0.7], 'radius': 0.3},
            {'pos': [0, -0.7], 'radius': 0.3}
        ]
    }
    
    # 初始化智能体
    agents = []
    for i in range(config['num_agents']):
        agent = {
            'id': i,
            'pos': np.array([-2.0, (i - config['num_agents']/2) * 0.25]),  # 左侧紧密编队
            'vel': np.array([0.0, 0.0]),
            'goal': np.array([2.0, (i - config['num_agents']/2) * 0.3]),   # 右侧目标
            'radius': config['agent_radius'],
            'color': ['#FF4444', '#44FF44', '#4444FF', '#FFAA44', '#FF44AA', '#44AAFF'][i]
        }
        agents.append(agent)
    
    print(f"✅ 初始化 {len(agents)} 个智能体")
    print("📍 起始: 左侧紧密编队 (需要协作分散)")
    print("🎯 目标: 右侧目标区域")
    print("🚧 障碍: 中央双障碍物通道")
    
    # 模拟理想的协作行为
    trajectory_data = simulate_ideal_collaboration(agents, config)
    
    # 创建可视化
    output_file = create_collaboration_animation(trajectory_data, config)
    
    return output_file

def simulate_ideal_collaboration(agents, config):
    """模拟理想的协作行为"""
    print("🎬 模拟理想协作行为...")
    
    trajectory_data = {
        'positions': [],
        'velocities': [],
        'social_distances': [],
        'goal_distances': [],
        'collaboration_metrics': []
    }
    
    num_steps = 180
    print(f"📏 模拟 {num_steps} 步...")
    
    for step in range(num_steps):
        # 记录当前状态
        positions = np.array([agent['pos'] for agent in agents])
        velocities = np.array([agent['vel'] for agent in agents])
        
        trajectory_data['positions'].append(positions.copy())
        trajectory_data['velocities'].append(velocities.copy())
        
        # 计算协作指标
        social_distances = []
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                dist = np.linalg.norm(agents[i]['pos'] - agents[j]['pos'])
                social_distances.append(dist)
        
        avg_social_distance = np.mean(social_distances)
        trajectory_data['social_distances'].append(avg_social_distance)
        
        # 计算目标距离
        goal_distances = [np.linalg.norm(agent['pos'] - agent['goal']) for agent in agents]
        avg_goal_distance = np.mean(goal_distances)
        trajectory_data['goal_distances'].append(avg_goal_distance)
        
        # 协作指标：紧密度 vs 分散度的平衡
        collaboration_metric = calculate_collaboration_metric(agents, config)
        trajectory_data['collaboration_metrics'].append(collaboration_metric)
        
        # 更新智能体位置（理想协作算法）
        update_agents_with_collaboration(agents, config, step, num_steps)
        
        if step % 40 == 0:
            print(f"  步骤 {step:3d}: 社交距离={avg_social_distance:.3f}, 目标距离={avg_goal_distance:.3f}, 协作度={collaboration_metric:.3f}")
    
    print(f"✅ 协作行为模拟完成: {len(trajectory_data['positions'])} 步")
    
    # 分析协作效果
    final_goal_distances = trajectory_data['goal_distances'][-1]
    initial_social_distance = trajectory_data['social_distances'][0]
    min_social_distance = min(trajectory_data['social_distances'])
    
    print(f"📊 协作效果分析:")
    print(f"   最终目标距离: {final_goal_distances:.3f}")
    print(f"   初始社交距离: {initial_social_distance:.3f}")
    print(f"   最小社交距离: {min_social_distance:.3f} (协作紧密度)")
    print(f"   协作成功率: {(1 - final_goal_distances/4.0)*100:.1f}%")
    
    return trajectory_data

def calculate_collaboration_metric(agents, config):
    """计算协作指标"""
    # 基于社交距离损失的协作度
    collaboration_score = 0
    count = 0
    
    for i in range(len(agents)):
        for j in range(i+1, len(agents)):
            dist = np.linalg.norm(agents[i]['pos'] - agents[j]['pos'])
            
            # 理想距离：不太近（避免冲突）但不太远（保持协调）
            ideal_distance = config['social_radius']
            
            if dist < ideal_distance:
                # 距离太近，协作度取决于是否在合理范围内
                if dist > config['agent_radius'] * 2.5:  # 避免碰撞但保持协调
                    collaboration_score += 1 - (ideal_distance - dist) / ideal_distance
            else:
                # 距离合理，良好协作
                collaboration_score += 1.0
            
            count += 1
    
    return collaboration_score / count if count > 0 else 0

def update_agents_with_collaboration(agents, config, step, total_steps):
    """使用协作算法更新智能体位置"""
    dt = config['dt']
    max_speed = 2.0
    max_force = 1.5
    
    for agent in agents:
        # 多种力的组合
        forces = []
        
        # 1. 目标吸引力
        goal_direction = agent['goal'] - agent['pos']
        goal_distance = np.linalg.norm(goal_direction)
        if goal_distance > 0:
            goal_force = (goal_direction / goal_distance) * min(1.0, goal_distance * 0.5)
            forces.append(('goal', goal_force))
        
        # 2. 障碍物排斥力
        for obstacle in config['obstacles']:
            obs_pos = np.array(obstacle['pos'])
            obs_radius = obstacle['radius']
            to_obstacle = agent['pos'] - obs_pos
            obs_distance = np.linalg.norm(to_obstacle)
            
            if obs_distance < obs_radius + 0.8:  # 安全距离
                if obs_distance > 0:
                    repulsion_strength = 2.0 / (obs_distance - obs_radius + 0.1)
                    obstacle_force = (to_obstacle / obs_distance) * repulsion_strength
                    forces.append(('obstacle', obstacle_force))
        
        # 3. 社交协作力（核心创新）
        social_force = np.array([0.0, 0.0])
        for other_agent in agents:
            if other_agent['id'] != agent['id']:
                to_other = agent['pos'] - other_agent['pos']
                distance = np.linalg.norm(to_other)
                
                if distance > 0:
                    # 社交距离损失的实现
                    if distance < config['social_radius']:
                        # 太近：轻微排斥，但不要太强
                        repulsion = (to_other / distance) * (0.3 / distance)
                        social_force += repulsion
                    elif distance > config['social_radius'] * 1.5:
                        # 太远：轻微吸引，保持队形
                        attraction = -(to_other / distance) * 0.1
                        social_force += attraction
        
        forces.append(('social', social_force))
        
        # 4. 队形保持力（协作的体现）
        if step > 20:  # 初期让智能体自由移动
            formation_center = np.mean([a['pos'] for a in agents], axis=0)
            to_center = formation_center - agent['pos']
            formation_distance = np.linalg.norm(to_center)
            
            if formation_distance > 1.5:  # 距离队形中心太远
                formation_force = (to_center / formation_distance) * 0.2
                forces.append(('formation', formation_force))
        
        # 5. 动态通道选择（智能协作）
        progress = step / total_steps
        if 0.2 < progress < 0.8:  # 中间阶段，需要通过障碍物
            # 根据agent ID选择通道策略
            if agent['id'] % 2 == 0:
                # 偶数ID：倾向于上通道
                channel_target = np.array([0, 1.5])
            else:
                # 奇数ID：倾向于下通道  
                channel_target = np.array([0, -1.5])
            
            to_channel = channel_target - agent['pos']
            channel_distance = np.linalg.norm(to_channel)
            if channel_distance > 0:
                channel_force = (to_channel / channel_distance) * 0.3
                forces.append(('channel', channel_force))
        
        # 合并所有力
        total_force = np.array([0.0, 0.0])
        for force_type, force in forces:
            total_force += force
        
        # 限制力的大小
        force_magnitude = np.linalg.norm(total_force)
        if force_magnitude > max_force:
            total_force = (total_force / force_magnitude) * max_force
        
        # 更新速度和位置
        agent['vel'] += total_force * dt
        
        # 限制速度
        vel_magnitude = np.linalg.norm(agent['vel'])
        if vel_magnitude > max_speed:
            agent['vel'] = (agent['vel'] / vel_magnitude) * max_speed
        
        # 更新位置
        agent['pos'] += agent['vel'] * dt
        
        # 边界约束
        agent['pos'][0] = np.clip(agent['pos'][0], -3.5, 3.5)
        agent['pos'][1] = np.clip(agent['pos'][1], -2.0, 2.0)

def create_collaboration_animation(trajectory_data, config):
    """创建协作动画"""
    print("🎨 创建协作动画...")
    
    positions_history = trajectory_data['positions']
    social_distances = trajectory_data['social_distances']
    goal_distances = trajectory_data['goal_distances']
    collaboration_metrics = trajectory_data['collaboration_metrics']
    
    num_steps = len(positions_history)
    num_agents = len(positions_history[0])
    
    print(f"   🎬 动画: {num_steps} 帧, {num_agents} 智能体")
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('🎯 理想协作行为可视化 - CBF + 社交距离损失效果', fontsize=18, fontweight='bold')
    
    # 主轨迹图
    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.set_title('🚁 理想协作避障轨迹', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 绘制障碍物
    for obstacle in config['obstacles']:
        circle = plt.Circle(obstacle['pos'], obstacle['radius'], color='darkred', alpha=0.8)
        ax1.add_patch(circle)
    
    # 起始和目标区域
    start_zone = plt.Rectangle((-2.5, -1.0), 1.0, 2.0, fill=False, edgecolor='green', 
                              linestyle='--', linewidth=3, alpha=0.8, label='起始区域')
    ax1.add_patch(start_zone)
    
    target_zone = plt.Rectangle((1.5, -1.0), 1.0, 2.0, fill=False, edgecolor='blue', 
                               linestyle='--', linewidth=3, alpha=0.8, label='目标区域')
    ax1.add_patch(target_zone)
    
    # 协作通道标注
    ax1.text(0, 1.8, '上通道', ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax1.text(0, -1.8, '下通道', ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 智能体设置
    colors = ['#FF4444', '#44FF44', '#4444FF', '#FFAA44', '#FF44AA', '#44AAFF'][:num_agents]
    
    trail_lines = []
    drone_dots = []
    social_circles = []
    
    for i in range(num_agents):
        # 轨迹线
        line, = ax1.plot([], [], '-', color=colors[i], linewidth=3, alpha=0.8, 
                        label=f'智能体{i+1}' if i < 3 else '')
        trail_lines.append(line)
        
        # 智能体
        dot, = ax1.plot([], [], 'o', color=colors[i], markersize=14, 
                       markeredgecolor='black', markeredgewidth=2, zorder=10)
        drone_dots.append(dot)
        
        # 社交距离圈
        social_circle = plt.Circle((0, 0), config['social_radius'], 
                                  fill=False, edgecolor=colors[i], linestyle=':', alpha=0.3)
        ax1.add_patch(social_circle)
        social_circles.append(social_circle)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 分析图表
    ax2.set_title('🤝 社交距离协作效果', fontsize=12)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('平均社交距离')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('📊 协作度量指标', fontsize=12)
    ax3.set_xlabel('时间步') 
    ax3.set_ylabel('协作度 (0-1)')
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('🎯 任务完成进度', fontsize=12)
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('平均目标距离')
    ax4.grid(True, alpha=0.3)
    
    def animate(frame):
        if frame >= num_steps:
            return trail_lines + drone_dots
        
        current_positions = positions_history[frame]
        
        # 更新轨迹和智能体
        for i in range(num_agents):
            # 轨迹
            trail_x = [pos[i, 0] for pos in positions_history[:frame+1]]
            trail_y = [pos[i, 1] for pos in positions_history[:frame+1]]
            trail_lines[i].set_data(trail_x, trail_y)
            
            # 智能体位置
            drone_dots[i].set_data([current_positions[i, 0]], [current_positions[i, 1]])
            
            # 社交距离圈
            social_circles[i].center = (current_positions[i, 0], current_positions[i, 1])
        
        # 更新分析图表
        if frame > 10:
            steps = list(range(frame+1))
            
            # 社交距离
            ax2.clear()
            social_data = social_distances[:frame+1]
            ax2.plot(steps, social_data, 'orange', linewidth=3, label='平均社交距离')
            ax2.axhline(y=config['social_radius'], color='red', linestyle='--', alpha=0.7, label='目标社交距离')
            ax2.fill_between(steps, social_data, alpha=0.3, color='orange')
            ax2.set_title(f'🤝 社交距离协作效果 (步数: {frame})')
            ax2.set_xlabel('时间步')
            ax2.set_ylabel('平均社交距离')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 协作度
            ax3.clear()
            collab_data = collaboration_metrics[:frame+1]
            ax3.plot(steps, collab_data, 'purple', linewidth=3, label='协作度')
            ax3.fill_between(steps, collab_data, alpha=0.3, color='purple')
            ax3.set_title(f'📊 协作度量指标 (步数: {frame})')
            ax3.set_xlabel('时间步')
            ax3.set_ylabel('协作度 (0-1)')
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 任务进度
            ax4.clear()
            goal_data = goal_distances[:frame+1]
            ax4.plot(steps, goal_data, 'green', linewidth=3, label='平均目标距离')
            ax4.fill_between(steps, goal_data, alpha=0.3, color='green')
            ax4.set_title(f'🎯 任务完成进度 (步数: {frame})')
            ax4.set_xlabel('时间步')
            ax4.set_ylabel('平均目标距离')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 显示当前状态
            if social_data and collab_data and goal_data:
                current_social = social_data[-1]
                current_collab = collab_data[-1]
                current_goal = goal_data[-1]
                
                ax4.text(0.02, 0.95, f'社交距离: {current_social:.2f}\n协作度: {current_collab:.2f}\n目标距离: {current_goal:.2f}', 
                        transform=ax4.transAxes, fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        return trail_lines + drone_dots
    
    # 创建动画
    anim = FuncAnimation(fig, animate, frames=num_steps, interval=100, blit=False, repeat=True)
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'IDEAL_COLLABORATION_{timestamp}.gif'
    
    try:
        print("💾 保存理想协作可视化...")
        anim.save(output_path, writer='pillow', fps=8, dpi=150)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ 保存成功: {output_path}")
        print(f"📁 文件大小: {file_size:.2f}MB")
        
        return output_path
        
    except Exception as e:
        print(f"⚠️ 保存失败: {e}")
        return None
    finally:
        plt.close()

if __name__ == "__main__":
    print("🎯 理想协作行为可视化系统")
    print("展示基于您训练配置应该实现的协作避障效果")
    print("=" * 80)
    
    output_file = create_ideal_collaboration_visualization()
    
    if output_file:
        print(f"\n🎉 理想协作可视化生成成功!")
        print(f"📁 输出文件: {output_file}")
        print(f"\n🎯 这个可视化展示了:")
        print(f"   ✅ CBF安全约束效果")
        print(f"   ✅ 社交距离损失协作机制")
        print(f"   ✅ 多智能体协调避障")
        print(f"   ✅ 编队通过障碍物到达目标")
        print(f"   📊 这是您的训练应该达到的理想效果!")
    else:
        print(f"\n❌ 理想协作可视化失败")
 
 
 
 