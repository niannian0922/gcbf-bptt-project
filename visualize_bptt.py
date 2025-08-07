#!/usr/bin/env python3
import os
import torch
import yaml
import argparse
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import torch.nn as nn

from gcbfplus.env import DoubleIntegratorEnv
from gcbfplus.policy import BPTTPolicy, create_policy_from_config
from gcbfplus.trainer.bottleneck_metrics import BottleneckAnalyzer, BottleneckMetrics


def visualize_trajectory(env, policy_network, cbf_network, device, config, save_path=None):
    """
    Run a simulation using the trained policy and CBF networks and visualize the trajectory.
    
    Args:
        env: Environment to simulate in
        policy_network: Trained policy network
        cbf_network: Trained CBF network (can be None)
        device: Device to run computation on
        config: Configuration dictionary
        save_path: Path to save the animation
    """
    # 将网络设置为评估模式
    policy_network.eval()
    if cbf_network:
        cbf_network.eval()
    
    # 初始化环境
    state = env.reset()
    
    # 可视化参数
    simulation_steps = config.get('simulation_steps', 200)
    time_step = config.get('time_step', 0.05)
    max_agents_to_visualize = config.get('max_agents_to_visualize', None)
    
    # 如果启用，初始化瓶颈分析器
    bottleneck_analyzer = None
    if config.get('bottleneck_metrics', {}).get('enabled', False):
        bottleneck_analyzer = BottleneckAnalyzer(config['bottleneck_metrics'])
    
    # 初始化指标跟踪
    min_distances = []
    cbf_values_history = []
    control_efforts = []
    path_lengths = []
    alphas_history = []
    
    # 用于路径长度计算
    previous_positions = state.positions.clone()
    cumulative_path_lengths = torch.zeros(state.positions.shape[0], state.positions.shape[1])
    
    # 存储每步指标
    episode_rewards = []
    episode_costs = []
    episode_dones = []
    
    # 存储所有状态用于可视化
    all_positions = []
    all_goals = []
    all_actions = []
    
    # 如果可用，提取障碍物信息
    obstacles = None
    if hasattr(state, 'obstacles') and state.obstacles is not None:
        obstacles = state.obstacles[0].cpu().numpy()  # 形状 [n_obstacles, pos_dim+1]
    elif hasattr(env, 'obstacles') and env.obstacles is not None:
        obstacles = env.obstacles
    elif config.get('env', {}).get('obstacles', {}).get('bottleneck', False):
        # 基于配置生成瓶颈障碍物
        obstacles = []
    
    # 从状态获取观测
    observations = env.get_observations(state)
    
    # 将观测移动到设备以进行网络处理
    observations = observations.to(device)
    
    # 如果CBF网络可用，获取CBF值
    if cbf_network:
        cbf_values = cbf_network(observations.view(observations.shape[0], -1))
        cbf_values_history.append(cbf_values.cpu().numpy())
    
    # 从策略网络获取动作和alpha
    with torch.no_grad():
        actions, alpha = policy_network(observations)
    
    # 运行仿真并计算指标
    for step in range(simulation_steps):
        # 在步进前计算指标
        
        # 1. 最小智能体间距离
        positions = state.positions[0]  # [num_agents, 2]
        dists = torch.cdist(positions, positions)
        # 将对角线设置为大值以忽略自距离
        dists.fill_diagonal_(float('inf'))
        min_dist = torch.min(dists).item()
        min_distances.append(min_dist)
        
        # 2. 控制努力
        control_effort = torch.norm(actions, dim=-1).mean().item()
        control_efforts.append(control_effort)
        
        # 使用动态alpha进行仿真步进
        step_result = env.step(state, actions, alpha)
        
        # 3. 路径长度计算
        current_positions = step_result.next_state.positions
        step_distances = torch.norm(current_positions - previous_positions, dim=-1)
        cumulative_path_lengths += step_distances.cpu()
        previous_positions = current_positions.clone()
        
        # 存储位置用于可视化
        all_positions.append(current_positions[0].cpu().numpy())
        
        # 更新瓶颈分析器（如果启用）
        if bottleneck_analyzer:
            step_data = {
                'positions': current_positions.cpu(),
                'velocities': step_result.next_state.velocities.cpu(),
                'actions': actions.cpu(),
                'alpha': alpha.cpu() if alpha is not None else None,
                'rewards': step_result.reward.cpu(),
                'costs': step_result.cost.cpu()
            }
            bottleneck_analyzer.update(step_data)
        
        # 检查是否所有智能体都到达了目标
        if hasattr(step_result.next_state, 'goals'):
            all_goals.append(step_result.next_state.goals[0].cpu().numpy())
        else:
            all_goals.append(np.zeros_like(all_positions[-1]))  # 占位符
        
        # 检查碰撞
        if hasattr(step_result, 'info') and 'collisions' in step_result.info:
            collision_occurred = step_result.info['collisions']
        else:
            collision_occurred = step_result.cost[0].max() > 0
        
        # 更新状态
        state = step_result.next_state
        
        # 提前终止，如果达到所有目标并且我们已过了到目标的时间
        if hasattr(env, 'all_goals_reached') and env.all_goals_reached(state):
            break
        
        # 如果从未达到目标，记录最大时间
        path_lengths.append(cumulative_path_lengths.cpu().numpy())
        
    # 将列表转换为numpy数组
    all_positions = np.array(all_positions)
    all_goals = np.array(all_goals)
    
    # 创建双面板可视化布局
    fig, (ax_main, ax_metrics) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 主面板（左侧）用于动画
    ax = ax_main  # 保持向后兼容
    
    # 定义智能体和目标颜色
    agent_colors = plt.cm.tab10(np.linspace(0, 1, all_positions.shape[1]))
    goal_colors = agent_colors  # 使用相同颜色进行目标
    
    # 创建智能体补丁
    agent_patches = []
    for i in range(all_positions.shape[1]):
        if max_agents_to_visualize is None or i < max_agents_to_visualize:
            circle = plt.Circle(all_positions[0, i], env.config.get('car_radius', 0.05), 
                              color=agent_colors[i], alpha=0.8)
            ax.add_patch(circle)
            agent_patches.append(circle)
    
    # 添加目标标记
    for i in range(all_goals.shape[1]):
        if max_agents_to_visualize is None or i < max_agents_to_visualize:
            ax.plot(all_goals[-1, i, 0], all_goals[-1, i, 1], 'x', 
                   color=goal_colors[i], markersize=10, markeredgewidth=3)
    
    # 添加轨迹线
    trajectory_lines = []
    for i in range(all_positions.shape[1]):
        if max_agents_to_visualize is None or i < max_agents_to_visualize:
            line, = ax.plot([], [], color=agent_colors[i], alpha=0.5, linewidth=1)
            trajectory_lines.append(line)
    
    # 如果存在障碍物，添加障碍物
    if obstacles is not None and len(obstacles) > 0:
        for obs in obstacles:
            if len(obs) >= 3:  # [x, y, radius]
                circle = plt.Circle((obs[0], obs[1]), obs[2], 
                                  color='red', alpha=0.3)
                ax.add_patch(circle)
            elif len(obs) >= 4:  # [x, y, width, height] for rectangle
                rect = plt.Rectangle((obs[0]-obs[2]/2, obs[1]-obs[3]/2), 
                                   obs[2], obs[3], color='red', alpha=0.3)
                ax.add_patch(rect)
    
    # 设置绘图限制和标签
    ax.set_xlim(-0.5, env.config.get('area_size', 1.0) + 0.5)
    ax.set_ylim(-0.5, env.config.get('area_size', 1.0) + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 如果可用，获取可视化标题的CBF alpha值
    cbf_alpha = env.config.get('cbf_alpha', 'Dynamic')
    if hasattr(env, 'safety_layer') and hasattr(env.safety_layer, 'alpha'):
        cbf_alpha = env.safety_layer.alpha
    elif 'cbf_alpha' in config.get('env', {}):
        cbf_alpha = config['env']['cbf_alpha']
    
    title_text = f'多智能体导航 (Alpha: {cbf_alpha})'
    if bottleneck_analyzer:
        title_text += ' - 瓶颈场景'
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_xlabel('X位置')
    ax.set_ylabel('Y位置')
    
    # 添加信息显示的文本元素
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    metrics_text = ax.text(0.02, 0.85, '', transform=ax.transAxes,
                          verticalalignment='top', fontsize=9,
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 设置右侧指标面板
    ax_metrics.set_title('实时指标监控', fontsize=12, fontweight='bold')
    ax_metrics.set_xlabel('时间步')
    ax_metrics.set_ylabel('最小智能体间距离')
    ax_metrics.grid(True, alpha=0.3)
    
    # 初始化动态图表数据
    time_steps = []
    distance_values = []
    safety_threshold = 2 * env.config.get('car_radius', 0.05)  # 安全阈值
    
    # 创建空的线图对象
    distance_line, = ax_metrics.plot([], [], 'b-', linewidth=2, label='最小距离')
    threshold_line = ax_metrics.axhline(y=safety_threshold, color='r', linestyle='--', 
                                       linewidth=1, alpha=0.7, label='安全阈值')
    ax_metrics.legend()
    
    # 通信图可视化参数
    communication_range = config.get('env', {}).get('sensing_radius', 2.0)
    show_cbf_field = config.get('visualization', {}).get('show_cbf_field', True)
    highlighted_agent = config.get('visualization', {}).get('highlighted_agent', 0)
    
    def init():
        """初始化动画。"""
        for i, patch in enumerate(agent_patches):
            patch.center = all_positions[0, 0, i, 0], all_positions[0, 0, i, 1]
        for line in trajectory_lines:
            line.set_data([], [])
        info_text.set_text('')
        metrics_text.set_text('')
        
        # 初始化右侧面板
        distance_line.set_data([], [])
        ax_metrics.set_xlim(0, len(all_positions))
        if min_distances:
            ax_metrics.set_ylim(0, max(min_distances) * 1.1)
        
        return agent_patches + trajectory_lines + [info_text, metrics_text, distance_line]
    
    def animate(frame):
        """更新每一帧的动画。"""
        # 清除之前的动态元素
        ax.collections.clear()  # 清除通信线和CBF等高线
        
        # 重新添加障碍物
        if obstacles is not None and len(obstacles) > 0:
            for obs in obstacles:
                if len(obs) >= 3:  # [x, y, radius]
                    circle = plt.Circle((obs[0], obs[1]), obs[2], 
                                      color='red', alpha=0.3)
                    ax.add_patch(circle)
                elif len(obs) >= 4:  # [x, y, width, height] for rectangle
                    rect = plt.Rectangle((obs[0]-obs[2]/2, obs[1]-obs[3]/2), 
                                       obs[2], obs[3], color='red', alpha=0.3)
                    ax.add_patch(rect)
        
        # 更新智能体位置
        current_positions = np.zeros((all_positions.shape[1], 2))
        for i, patch in enumerate(agent_patches):
            if frame < len(all_positions):
                patch.center = all_positions[frame, i, 0], all_positions[frame, i, 1]
                current_positions[i] = [all_positions[frame, i, 0], all_positions[frame, i, 1]]
        
        # 绘制通信图
        if frame < len(all_positions):
            from gcbfplus.env.plot import plot_graph
            plot_graph(
                ax=ax,
                pos=current_positions,
                radius=env.config.get('car_radius', 0.05),
                color=['blue'] * len(current_positions),
                with_label=False,
                communication_range=communication_range,
                show_communication_graph=True
            )
        
        # 绘制CBF安全场（针对高亮智能体）
        if show_cbf_field and cbf_network and frame < len(all_positions) and highlighted_agent < len(current_positions):
            try:
                from gcbfplus.env.plot import plot_cbf_safety_field
                plot_cbf_safety_field(
                    ax=ax,
                    agent_positions=current_positions,
                    highlighted_agent_idx=highlighted_agent,
                    cbf_network=cbf_network,
                    env=env,
                    device=device,
                    grid_size=30,  # 降低网格大小以提高性能
                    field_radius=2.0
                )
            except Exception as e:
                print(f"CBF场可视化错误: {e}")
        
        # 更新轨迹线
        for i, line in enumerate(trajectory_lines):
            if frame < len(all_positions):
                line.set_data(all_positions[:frame+1, i, 0], all_positions[:frame+1, i, 1])
        
        # 更新文本信息
        info_text.set_text(f'时间步: {frame}\n智能体数量: {all_positions.shape[1]}\n高亮智能体: {highlighted_agent}')
        
        # 如果此帧有可用指标，更新指标文本
        if frame < len(min_distances):
            metrics_info = f'最小距离: {min_distances[frame]:.3f}\n'
            metrics_info += f'控制努力: {control_efforts[frame]:.3f}'
            if frame < len(cbf_values_history):
                avg_cbf = np.mean(cbf_values_history[frame])
                metrics_info += f'\n平均CBF值: {avg_cbf:.3f}'
            metrics_text.set_text(metrics_info)
            
            # 更新右侧面板的动态图表
            time_steps.append(frame)
            distance_values.append(min_distances[frame])
            distance_line.set_data(time_steps, distance_values)
            
            # 动态调整右侧面板的y轴范围
            if distance_values:
                y_min = min(0, min(distance_values) * 0.9)
                y_max = max(distance_values) * 1.1
                ax_metrics.set_ylim(y_min, y_max)
        
        return agent_patches + trajectory_lines + [info_text, metrics_text, distance_line]
    
    # 创建动画
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(all_positions),
                        interval=50, blit=True, repeat=True)
    
    # 如果请求，保存动画
    if save_path:
        try:
            anim.save(save_path, writer='pillow', fps=20)
            print(f'动画已保存至: {save_path}')
        except Exception as e:
            # 如果ffmpeg不可用，使用默认写入器
            print(f'保存为GIF时出错: {e}')
            anim.save(save_path, writer='pillow', fps=10)
    
    # 打印定量指标摘要
    print("\n=== 仿真指标摘要 ===")
    print(f"平均最小距离: {np.mean(min_distances):.4f}")
    print(f"最小距离的标准差: {np.std(min_distances):.4f}")
    print(f"平均控制努力: {np.mean(control_efforts):.4f}")
    if cbf_values_history:
        avg_cbf_values = [np.mean(cbf_vals) for cbf_vals in cbf_values_history]
        print(f"平均CBF值: {np.mean(avg_cbf_values):.4f}")
    
    # 计算平均路径长度
    if path_lengths:
        final_path_lengths = cumulative_path_lengths.numpy()
        avg_path_length = np.mean(final_path_lengths)
        print(f"平均路径长度: {avg_path_length:.4f}")
    
    # 如果启用，执行瓶颈分析
    bottleneck_metrics = None
    if bottleneck_analyzer:
        print("\n=== 瓶颈指标分析 ===")
        try:
            bottleneck_metrics = bottleneck_analyzer.analyze()
            
            print(f"吞吐量: {bottleneck_metrics.throughput:.4f} 智能体/秒")
            print(f"速度波动: {bottleneck_metrics.velocity_fluctuation:.4f}")
            print(f"总等待时间: {bottleneck_metrics.total_waiting_time:.4f} 秒")
            print(f"协调效率: {bottleneck_metrics.coordination_efficiency:.4f}")
            print(f"碰撞率: {bottleneck_metrics.collision_rate:.4f}")
            print(f"完成率: {bottleneck_metrics.completion_rate:.4f}")
            print(f"平均瓶颈时间: {bottleneck_metrics.avg_bottleneck_time:.4f} 秒")
            
        except Exception as e:
            print(f"瓶颈分析错误: {e}")
            bottleneck_metrics = None
    
    # 返回动画和指标
    metrics = {
        'min_distances': min_distances,
        'control_efforts': control_efforts,
        'cbf_values': cbf_values_history,
        'path_lengths': path_lengths
    }
    
    # 如果可用，添加瓶颈指标
    if bottleneck_metrics:
        metrics.update({
            'bottleneck_throughput': bottleneck_metrics.throughput,
            'bottleneck_velocity_fluctuation': bottleneck_metrics.velocity_fluctuation,
            'bottleneck_total_waiting_time': bottleneck_metrics.total_waiting_time,
            'bottleneck_coordination_efficiency': bottleneck_metrics.coordination_efficiency,
            'bottleneck_collision_rate': bottleneck_metrics.collision_rate,
            'bottleneck_completion_rate': bottleneck_metrics.completion_rate,
            'bottleneck_avg_bottleneck_time': bottleneck_metrics.avg_bottleneck_time
        })
    
    return anim, metrics


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='可视化训练的BPTT策略')
    parser.add_argument('--model_dir', type=str, required=True, help='包含训练模型的目录')
    parser.add_argument('--config', type=str, help='配置文件路径（如果与模型目录不同）')
    parser.add_argument('--device', type=str, default='auto', help='设备（cuda/cpu/auto）')
    parser.add_argument('--output', type=str, help='输出GIF文件路径')
    parser.add_argument('--step', type=int, help='要加载的特定模型步骤')
    parser.add_argument('--save_metrics', type=str, help='保存指标的JSON文件路径')
    
    args = parser.parse_args()
    
    # 设置设备 - 强制使用GPU（如果可用）
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
        # 传统支持cuda标志
    
    # 加载配置
    config_path = args.config if args.config else os.path.join(args.model_dir, 'config.yaml')
    if not os.path.exists(config_path):
        config_path = os.path.join(args.model_dir, '..', 'config.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确定要使用的模型步骤
    if args.step:
        model_step = args.step
    else:
        # 首先检查是否有final目录
        final_dir = os.path.join(args.model_dir, 'final')
        if os.path.exists(final_dir):
            model_step = 'final'
        else:
            # 查找最新步骤
            models_dir = os.path.join(args.model_dir, 'models')
            if os.path.exists(models_dir):
                steps = [int(d) for d in os.listdir(models_dir) if d.isdigit()]
                if steps:
                    model_step = max(steps)
                else:
                    raise ValueError("在models目录中找不到训练步骤")
            else:
                raise ValueError("找不到models目录")
    
    print(f"从步骤加载模型: {model_step}")
    
    # 基于配置和类型创建环境
    env_config = config.get('env', {})
    
    # 创建环境
    if 'env_type' in config:
        env_type = config['env_type']
    else:
        env_type = 'double_integrator'  # 默认
    
    if env_type == 'double_integrator':
        env = DoubleIntegratorEnv(env_config)
    else:
        raise ValueError(f"不支持的环境类型: {env_type}")
    
    # 将环境移动到设备
    env = env.to(device)
    
    # 创建策略网络
    # 使用YAML文件中的策略配置
    network_config = config.get('networks', {})
    policy_config = network_config.get('policy', {})
    
    if policy_config:
        # 使用YAML文件中的配置（支持视觉和其他高级特性）
        obs_shape = env.observation_shape
        action_shape = env.action_shape
        
        # 确保policy_head具有正确的输出维度
        if 'policy_head' not in policy_config:
            policy_config['policy_head'] = {}
        policy_config['policy_head']['output_dim'] = action_shape[-1]
    else:
        # 后备方案：创建默认配置
        obs_shape = env.observation_shape
        action_shape = env.action_shape
        
        if len(obs_shape) > 2:  # 视觉输入
            policy_config = {
                'perception': {
                    'use_vision': True,
                    'input_dim': obs_shape[-3:],
                    'output_dim': 256,
                    'vision': {
                        'input_channels': obs_shape[-3],
                        'channels': [32, 64, 128],
                        'height': obs_shape[-2],
                        'width': obs_shape[-1]
                    }
                },
                'memory': {
                    'hidden_dim': 128,
                    'num_layers': 1
                },
                'policy_head': {
                    'output_dim': action_shape[-1],
                    'hidden_dims': [256],
                    'activation': 'relu',
                    'predict_alpha': True
                }
            }
        else:  # 状态输入
            policy_config = {
                'perception': {
                    'use_vision': False,
                    'input_dim': obs_shape[-1],
                    'output_dim': 128,
                    'hidden_dims': [256, 256],
                    'activation': 'relu'
                },
                'memory': {
                    'hidden_dim': 128,
                    'num_layers': 1
                },
                'policy_head': {
                    'output_dim': action_shape[-1],
                    'hidden_dims': [256, 256],
                    'activation': 'relu',
                    'predict_alpha': True
                }
            }
    
    # 创建策略网络
    policy_network = create_policy_from_config(policy_config)
    policy_network = policy_network.to(device)
    
    # 如果可用，创建CBF网络
    cbf_network = None
    cbf_config = network_config.get('cbf')
    if cbf_config:
        # 目前，我们将使用简单的MLP作为CBF网络
        obs_dim = obs_shape[-1] if len(obs_shape) <= 2 else np.prod(obs_shape[-3:])
        cbf_network = nn.Sequential(
            nn.Linear(obs_dim * env_config.get('num_agents', 8), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)
    
    # 加载训练的模型
    if model_step == 'final':
        model_path = os.path.join(args.model_dir, 'final')
    else:
        model_path = os.path.join(args.model_dir, 'models', str(model_step))
    
    # 加载策略网络
    policy_path = os.path.join(model_path, 'policy.pt')
    if os.path.exists(policy_path):
        policy_network.load_state_dict(torch.load(policy_path, map_location=device))
        print(f"从{policy_path}加载策略网络")
    else:
        print(f"警告：在{policy_path}找不到策略网络")
    
    # 如果存在，加载CBF网络
    if cbf_network:
        cbf_path = os.path.join(model_path, 'cbf.pt')
        if os.path.exists(cbf_path):
            cbf_network.load_state_dict(torch.load(cbf_path, map_location=device))
            print(f"从{cbf_path}加载CBF网络")
    
    # 输出路径
    if args.output:
        output_path = args.output
        # 如果未指定扩展名，确保扩展名为.gif
        if not output_path.endswith('.gif'):
            output_path += '.gif'
    else:
        output_path = os.path.join(args.model_dir, f'visualization_step_{model_step}.gif')
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 获取可视化指标的CBF alpha值
    cbf_alpha = env_config.get('cbf_alpha', 'Dynamic')
    
    # 将CBF alpha添加到训练配置以进行可视化
    if 'cbf_alpha' not in config.get('env', {}):
        config.setdefault('env', {})['cbf_alpha'] = cbf_alpha
    
    # 运行可视化
    print(f"运行可视化并保存到: {output_path}")
    try:
        anim, metrics = visualize_trajectory(
            env=env,
            policy_network=policy_network,
            cbf_network=cbf_network,
            config=config,
            save_path=output_path,
            device=device
        )
        
        # 用实验详情增强指标
        metrics.update({
            'model_step': model_step,
            'config_path': config_path,
            'device': str(device),
            'cbf_alpha': cbf_alpha
        })
        
        # 如果指定，将指标保存到文件
        if args.save_metrics:
            import json
            metrics_serializable = {}
            for key, value in metrics.items():
                if isinstance(value, (list, np.ndarray)):
                    metrics_serializable[key] = np.array(value).tolist()
                else:
                    metrics_serializable[key] = value
            
            with open(args.save_metrics, 'w', encoding='utf-8') as f:
                json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)
            print(f"指标已保存到: {args.save_metrics}")
        else:
            # 将指标保存到模型目录中的标准位置
            metrics_path = os.path.join(args.model_dir, f'metrics_step_{model_step}.json')
            metrics_serializable = {}
            for key, value in metrics.items():
                if isinstance(value, (list, np.ndarray)):
                    metrics_serializable[key] = np.array(value).tolist()
                else:
                    metrics_serializable[key] = value
            
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)
        
        # 打印摘要信息
        print(f"\n可视化完成！")
        print(f"动画已保存到: {output_path}")
        print(f"指标已保存到: {metrics_path if not args.save_metrics else args.save_metrics}")
        
    except Exception as e:
        print(f"可视化过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

# End of visualization script 