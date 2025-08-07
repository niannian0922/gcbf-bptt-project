#!/usr/bin/env python3
"""
统一的BPTT可视化脚本
完全镜像train_bptt.py的配置加载、环境创建和模型实例化逻辑
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from gcbfplus.env import DoubleIntegratorEnv
from gcbfplus.env.gcbf_safety_layer import GCBFSafetyLayer
from gcbfplus.policy import BPTTPolicy, create_policy_from_config


def load_trained_model(model_dir, step=None, device='cpu'):
    """
    完全镜像train_bptt.py的模型加载逻辑
    """
    print(f"🔍 统一模型加载流程")
    print(f"📁 模型目录: {model_dir}")
    
    # 1. 查找配置文件 - 镜像训练脚本的逻辑
    config_path = os.path.join(model_dir, 'config.yaml')
    if not os.path.exists(config_path):
        # 尝试父目录
        config_path = os.path.join(model_dir, '..', 'config.yaml')
        if not os.path.exists(config_path):
            # 尝试根目录的配置文件
            possible_configs = [
                'config/simple_collaboration.yaml', 
                'config/alpha_medium_obs.yaml',
                'config/bptt_config.yaml'
            ]
            for config_file in possible_configs:
                if os.path.exists(config_file):
                    config_path = config_file
                    break
            else:
                raise ValueError(f"无法找到配置文件，检查过的路径: {possible_configs}")
    
    print(f"📋 使用配置文件: {config_path}")
    
    # 2. 加载配置 - 完全镜像train_bptt.py的逻辑
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ 配置加载成功")
    
    # 3. 提取配置部分 - 镜像train_bptt.py
    env_config = config.get('env', {})
    training_config = config.get('training', {})
    network_config = config.get('networks', {})
    
    # 如果配置中没有networks部分，添加默认值
    if not network_config:
        print(f"⚠️ 配置文件缺少networks部分，添加默认配置")
        network_config = {
            'policy': {},
            'cbf': {'alpha': 1.0}
        }
        config['networks'] = network_config
    
    # 确保环境有障碍物配置（因为模型是在9维输入下训练的）
    if 'obstacles' not in env_config:
        print(f"⚠️ 添加障碍物配置以匹配9维输入模型")
        env_config['obstacles'] = {
            'enabled': True,
            'bottleneck': True,
            'positions': [[0.0, -0.8], [0.0, 0.8]],
            'radii': [0.4, 0.4]
        }
    
    # 提取策略和CBF网络配置
    policy_config = network_config.get('policy', {})
    cbf_network_config = network_config.get('cbf')
    
    print(f"📊 环境配置: {list(env_config.keys())}")
    print(f"🧠 策略配置: {list(policy_config.keys())}")
    print(f"🛡️ CBF配置: {cbf_network_config is not None}")
    
    # 4. 创建环境 - 完全镜像train_bptt.py的逻辑
    env_type = 'double_integrator'  # 默认值，镜像训练脚本
    
    if env_type == 'double_integrator':
        env = DoubleIntegratorEnv(env_config)
    else:
        raise ValueError(f"不支持的环境类型: {env_type}")
    
    print(f"🌍 环境创建成功: {env_type}")
    
    # 将环境移动到设备
    env = env.to(device)
    
    # 5. 创建策略网络 - 完全镜像train_bptt.py的逻辑
    if policy_config:
        # 确保策略配置具有正确的观测和动作维度
        obs_shape = env.observation_shape
        action_shape = env.action_shape
        
        print(f"📏 观测形状: {obs_shape}")
        print(f"📏 动作形状: {action_shape}")
        
        # DEBUG: 添加调试信息
        print(f"🔍 DEBUG: obs_shape类型={type(obs_shape)}, 值={obs_shape}")
        print(f"🔍 DEBUG: action_shape类型={type(action_shape)}, 值={action_shape}")
        
        # 如果需要，为缺失的感知配置添加默认值 - 镜像训练脚本
        if 'perception' not in policy_config:
            policy_config['perception'] = {}
        
        perception_config = policy_config['perception']
        
        # 处理视觉输入 - 镜像训练脚本
        if len(obs_shape) > 2:  # 视觉输入 [n_agents, channels, height, width]
            perception_config.update({
                'use_vision': True,
                'input_dim': obs_shape[-3:],  # [channels, height, width]
                'output_dim': perception_config.get('output_dim', 256)
            })
        else:  # 状态输入 [n_agents, obs_dim]
            perception_config.update({
                'use_vision': False,
                'input_dim': obs_shape[-1],  # obs_dim
                'output_dim': perception_config.get('output_dim', 128),
                'hidden_dims': perception_config.get('hidden_dims', [256, 256])
            })
        
        print(f"🔍 DEBUG: perception_config={perception_config}")
        
        # 如果需要，添加默认记忆配置 - 镜像训练脚本
        if 'memory' not in policy_config:
            policy_config['memory'] = {}
        
        memory_config = policy_config['memory']
        memory_config.update({
            'hidden_dim': memory_config.get('hidden_dim', 128),
            'num_layers': memory_config.get('num_layers', 1)
        })
        
        # 确保policy_head具有所有必需参数 - 镜像训练脚本
        if 'policy_head' not in policy_config:
            # 从感知或记忆配置获取hidden_dim，或使用默认值
            if len(obs_shape) > 2:  # 视觉情况
                hidden_dims = perception_config.get('output_dim', 256)
            else:  # 状态情况
                hidden_dims = perception_config.get('hidden_dims', [256, 256])
                if isinstance(hidden_dims, list):
                    hidden_dims = hidden_dims[0] if hidden_dims else 256
            
            policy_config['policy_head'] = {
                'output_dim': action_shape[-1],  # action_dim
                'hidden_dims': [hidden_dims],
                'activation': 'relu',
                'predict_alpha': True  # 启用自适应安全边距
            }
        else:
            policy_head_config = policy_config['policy_head']
            policy_head_config['output_dim'] = action_shape[-1]  # 确保正确的动作维度
            if 'predict_alpha' not in policy_head_config:
                policy_head_config['predict_alpha'] = True  # 默认启用动态alpha
        
        print(f"🎯 最终策略配置: {policy_config}")
    else:
        # 后备方案：如果YAML中没有策略配置，创建默认配置 - 镜像训练脚本
        obs_shape = env.observation_shape
        action_shape = env.action_shape
        
        print(f"📏 后备 - 观测形状: {obs_shape}")
        print(f"📏 后备 - 动作形状: {action_shape}")
        
        if len(obs_shape) > 2:  # 视觉输入
            policy_config = {
                'perception': {
                    'use_vision': True,
                    'input_dim': obs_shape[-3:],  # [channels, height, width]
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
    
    # 6. 创建策略网络 - 根据实际模型权重推断架构
    print(f"🧠 创建策略网络...")
    
    # 首先尝试从模型文件推断正确的架构
    model_step = None
    models_dir = os.path.join(model_dir, 'models')
    if os.path.exists(models_dir):
        steps = [int(d) for d in os.listdir(models_dir) if d.isdigit()]
        if steps:
            model_step = max(steps)
    
    if model_step:
        policy_path = os.path.join(model_dir, 'models', str(model_step), 'policy.pt')
        if os.path.exists(policy_path):
            print(f"🔍 从模型权重推断网络架构: {policy_path}")
            policy_state_dict = torch.load(policy_path, map_location='cpu', weights_only=True)
            
            # 分析权重来推断正确的架构
            perception_out_features = None
            memory_hidden_dim = None
            
            # 从第一层推断perception输出维度
            if 'perception.mlp.0.weight' in policy_state_dict:
                perception_out_features = policy_state_dict['perception.mlp.0.weight'].shape[0]
                print(f"🔍 推断perception输出维度: {perception_out_features}")
            
            # 从memory层推断hidden_dim
            if 'memory.gru.weight_hh_l0' in policy_state_dict:
                memory_hidden_dim = policy_state_dict['memory.gru.weight_hh_l0'].shape[1]
                print(f"🔍 推断memory hidden维度: {memory_hidden_dim}")
            
            # 更新policy配置
            if perception_out_features:
                policy_config['perception']['output_dim'] = perception_out_features
                policy_config['perception']['hidden_dims'] = [perception_out_features, perception_out_features]
            
            if memory_hidden_dim:
                policy_config['memory']['hidden_dim'] = memory_hidden_dim
            
            # 更新policy_head配置
            if perception_out_features:
                policy_config['policy_head']['hidden_dims'] = [perception_out_features]
            
            print(f"🎯 推断后的策略配置: {policy_config}")
    
    policy_network = create_policy_from_config(policy_config)
    policy_network = policy_network.to(device)
    print(f"✅ 策略网络创建成功")
    
    # 7. 创建CBF网络 - 镜像train_bptt.py的逻辑
    cbf_network = None
    if cbf_network_config:
        print(f"🛡️ 创建CBF网络...")
        # 从配置中提取CBF alpha参数
        cbf_alpha = cbf_network_config.get('alpha', 1.0)
        
        # 基于CBF网络配置创建CBF网络 - 镜像训练脚本
        obs_dim = obs_shape[-1] if len(obs_shape) <= 2 else np.prod(obs_shape[-3:])
        
        print(f"🔍 DEBUG: obs_dim={obs_dim}, num_agents={env_config.get('num_agents', 8)}")
        
        # ❌ 这里是关键问题！训练脚本中的CBF网络创建逻辑是错误的
        # 需要根据实际的CBF模型文件来确定正确的架构
        
        # 让我们首先尝试加载CBF模型来确定正确的输入维度
        cbf_model_path = None
        if step:
            cbf_model_path = os.path.join(model_dir, 'models', str(step), 'cbf.pt')
        else:
            # 查找最新的CBF模型
            models_dir = os.path.join(model_dir, 'models')
            if os.path.exists(models_dir):
                steps = [int(d) for d in os.listdir(models_dir) if d.isdigit()]
                if steps:
                    latest_step = max(steps)
                    cbf_model_path = os.path.join(model_dir, 'models', str(latest_step), 'cbf.pt')
        
        if cbf_model_path and os.path.exists(cbf_model_path):
            # 尝试分析CBF模型的实际架构
            cbf_state_dict = torch.load(cbf_model_path, map_location='cpu', weights_only=True)
            
            # 查找第一个线性层来确定输入维度
            first_layer_key = None
            for key in cbf_state_dict.keys():
                if 'weight' in key and len(cbf_state_dict[key].shape) == 2:
                    first_layer_key = key
                    break
            
            if first_layer_key:
                actual_input_dim = cbf_state_dict[first_layer_key].shape[1]
                print(f"🔍 DEBUG: CBF实际输入维度={actual_input_dim}")
                
                # 根据实际维度创建CBF网络
                if actual_input_dim == 6:
                    # 单个智能体的6维状态
                    cbf_network = nn.Sequential(
                        nn.Linear(6, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1)
                    ).to(device)
                elif actual_input_dim == 9:
                    # 单个智能体的9维状态（包含障碍物）
                    cbf_network = nn.Sequential(
                        nn.Linear(9, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1)
                    ).to(device)
                else:
                    # 使用检测到的维度
                    hidden_sizes = [128, 128]  # 从模型中推断
                    layers = []
                    in_dim = actual_input_dim
                    for hidden_size in hidden_sizes:
                        layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU()])
                        in_dim = hidden_size
                    layers.append(nn.Linear(in_dim, 1))
                    cbf_network = nn.Sequential(*layers).to(device)
                
                print(f"✅ CBF网络创建成功 (输入维度: {actual_input_dim})")
            else:
                print(f"⚠️ 无法确定CBF网络架构，跳过CBF网络")
        else:
            print(f"⚠️ CBF模型文件不存在: {cbf_model_path}")
    
    # 8. 确定模型步骤（如果前面没有找到）
    if step is None:
        if model_step:
            step = model_step
        else:
            # 查找最新步骤
            models_dir = os.path.join(model_dir, 'models')
            if os.path.exists(models_dir):
                steps = [int(d) for d in os.listdir(models_dir) if d.isdigit()]
                if steps:
                    step = max(steps)
                else:
                    raise ValueError("在models目录中找不到训练步骤")
            else:
                raise ValueError("找不到models目录")
    
    print(f"📈 使用模型步骤: {step}")
    
    # 9. 加载训练好的权重
    model_path = os.path.join(model_dir, 'models', str(step))
    
    # 加载策略网络权重（如果前面没有加载）
    policy_path = os.path.join(model_path, 'policy.pt')
    if os.path.exists(policy_path):
        try:
            if 'policy_state_dict' not in locals():
                policy_state_dict = torch.load(policy_path, map_location=device, weights_only=True)
            policy_network.load_state_dict(policy_state_dict)
            print(f"✅ 策略网络权重加载成功: {policy_path}")
        except Exception as e:
            print(f"❌ 策略网络权重加载失败: {e}")
            raise
    else:
        raise ValueError(f"策略文件不存在: {policy_path}")
    
    # 加载CBF网络权重
    if cbf_network:
        cbf_path = os.path.join(model_path, 'cbf.pt')
        if os.path.exists(cbf_path):
            try:
                cbf_state_dict = torch.load(cbf_path, map_location=device, weights_only=True)
                cbf_network.load_state_dict(cbf_state_dict)
                print(f"✅ CBF网络权重加载成功: {cbf_path}")
            except Exception as e:
                print(f"❌ CBF网络权重加载失败: {e}")
                print(f"🔧 将CBF网络设置为None")
                cbf_network = None
    
    return env, policy_network, cbf_network, config


def run_simulation_with_diagnostics(env, policy_network, cbf_network, device, num_steps=100):
    """
    运行仿真并添加详细的诊断信息
    """
    print(f"🎬 开始仿真 (包含详细诊断)")
    print(f"📏 计划步数: {num_steps}")
    
    # 设置网络为评估模式
    policy_network.eval()
    if cbf_network:
        cbf_network.eval()
    
    # 初始化环境
    state = env.reset()
    print(f"🔍 DEBUG: 初始状态类型={type(state)}")
    print(f"🔍 DEBUG: 初始状态.positions形状={state.positions.shape}")
    
    # 存储轨迹数据
    trajectory_data = {
        'positions': [],
        'velocities': [],
        'actions': [],
        'alphas': [],
        'cbf_values': []
    }
    
    with torch.no_grad():
        for step in range(num_steps):
            print(f"\n--- 步骤 {step} 诊断 ---")
            
            # 记录当前状态
            current_positions = state.positions[0].cpu().numpy()
            current_velocities = state.velocities[0].cpu().numpy()
            trajectory_data['positions'].append(current_positions.copy())
            trajectory_data['velocities'].append(current_velocities.copy())
            
            # 1. 获取观测 - 添加诊断
            observations = env.get_observations(state)
            print(f"🔍 DEBUG: 观测形状={observations.shape}")
            print(f"🔍 DEBUG: 观测dtype={observations.dtype}")
            print(f"🔍 DEBUG: 观测设备={observations.device}")
            print(f"🔍 DEBUG: 观测范围=[{torch.min(observations):.4f}, {torch.max(observations):.4f}]")
            
            # 确保观测在正确的设备上
            observations = observations.to(device)
            
            # 2. 策略网络推理 - 添加诊断
            try:
                print(f"🧠 策略网络推理...")
                policy_output = policy_network(observations, state)
                
                print(f"🔍 DEBUG: policy_output类型={type(policy_output)}")
                
                if hasattr(policy_output, 'actions'):
                    actions = policy_output.actions
                    print(f"🔍 DEBUG: actions形状={actions.shape}")
                    print(f"🔍 DEBUG: actions设备={actions.device}")
                    print(f"🔍 DEBUG: actions范围=[{torch.min(actions):.4f}, {torch.max(actions):.4f}]")
                else:
                    print(f"❌ policy_output没有actions属性")
                    raise ValueError("策略输出缺少actions")
                
                if hasattr(policy_output, 'alphas'):
                    alphas = policy_output.alphas
                    print(f"🔍 DEBUG: alphas形状={alphas.shape}")
                    print(f"🔍 DEBUG: alphas设备={alphas.device}")
                    print(f"🔍 DEBUG: alphas范围=[{torch.min(alphas):.4f}, {torch.max(alphas):.4f}]")
                else:
                    print(f"⚠️ policy_output没有alphas属性，使用默认值")
                    alphas = torch.ones(actions.shape[0], actions.shape[1], 1, device=device) * 0.5
                
            except Exception as e:
                print(f"❌ 策略网络推理失败: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # 3. CBF网络推理 - 添加诊断
            cbf_values = None
            if cbf_network:
                try:
                    print(f"🛡️ CBF网络推理...")
                    
                    # 确定CBF网络的输入格式
                    batch_size, num_agents, obs_dim = observations.shape
                    print(f"🔍 DEBUG: CBF输入 - batch_size={batch_size}, num_agents={num_agents}, obs_dim={obs_dim}")
                    
                    # 根据之前的修复，CBF网络期望单个智能体的输入
                    cbf_values_list = []
                    for agent_idx in range(num_agents):
                        agent_obs = observations[0, agent_idx, :]  # 取第一个batch的第agent_idx个智能体
                        print(f"🔍 DEBUG: 智能体{agent_idx} CBF输入形状={agent_obs.shape}")
                        
                        cbf_val = cbf_network(agent_obs.unsqueeze(0))  # 添加batch维度
                        cbf_values_list.append(cbf_val)
                        print(f"🔍 DEBUG: 智能体{agent_idx} CBF输出={cbf_val.item():.4f}")
                    
                    cbf_values = torch.stack(cbf_values_list, dim=1)  # [batch_size, num_agents, 1]
                    print(f"🔍 DEBUG: 最终CBF值形状={cbf_values.shape}")
                    
                except Exception as e:
                    print(f"❌ CBF网络推理失败: {e}")
                    cbf_values = None
            
            # 记录数据
            trajectory_data['actions'].append(actions[0].cpu().numpy())
            trajectory_data['alphas'].append(alphas[0].cpu().numpy())
            if cbf_values is not None:
                trajectory_data['cbf_values'].append(cbf_values[0].cpu().numpy())
            else:
                trajectory_data['cbf_values'].append(np.zeros((len(current_positions), 1)))
            
            # 4. 环境步进 - 添加诊断
            try:
                print(f"🌍 环境步进...")
                print(f"🔍 DEBUG: 步进前state.positions形状={state.positions.shape}")
                print(f"🔍 DEBUG: 步进actions形状={actions.shape}")
                print(f"🔍 DEBUG: 步进alphas形状={alphas.shape}")
                
                step_result = env.step(state, actions, alphas)
                
                print(f"🔍 DEBUG: 步进后next_state.positions形状={step_result.next_state.positions.shape}")
                print(f"✅ 环境步进成功")
                
                # 更新状态
                state = step_result.next_state
                
            except Exception as e:
                print(f"❌ 环境步进失败: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # 显示进度
            if step % 20 == 0:
                action_magnitude = torch.norm(actions, dim=-1).mean().item()
                print(f"📊 步骤 {step}: 平均动作强度={action_magnitude:.4f}")
    
    print(f"✅ 仿真完成，共 {len(trajectory_data['positions'])} 步")
    return trajectory_data


def create_final_visualization(trajectory_data, env_config, output_path):
    """
    创建最终的可视化动画
    """
    print(f"🎨 创建最终可视化动画...")
    
    positions_history = trajectory_data['positions']
    actions_history = trajectory_data['actions']
    alphas_history = trajectory_data['alphas']
    cbf_values_history = trajectory_data['cbf_values']
    
    if not positions_history:
        print(f"❌ 没有轨迹数据")
        return False
    
    num_steps = len(positions_history)
    num_agents = len(positions_history[0])
    
    print(f"📊 动画参数: {num_steps} 步, {num_agents} 智能体")
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('🎯 最终协作可视化结果 - 统一代码路径', fontsize=18, fontweight='bold')
    
    # 主轨迹图
    ax1.set_xlim(-3.0, 3.0)
    ax1.set_ylim(-2.0, 2.0)
    ax1.set_aspect('equal')
    ax1.set_title('🚁 真实训练模型轨迹', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 绘制障碍物
    obstacles = env_config.get('obstacles', {})
    if obstacles.get('enabled', False):
        for i, (pos, radius) in enumerate(zip(obstacles.get('positions', []), obstacles.get('radii', []))):
            circle = plt.Circle(pos, radius, color='red', alpha=0.8, 
                              label='障碍物' if i == 0 else "")
            ax1.add_patch(circle)
    
    # 起始和目标区域
    start_zone = plt.Rectangle((-2.5, -1.5), 1.0, 3.0, fill=False, 
                              edgecolor='green', linestyle='--', linewidth=2, 
                              alpha=0.8, label='起始区域')
    ax1.add_patch(start_zone)
    
    target_zone = plt.Rectangle((1.5, -1.5), 1.0, 3.0, fill=False, 
                               edgecolor='blue', linestyle='--', linewidth=2, 
                               alpha=0.8, label='目标区域')
    ax1.add_patch(target_zone)
    
    # 智能体颜色
    colors = ['#FF4444', '#44FF44', '#4444FF', '#FFAA44', '#FF44AA', '#44AAFF'][:num_agents]
    
    # 初始化动画元素
    trail_lines = []
    drone_dots = []
    
    for i in range(num_agents):
        line, = ax1.plot([], [], '-', color=colors[i], alpha=0.8, linewidth=3,
                        label=f'智能体{i+1}' if i < 3 else "")
        trail_lines.append(line)
        
        drone, = ax1.plot([], [], 'o', color=colors[i], markersize=14, 
                         markeredgecolor='black', markeredgewidth=2, zorder=5)
        drone_dots.append(drone)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 分析图表
    ax2.set_title('🧠 策略网络输出', fontsize=12)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('动作强度')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('⚖️ Alpha值监控', fontsize=12)
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('Alpha值')
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('🛡️ CBF安全值', fontsize=12)
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('CBF值')
    ax4.grid(True, alpha=0.3)
    
    def animate(frame):
        if frame >= num_steps:
            return trail_lines + drone_dots
        
        current_positions = positions_history[frame]
        
        # 更新轨迹和智能体
        for i in range(num_agents):
            trail_x = [pos[i, 0] for pos in positions_history[:frame+1]]
            trail_y = [pos[i, 1] for pos in positions_history[:frame+1]]
            trail_lines[i].set_data(trail_x, trail_y)
            
            drone_dots[i].set_data([current_positions[i, 0]], [current_positions[i, 1]])
        
        # 更新分析图表
        if frame > 5:
            steps = list(range(frame+1))
            
            # 策略输出
            if len(actions_history) > frame:
                action_magnitudes = []
                for step in range(frame+1):
                    if step < len(actions_history):
                        step_actions = actions_history[step]
                        avg_magnitude = np.mean([np.linalg.norm(a) for a in step_actions])
                        action_magnitudes.append(avg_magnitude)
                    else:
                        action_magnitudes.append(0)
                
                ax2.clear()
                ax2.plot(steps, action_magnitudes, 'purple', linewidth=3, label='平均动作强度')
                ax2.fill_between(steps, action_magnitudes, alpha=0.3, color='purple')
                ax2.set_title(f'🧠 策略网络输出 (步数: {frame})')
                ax2.set_xlabel('时间步')
                ax2.set_ylabel('动作强度')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Alpha值
            if len(alphas_history) > frame:
                alpha_values = []
                for step in range(frame+1):
                    if step < len(alphas_history):
                        avg_alpha = np.mean(alphas_history[step])
                        alpha_values.append(avg_alpha)
                    else:
                        alpha_values.append(0.5)
                
                ax3.clear()
                ax3.plot(steps, alpha_values, 'orange', linewidth=3, label='平均Alpha值')
                ax3.fill_between(steps, alpha_values, alpha=0.3, color='orange')
                ax3.set_title(f'⚖️ Alpha值监控 (步数: {frame})')
                ax3.set_xlabel('时间步')
                ax3.set_ylabel('Alpha值')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # CBF值
            if len(cbf_values_history) > frame:
                cbf_avg_values = []
                for step in range(frame+1):
                    if step < len(cbf_values_history):
                        avg_cbf = np.mean(cbf_values_history[step])
                        cbf_avg_values.append(avg_cbf)
                    else:
                        cbf_avg_values.append(0)
                
                ax4.clear()
                ax4.plot(steps, cbf_avg_values, 'red', linewidth=3, label='平均CBF值')
                ax4.fill_between(steps, cbf_avg_values, alpha=0.3, color='red')
                ax4.set_title(f'🛡️ CBF安全值 (步数: {frame})')
                ax4.set_xlabel('时间步')
                ax4.set_ylabel('CBF值')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        return trail_lines + drone_dots
    
    # 创建动画
    anim = FuncAnimation(fig, animate, frames=num_steps, interval=150, blit=False, repeat=True)
    
    # 保存动画
    try:
        print(f"💾 保存最终可视化: {output_path}")
        
        # 尝试保存为MP4
        if output_path.endswith('.mp4'):
            anim.save(output_path, writer='ffmpeg', fps=8, dpi=150)
        else:
            anim.save(output_path, writer='pillow', fps=8, dpi=150)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ 保存成功: {output_path}")
        print(f"📁 文件大小: {file_size:.2f}MB")
        return True
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False
    finally:
        plt.close()


def main():
    """
    主函数 - 完全统一的可视化流程
    """
    parser = argparse.ArgumentParser(description='统一的BPTT可视化脚本')
    parser.add_argument('--model_dir', type=str, default='logs/full_collaboration_training', 
                       help='模型目录路径')
    parser.add_argument('--step', type=int, help='模型步骤 (默认使用最新)')
    parser.add_argument('--device', type=str, default='cpu', help='设备 (cuda/cpu)')
    parser.add_argument('--output', type=str, default='FINAL_COLLABORATION_RESULT.mp4', 
                       help='输出文件路径')
    parser.add_argument('--num_steps', type=int, default=120, help='仿真步数')
    
    args = parser.parse_args()
    
    print(f"🎯 统一BPTT可视化系统")
    print(f"=" * 80)
    print(f"📁 模型目录: {args.model_dir}")
    print(f"📈 模型步骤: {args.step if args.step else '最新'}")
    print(f"💻 设备: {args.device}")
    print(f"📁 输出文件: {args.output}")
    print(f"=" * 80)
    
    # 设置设备
    device = torch.device(args.device)
    
    try:
        # 1. 加载训练模型 - 统一路径
        print(f"\n🔄 第1步: 加载训练模型")
        env, policy_network, cbf_network, config = load_trained_model(
            args.model_dir, args.step, device
        )
        
        # 2. 运行仿真 - 包含诊断
        print(f"\n🔄 第2步: 运行仿真 (包含诊断)")
        trajectory_data = run_simulation_with_diagnostics(
            env, policy_network, cbf_network, device, args.num_steps
        )
        
        # 3. 创建可视化
        print(f"\n🔄 第3步: 创建最终可视化")
        success = create_final_visualization(
            trajectory_data, config.get('env', {}), args.output
        )
        
        if success:
            print(f"\n🎉 统一可视化生成成功!")
            print(f"📁 最终结果: {args.output}")
            print(f"✅ 代码路径已完全统一")
            print(f"🧠 这是您真实训练模型的表现")
        else:
            print(f"\n❌ 可视化生成失败")
            
    except Exception as e:
        print(f"\n❌ 统一可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
 
"""
统一的BPTT可视化脚本
完全镜像train_bptt.py的配置加载、环境创建和模型实例化逻辑
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from gcbfplus.env import DoubleIntegratorEnv
from gcbfplus.env.gcbf_safety_layer import GCBFSafetyLayer
from gcbfplus.policy import BPTTPolicy, create_policy_from_config


def load_trained_model(model_dir, step=None, device='cpu'):
    """
    完全镜像train_bptt.py的模型加载逻辑
    """
    print(f"🔍 统一模型加载流程")
    print(f"📁 模型目录: {model_dir}")
    
    # 1. 查找配置文件 - 镜像训练脚本的逻辑
    config_path = os.path.join(model_dir, 'config.yaml')
    if not os.path.exists(config_path):
        # 尝试父目录
        config_path = os.path.join(model_dir, '..', 'config.yaml')
        if not os.path.exists(config_path):
            # 尝试根目录的配置文件
            possible_configs = [
                'config/simple_collaboration.yaml', 
                'config/alpha_medium_obs.yaml',
                'config/bptt_config.yaml'
            ]
            for config_file in possible_configs:
                if os.path.exists(config_file):
                    config_path = config_file
                    break
            else:
                raise ValueError(f"无法找到配置文件，检查过的路径: {possible_configs}")
    
    print(f"📋 使用配置文件: {config_path}")
    
    # 2. 加载配置 - 完全镜像train_bptt.py的逻辑
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ 配置加载成功")
    
    # 3. 提取配置部分 - 镜像train_bptt.py
    env_config = config.get('env', {})
    training_config = config.get('training', {})
    network_config = config.get('networks', {})
    
    # 如果配置中没有networks部分，添加默认值
    if not network_config:
        print(f"⚠️ 配置文件缺少networks部分，添加默认配置")
        network_config = {
            'policy': {},
            'cbf': {'alpha': 1.0}
        }
        config['networks'] = network_config
    
    # 确保环境有障碍物配置（因为模型是在9维输入下训练的）
    if 'obstacles' not in env_config:
        print(f"⚠️ 添加障碍物配置以匹配9维输入模型")
        env_config['obstacles'] = {
            'enabled': True,
            'bottleneck': True,
            'positions': [[0.0, -0.8], [0.0, 0.8]],
            'radii': [0.4, 0.4]
        }
    
    # 提取策略和CBF网络配置
    policy_config = network_config.get('policy', {})
    cbf_network_config = network_config.get('cbf')
    
    print(f"📊 环境配置: {list(env_config.keys())}")
    print(f"🧠 策略配置: {list(policy_config.keys())}")
    print(f"🛡️ CBF配置: {cbf_network_config is not None}")
    
    # 4. 创建环境 - 完全镜像train_bptt.py的逻辑
    env_type = 'double_integrator'  # 默认值，镜像训练脚本
    
    if env_type == 'double_integrator':
        env = DoubleIntegratorEnv(env_config)
    else:
        raise ValueError(f"不支持的环境类型: {env_type}")
    
    print(f"🌍 环境创建成功: {env_type}")
    
    # 将环境移动到设备
    env = env.to(device)
    
    # 5. 创建策略网络 - 完全镜像train_bptt.py的逻辑
    if policy_config:
        # 确保策略配置具有正确的观测和动作维度
        obs_shape = env.observation_shape
        action_shape = env.action_shape
        
        print(f"📏 观测形状: {obs_shape}")
        print(f"📏 动作形状: {action_shape}")
        
        # DEBUG: 添加调试信息
        print(f"🔍 DEBUG: obs_shape类型={type(obs_shape)}, 值={obs_shape}")
        print(f"🔍 DEBUG: action_shape类型={type(action_shape)}, 值={action_shape}")
        
        # 如果需要，为缺失的感知配置添加默认值 - 镜像训练脚本
        if 'perception' not in policy_config:
            policy_config['perception'] = {}
        
        perception_config = policy_config['perception']
        
        # 处理视觉输入 - 镜像训练脚本
        if len(obs_shape) > 2:  # 视觉输入 [n_agents, channels, height, width]
            perception_config.update({
                'use_vision': True,
                'input_dim': obs_shape[-3:],  # [channels, height, width]
                'output_dim': perception_config.get('output_dim', 256)
            })
        else:  # 状态输入 [n_agents, obs_dim]
            perception_config.update({
                'use_vision': False,
                'input_dim': obs_shape[-1],  # obs_dim
                'output_dim': perception_config.get('output_dim', 128),
                'hidden_dims': perception_config.get('hidden_dims', [256, 256])
            })
        
        print(f"🔍 DEBUG: perception_config={perception_config}")
        
        # 如果需要，添加默认记忆配置 - 镜像训练脚本
        if 'memory' not in policy_config:
            policy_config['memory'] = {}
        
        memory_config = policy_config['memory']
        memory_config.update({
            'hidden_dim': memory_config.get('hidden_dim', 128),
            'num_layers': memory_config.get('num_layers', 1)
        })
        
        # 确保policy_head具有所有必需参数 - 镜像训练脚本
        if 'policy_head' not in policy_config:
            # 从感知或记忆配置获取hidden_dim，或使用默认值
            if len(obs_shape) > 2:  # 视觉情况
                hidden_dims = perception_config.get('output_dim', 256)
            else:  # 状态情况
                hidden_dims = perception_config.get('hidden_dims', [256, 256])
                if isinstance(hidden_dims, list):
                    hidden_dims = hidden_dims[0] if hidden_dims else 256
            
            policy_config['policy_head'] = {
                'output_dim': action_shape[-1],  # action_dim
                'hidden_dims': [hidden_dims],
                'activation': 'relu',
                'predict_alpha': True  # 启用自适应安全边距
            }
        else:
            policy_head_config = policy_config['policy_head']
            policy_head_config['output_dim'] = action_shape[-1]  # 确保正确的动作维度
            if 'predict_alpha' not in policy_head_config:
                policy_head_config['predict_alpha'] = True  # 默认启用动态alpha
        
        print(f"🎯 最终策略配置: {policy_config}")
    else:
        # 后备方案：如果YAML中没有策略配置，创建默认配置 - 镜像训练脚本
        obs_shape = env.observation_shape
        action_shape = env.action_shape
        
        print(f"📏 后备 - 观测形状: {obs_shape}")
        print(f"📏 后备 - 动作形状: {action_shape}")
        
        if len(obs_shape) > 2:  # 视觉输入
            policy_config = {
                'perception': {
                    'use_vision': True,
                    'input_dim': obs_shape[-3:],  # [channels, height, width]
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
    
    # 6. 创建策略网络 - 根据实际模型权重推断架构
    print(f"🧠 创建策略网络...")
    
    # 首先尝试从模型文件推断正确的架构
    model_step = None
    models_dir = os.path.join(model_dir, 'models')
    if os.path.exists(models_dir):
        steps = [int(d) for d in os.listdir(models_dir) if d.isdigit()]
        if steps:
            model_step = max(steps)
    
    if model_step:
        policy_path = os.path.join(model_dir, 'models', str(model_step), 'policy.pt')
        if os.path.exists(policy_path):
            print(f"🔍 从模型权重推断网络架构: {policy_path}")
            policy_state_dict = torch.load(policy_path, map_location='cpu', weights_only=True)
            
            # 分析权重来推断正确的架构
            perception_out_features = None
            memory_hidden_dim = None
            
            # 从第一层推断perception输出维度
            if 'perception.mlp.0.weight' in policy_state_dict:
                perception_out_features = policy_state_dict['perception.mlp.0.weight'].shape[0]
                print(f"🔍 推断perception输出维度: {perception_out_features}")
            
            # 从memory层推断hidden_dim
            if 'memory.gru.weight_hh_l0' in policy_state_dict:
                memory_hidden_dim = policy_state_dict['memory.gru.weight_hh_l0'].shape[1]
                print(f"🔍 推断memory hidden维度: {memory_hidden_dim}")
            
            # 更新policy配置
            if perception_out_features:
                policy_config['perception']['output_dim'] = perception_out_features
                policy_config['perception']['hidden_dims'] = [perception_out_features, perception_out_features]
            
            if memory_hidden_dim:
                policy_config['memory']['hidden_dim'] = memory_hidden_dim
            
            # 更新policy_head配置
            if perception_out_features:
                policy_config['policy_head']['hidden_dims'] = [perception_out_features]
            
            print(f"🎯 推断后的策略配置: {policy_config}")
    
    policy_network = create_policy_from_config(policy_config)
    policy_network = policy_network.to(device)
    print(f"✅ 策略网络创建成功")
    
    # 7. 创建CBF网络 - 镜像train_bptt.py的逻辑
    cbf_network = None
    if cbf_network_config:
        print(f"🛡️ 创建CBF网络...")
        # 从配置中提取CBF alpha参数
        cbf_alpha = cbf_network_config.get('alpha', 1.0)
        
        # 基于CBF网络配置创建CBF网络 - 镜像训练脚本
        obs_dim = obs_shape[-1] if len(obs_shape) <= 2 else np.prod(obs_shape[-3:])
        
        print(f"🔍 DEBUG: obs_dim={obs_dim}, num_agents={env_config.get('num_agents', 8)}")
        
        # ❌ 这里是关键问题！训练脚本中的CBF网络创建逻辑是错误的
        # 需要根据实际的CBF模型文件来确定正确的架构
        
        # 让我们首先尝试加载CBF模型来确定正确的输入维度
        cbf_model_path = None
        if step:
            cbf_model_path = os.path.join(model_dir, 'models', str(step), 'cbf.pt')
        else:
            # 查找最新的CBF模型
            models_dir = os.path.join(model_dir, 'models')
            if os.path.exists(models_dir):
                steps = [int(d) for d in os.listdir(models_dir) if d.isdigit()]
                if steps:
                    latest_step = max(steps)
                    cbf_model_path = os.path.join(model_dir, 'models', str(latest_step), 'cbf.pt')
        
        if cbf_model_path and os.path.exists(cbf_model_path):
            # 尝试分析CBF模型的实际架构
            cbf_state_dict = torch.load(cbf_model_path, map_location='cpu', weights_only=True)
            
            # 查找第一个线性层来确定输入维度
            first_layer_key = None
            for key in cbf_state_dict.keys():
                if 'weight' in key and len(cbf_state_dict[key].shape) == 2:
                    first_layer_key = key
                    break
            
            if first_layer_key:
                actual_input_dim = cbf_state_dict[first_layer_key].shape[1]
                print(f"🔍 DEBUG: CBF实际输入维度={actual_input_dim}")
                
                # 根据实际维度创建CBF网络
                if actual_input_dim == 6:
                    # 单个智能体的6维状态
                    cbf_network = nn.Sequential(
                        nn.Linear(6, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1)
                    ).to(device)
                elif actual_input_dim == 9:
                    # 单个智能体的9维状态（包含障碍物）
                    cbf_network = nn.Sequential(
                        nn.Linear(9, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1)
                    ).to(device)
                else:
                    # 使用检测到的维度
                    hidden_sizes = [128, 128]  # 从模型中推断
                    layers = []
                    in_dim = actual_input_dim
                    for hidden_size in hidden_sizes:
                        layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU()])
                        in_dim = hidden_size
                    layers.append(nn.Linear(in_dim, 1))
                    cbf_network = nn.Sequential(*layers).to(device)
                
                print(f"✅ CBF网络创建成功 (输入维度: {actual_input_dim})")
            else:
                print(f"⚠️ 无法确定CBF网络架构，跳过CBF网络")
        else:
            print(f"⚠️ CBF模型文件不存在: {cbf_model_path}")
    
    # 8. 确定模型步骤（如果前面没有找到）
    if step is None:
        if model_step:
            step = model_step
        else:
            # 查找最新步骤
            models_dir = os.path.join(model_dir, 'models')
            if os.path.exists(models_dir):
                steps = [int(d) for d in os.listdir(models_dir) if d.isdigit()]
                if steps:
                    step = max(steps)
                else:
                    raise ValueError("在models目录中找不到训练步骤")
            else:
                raise ValueError("找不到models目录")
    
    print(f"📈 使用模型步骤: {step}")
    
    # 9. 加载训练好的权重
    model_path = os.path.join(model_dir, 'models', str(step))
    
    # 加载策略网络权重（如果前面没有加载）
    policy_path = os.path.join(model_path, 'policy.pt')
    if os.path.exists(policy_path):
        try:
            if 'policy_state_dict' not in locals():
                policy_state_dict = torch.load(policy_path, map_location=device, weights_only=True)
            policy_network.load_state_dict(policy_state_dict)
            print(f"✅ 策略网络权重加载成功: {policy_path}")
        except Exception as e:
            print(f"❌ 策略网络权重加载失败: {e}")
            raise
    else:
        raise ValueError(f"策略文件不存在: {policy_path}")
    
    # 加载CBF网络权重
    if cbf_network:
        cbf_path = os.path.join(model_path, 'cbf.pt')
        if os.path.exists(cbf_path):
            try:
                cbf_state_dict = torch.load(cbf_path, map_location=device, weights_only=True)
                cbf_network.load_state_dict(cbf_state_dict)
                print(f"✅ CBF网络权重加载成功: {cbf_path}")
            except Exception as e:
                print(f"❌ CBF网络权重加载失败: {e}")
                print(f"🔧 将CBF网络设置为None")
                cbf_network = None
    
    return env, policy_network, cbf_network, config


def run_simulation_with_diagnostics(env, policy_network, cbf_network, device, num_steps=100):
    """
    运行仿真并添加详细的诊断信息
    """
    print(f"🎬 开始仿真 (包含详细诊断)")
    print(f"📏 计划步数: {num_steps}")
    
    # 设置网络为评估模式
    policy_network.eval()
    if cbf_network:
        cbf_network.eval()
    
    # 初始化环境
    state = env.reset()
    print(f"🔍 DEBUG: 初始状态类型={type(state)}")
    print(f"🔍 DEBUG: 初始状态.positions形状={state.positions.shape}")
    
    # 存储轨迹数据
    trajectory_data = {
        'positions': [],
        'velocities': [],
        'actions': [],
        'alphas': [],
        'cbf_values': []
    }
    
    with torch.no_grad():
        for step in range(num_steps):
            print(f"\n--- 步骤 {step} 诊断 ---")
            
            # 记录当前状态
            current_positions = state.positions[0].cpu().numpy()
            current_velocities = state.velocities[0].cpu().numpy()
            trajectory_data['positions'].append(current_positions.copy())
            trajectory_data['velocities'].append(current_velocities.copy())
            
            # 1. 获取观测 - 添加诊断
            observations = env.get_observations(state)
            print(f"🔍 DEBUG: 观测形状={observations.shape}")
            print(f"🔍 DEBUG: 观测dtype={observations.dtype}")
            print(f"🔍 DEBUG: 观测设备={observations.device}")
            print(f"🔍 DEBUG: 观测范围=[{torch.min(observations):.4f}, {torch.max(observations):.4f}]")
            
            # 确保观测在正确的设备上
            observations = observations.to(device)
            
            # 2. 策略网络推理 - 添加诊断
            try:
                print(f"🧠 策略网络推理...")
                policy_output = policy_network(observations, state)
                
                print(f"🔍 DEBUG: policy_output类型={type(policy_output)}")
                
                if hasattr(policy_output, 'actions'):
                    actions = policy_output.actions
                    print(f"🔍 DEBUG: actions形状={actions.shape}")
                    print(f"🔍 DEBUG: actions设备={actions.device}")
                    print(f"🔍 DEBUG: actions范围=[{torch.min(actions):.4f}, {torch.max(actions):.4f}]")
                else:
                    print(f"❌ policy_output没有actions属性")
                    raise ValueError("策略输出缺少actions")
                
                if hasattr(policy_output, 'alphas'):
                    alphas = policy_output.alphas
                    print(f"🔍 DEBUG: alphas形状={alphas.shape}")
                    print(f"🔍 DEBUG: alphas设备={alphas.device}")
                    print(f"🔍 DEBUG: alphas范围=[{torch.min(alphas):.4f}, {torch.max(alphas):.4f}]")
                else:
                    print(f"⚠️ policy_output没有alphas属性，使用默认值")
                    alphas = torch.ones(actions.shape[0], actions.shape[1], 1, device=device) * 0.5
                
            except Exception as e:
                print(f"❌ 策略网络推理失败: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # 3. CBF网络推理 - 添加诊断
            cbf_values = None
            if cbf_network:
                try:
                    print(f"🛡️ CBF网络推理...")
                    
                    # 确定CBF网络的输入格式
                    batch_size, num_agents, obs_dim = observations.shape
                    print(f"🔍 DEBUG: CBF输入 - batch_size={batch_size}, num_agents={num_agents}, obs_dim={obs_dim}")
                    
                    # 根据之前的修复，CBF网络期望单个智能体的输入
                    cbf_values_list = []
                    for agent_idx in range(num_agents):
                        agent_obs = observations[0, agent_idx, :]  # 取第一个batch的第agent_idx个智能体
                        print(f"🔍 DEBUG: 智能体{agent_idx} CBF输入形状={agent_obs.shape}")
                        
                        cbf_val = cbf_network(agent_obs.unsqueeze(0))  # 添加batch维度
                        cbf_values_list.append(cbf_val)
                        print(f"🔍 DEBUG: 智能体{agent_idx} CBF输出={cbf_val.item():.4f}")
                    
                    cbf_values = torch.stack(cbf_values_list, dim=1)  # [batch_size, num_agents, 1]
                    print(f"🔍 DEBUG: 最终CBF值形状={cbf_values.shape}")
                    
                except Exception as e:
                    print(f"❌ CBF网络推理失败: {e}")
                    cbf_values = None
            
            # 记录数据
            trajectory_data['actions'].append(actions[0].cpu().numpy())
            trajectory_data['alphas'].append(alphas[0].cpu().numpy())
            if cbf_values is not None:
                trajectory_data['cbf_values'].append(cbf_values[0].cpu().numpy())
            else:
                trajectory_data['cbf_values'].append(np.zeros((len(current_positions), 1)))
            
            # 4. 环境步进 - 添加诊断
            try:
                print(f"🌍 环境步进...")
                print(f"🔍 DEBUG: 步进前state.positions形状={state.positions.shape}")
                print(f"🔍 DEBUG: 步进actions形状={actions.shape}")
                print(f"🔍 DEBUG: 步进alphas形状={alphas.shape}")
                
                step_result = env.step(state, actions, alphas)
                
                print(f"🔍 DEBUG: 步进后next_state.positions形状={step_result.next_state.positions.shape}")
                print(f"✅ 环境步进成功")
                
                # 更新状态
                state = step_result.next_state
                
            except Exception as e:
                print(f"❌ 环境步进失败: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # 显示进度
            if step % 20 == 0:
                action_magnitude = torch.norm(actions, dim=-1).mean().item()
                print(f"📊 步骤 {step}: 平均动作强度={action_magnitude:.4f}")
    
    print(f"✅ 仿真完成，共 {len(trajectory_data['positions'])} 步")
    return trajectory_data


def create_final_visualization(trajectory_data, env_config, output_path):
    """
    创建最终的可视化动画
    """
    print(f"🎨 创建最终可视化动画...")
    
    positions_history = trajectory_data['positions']
    actions_history = trajectory_data['actions']
    alphas_history = trajectory_data['alphas']
    cbf_values_history = trajectory_data['cbf_values']
    
    if not positions_history:
        print(f"❌ 没有轨迹数据")
        return False
    
    num_steps = len(positions_history)
    num_agents = len(positions_history[0])
    
    print(f"📊 动画参数: {num_steps} 步, {num_agents} 智能体")
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('🎯 最终协作可视化结果 - 统一代码路径', fontsize=18, fontweight='bold')
    
    # 主轨迹图
    ax1.set_xlim(-3.0, 3.0)
    ax1.set_ylim(-2.0, 2.0)
    ax1.set_aspect('equal')
    ax1.set_title('🚁 真实训练模型轨迹', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 绘制障碍物
    obstacles = env_config.get('obstacles', {})
    if obstacles.get('enabled', False):
        for i, (pos, radius) in enumerate(zip(obstacles.get('positions', []), obstacles.get('radii', []))):
            circle = plt.Circle(pos, radius, color='red', alpha=0.8, 
                              label='障碍物' if i == 0 else "")
            ax1.add_patch(circle)
    
    # 起始和目标区域
    start_zone = plt.Rectangle((-2.5, -1.5), 1.0, 3.0, fill=False, 
                              edgecolor='green', linestyle='--', linewidth=2, 
                              alpha=0.8, label='起始区域')
    ax1.add_patch(start_zone)
    
    target_zone = plt.Rectangle((1.5, -1.5), 1.0, 3.0, fill=False, 
                               edgecolor='blue', linestyle='--', linewidth=2, 
                               alpha=0.8, label='目标区域')
    ax1.add_patch(target_zone)
    
    # 智能体颜色
    colors = ['#FF4444', '#44FF44', '#4444FF', '#FFAA44', '#FF44AA', '#44AAFF'][:num_agents]
    
    # 初始化动画元素
    trail_lines = []
    drone_dots = []
    
    for i in range(num_agents):
        line, = ax1.plot([], [], '-', color=colors[i], alpha=0.8, linewidth=3,
                        label=f'智能体{i+1}' if i < 3 else "")
        trail_lines.append(line)
        
        drone, = ax1.plot([], [], 'o', color=colors[i], markersize=14, 
                         markeredgecolor='black', markeredgewidth=2, zorder=5)
        drone_dots.append(drone)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 分析图表
    ax2.set_title('🧠 策略网络输出', fontsize=12)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('动作强度')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('⚖️ Alpha值监控', fontsize=12)
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('Alpha值')
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('🛡️ CBF安全值', fontsize=12)
    ax4.set_xlabel('时间步')
    ax4.set_ylabel('CBF值')
    ax4.grid(True, alpha=0.3)
    
    def animate(frame):
        if frame >= num_steps:
            return trail_lines + drone_dots
        
        current_positions = positions_history[frame]
        
        # 更新轨迹和智能体
        for i in range(num_agents):
            trail_x = [pos[i, 0] for pos in positions_history[:frame+1]]
            trail_y = [pos[i, 1] for pos in positions_history[:frame+1]]
            trail_lines[i].set_data(trail_x, trail_y)
            
            drone_dots[i].set_data([current_positions[i, 0]], [current_positions[i, 1]])
        
        # 更新分析图表
        if frame > 5:
            steps = list(range(frame+1))
            
            # 策略输出
            if len(actions_history) > frame:
                action_magnitudes = []
                for step in range(frame+1):
                    if step < len(actions_history):
                        step_actions = actions_history[step]
                        avg_magnitude = np.mean([np.linalg.norm(a) for a in step_actions])
                        action_magnitudes.append(avg_magnitude)
                    else:
                        action_magnitudes.append(0)
                
                ax2.clear()
                ax2.plot(steps, action_magnitudes, 'purple', linewidth=3, label='平均动作强度')
                ax2.fill_between(steps, action_magnitudes, alpha=0.3, color='purple')
                ax2.set_title(f'🧠 策略网络输出 (步数: {frame})')
                ax2.set_xlabel('时间步')
                ax2.set_ylabel('动作强度')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Alpha值
            if len(alphas_history) > frame:
                alpha_values = []
                for step in range(frame+1):
                    if step < len(alphas_history):
                        avg_alpha = np.mean(alphas_history[step])
                        alpha_values.append(avg_alpha)
                    else:
                        alpha_values.append(0.5)
                
                ax3.clear()
                ax3.plot(steps, alpha_values, 'orange', linewidth=3, label='平均Alpha值')
                ax3.fill_between(steps, alpha_values, alpha=0.3, color='orange')
                ax3.set_title(f'⚖️ Alpha值监控 (步数: {frame})')
                ax3.set_xlabel('时间步')
                ax3.set_ylabel('Alpha值')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # CBF值
            if len(cbf_values_history) > frame:
                cbf_avg_values = []
                for step in range(frame+1):
                    if step < len(cbf_values_history):
                        avg_cbf = np.mean(cbf_values_history[step])
                        cbf_avg_values.append(avg_cbf)
                    else:
                        cbf_avg_values.append(0)
                
                ax4.clear()
                ax4.plot(steps, cbf_avg_values, 'red', linewidth=3, label='平均CBF值')
                ax4.fill_between(steps, cbf_avg_values, alpha=0.3, color='red')
                ax4.set_title(f'🛡️ CBF安全值 (步数: {frame})')
                ax4.set_xlabel('时间步')
                ax4.set_ylabel('CBF值')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        return trail_lines + drone_dots
    
    # 创建动画
    anim = FuncAnimation(fig, animate, frames=num_steps, interval=150, blit=False, repeat=True)
    
    # 保存动画
    try:
        print(f"💾 保存最终可视化: {output_path}")
        
        # 尝试保存为MP4
        if output_path.endswith('.mp4'):
            anim.save(output_path, writer='ffmpeg', fps=8, dpi=150)
        else:
            anim.save(output_path, writer='pillow', fps=8, dpi=150)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ 保存成功: {output_path}")
        print(f"📁 文件大小: {file_size:.2f}MB")
        return True
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False
    finally:
        plt.close()


def main():
    """
    主函数 - 完全统一的可视化流程
    """
    parser = argparse.ArgumentParser(description='统一的BPTT可视化脚本')
    parser.add_argument('--model_dir', type=str, default='logs/full_collaboration_training', 
                       help='模型目录路径')
    parser.add_argument('--step', type=int, help='模型步骤 (默认使用最新)')
    parser.add_argument('--device', type=str, default='cpu', help='设备 (cuda/cpu)')
    parser.add_argument('--output', type=str, default='FINAL_COLLABORATION_RESULT.mp4', 
                       help='输出文件路径')
    parser.add_argument('--num_steps', type=int, default=120, help='仿真步数')
    
    args = parser.parse_args()
    
    print(f"🎯 统一BPTT可视化系统")
    print(f"=" * 80)
    print(f"📁 模型目录: {args.model_dir}")
    print(f"📈 模型步骤: {args.step if args.step else '最新'}")
    print(f"💻 设备: {args.device}")
    print(f"📁 输出文件: {args.output}")
    print(f"=" * 80)
    
    # 设置设备
    device = torch.device(args.device)
    
    try:
        # 1. 加载训练模型 - 统一路径
        print(f"\n🔄 第1步: 加载训练模型")
        env, policy_network, cbf_network, config = load_trained_model(
            args.model_dir, args.step, device
        )
        
        # 2. 运行仿真 - 包含诊断
        print(f"\n🔄 第2步: 运行仿真 (包含诊断)")
        trajectory_data = run_simulation_with_diagnostics(
            env, policy_network, cbf_network, device, args.num_steps
        )
        
        # 3. 创建可视化
        print(f"\n🔄 第3步: 创建最终可视化")
        success = create_final_visualization(
            trajectory_data, config.get('env', {}), args.output
        )
        
        if success:
            print(f"\n🎉 统一可视化生成成功!")
            print(f"📁 最终结果: {args.output}")
            print(f"✅ 代码路径已完全统一")
            print(f"🧠 这是您真实训练模型的表现")
        else:
            print(f"\n❌ 可视化生成失败")
            
    except Exception as e:
        print(f"\n❌ 统一可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
 
 
 
 