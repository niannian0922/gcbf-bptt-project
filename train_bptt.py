#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for multi-agent navigation with dynamic alpha
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path

from gcbfplus.env import DoubleIntegratorEnv
from gcbfplus.env.gcbf_safety_layer import GCBFSafetyLayer
from gcbfplus.policy import BPTTPolicy, create_policy_from_config
from gcbfplus.trainer.bptt_trainer import BPTTTrainer


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练BPTT策略')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--cuda', action='store_true', help='使用CUDA')
    parser.add_argument('--device', type=str, default=None, help='指定设备（cuda或cpu）')
    parser.add_argument('--env_type', type=str, default='double_integrator', help='环境类型')
    parser.add_argument('--log_dir', type=str, default='logs/bptt', help='日志目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置设备 - 强制使用GPU（如果可用）
    if args.device:
        device = torch.device(args.device)
    elif args.cuda or torch.cuda.is_available():
        device = torch.device('cuda')
        # 传统支持cuda标志
    else:
        device = torch.device('cpu')
    
    # 设置随机种子以确保可重现性
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # 启用异常检测以帮助调试梯度问题
    torch.autograd.set_detect_anomaly(True)
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 将配置保存到日志目录
    with open(os.path.join(args.log_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    # 使用默认值提取配置部分，处理缺失部分
    env_config = config.get('env', {})
    training_config = config.get('training', {})
    network_config = config.get('networks', {})
    
    # 提取策略和CBF网络配置
    policy_config = network_config.get('policy', {})
    cbf_network_config = network_config.get('cbf')  # 可能为None
    
    # 从命令行参数获取环境类型
    env_type = args.env_type
    
    # 1. 基于配置创建环境
    if env_type == 'double_integrator':
        env = DoubleIntegratorEnv(env_config)
    else:
        raise ValueError(f"不支持的环境类型: {env_type}")
    
    print(f"创建环境: {env_type}")
    
    # 将环境移动到设备
    env = env.to(device)
    
    # 2. 创建策略网络
    # 使用YAML文件中的策略配置
    if policy_config:
        # 确保策略配置具有正确的观测和动作维度
        obs_shape = env.observation_shape
        action_shape = env.action_shape
        
        print(f"观测形状: {obs_shape}")
        print(f"动作形状: {action_shape}")
        
        # 如果需要，为缺失的感知配置添加默认值
        if 'perception' not in policy_config:
            policy_config['perception'] = {}
        
        perception_config = policy_config['perception']
        
        # 处理视觉输入
        if len(obs_shape) > 2:  # 视觉输入 [n_agents, channels, height, width]
            perception_config.update({
                'use_vision': True,
                'input_dim': obs_shape[-3:],  # [channels, height, width]
                'output_dim': perception_config.get('output_dim', 256)
            })
        else:  # 状态输入 [n_agents, obs_dim]
            perception_config.update({
                'use_vision': False,
                'output_dim': perception_config.get('output_dim', 128),
                'hidden_dims': perception_config.get('hidden_dims', [256, 256])
            })
            # 只在config中沒有明確指定時才設置input_dim
            if 'input_dim' not in perception_config:
                perception_config['input_dim'] = obs_shape[-1]
        
        # 如果需要，添加默认记忆配置
        if 'memory' not in policy_config:
            policy_config['memory'] = {}
        
        memory_config = policy_config['memory']
        memory_config.update({
            'hidden_dim': memory_config.get('hidden_dim', 128),
            'num_layers': memory_config.get('num_layers', 1)
        })
        
        # 确保policy_head具有所有必需参数
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
        
        print(f"策略配置: {policy_config}")
    else:
        # 后备方案：如果YAML中没有策略配置，创建默认配置
        obs_shape = env.observation_shape
        action_shape = env.action_shape
        
        print(f"观测形状: {obs_shape}")
        print(f"动作形状: {action_shape}")
        
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
    
    # 创建策略网络
    policy_network = create_policy_from_config(policy_config)
    policy_network = policy_network.to(device)
    
    # 3. 创建CBF网络（可选）
    cbf_network = None
    if cbf_network_config:
        # 从配置中提取CBF alpha参数（训练器配置需要）
        cbf_alpha = cbf_network_config.get('alpha', 1.0)
        
        # 基于CBF网络配置创建CBF网络
        # 这里使用简单的MLP作为CBF网络的占位符
        obs_dim = obs_shape[-1] if len(obs_shape) <= 2 else np.prod(obs_shape[-3:])
        cbf_network = nn.Sequential(
            nn.Linear(obs_dim * env_config.get('num_agents', 8), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)
        
        print(f"创建CBF网络，alpha={cbf_alpha}")
    
    # 4. 创建训练器
    trainer_config = {
        'horizon_length': training_config.get('horizon_length', 64),
        'learning_rate': training_config.get('learning_rate', 1e-4),
        'training_steps': training_config.get('steps', 10000),
        'loss_weights': training_config.get('loss_weights', {}),
        'batch_size': training_config.get('batch_size', 32),
        'device': device,
        'log_interval': training_config.get('log_interval', 100),
        'save_interval': training_config.get('save_interval', 1000),
        'cbf_alpha': env_config.get('cbf_alpha', 1.0),  # 传递CBF alpha给训练器
        'temporal_decay': training_config.get('temporal_decay', {}),
        'wandb_config': training_config.get('wandb', {})
    }
    
    trainer = BPTTTrainer(
        env=env,
        policy_network=policy_network,
        cbf_network=cbf_network,
        config=trainer_config
    )
    
    print(f"开始训练，共{trainer_config['training_steps']}步...")
    print(f"使用设备: {device}")
    print(f"时间范围长度: {trainer_config['horizon_length']}")
    print(f"学习率: {trainer_config['learning_rate']}")
    print(f"批处理大小: {trainer_config['batch_size']}")
    
    # 保存目录已在 BPTTTrainer 初始化时设置
    
    # 开始训练
    trainer.train()
    
    print("训练完成！")

if __name__ == '__main__':
    main() 