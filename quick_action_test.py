#!/usr/bin/env python3
"""
🔍 快速动作测试
检查策略网络是否输出有效动作
"""

import torch
import numpy as np
import yaml

def quick_action_test():
    """快速测试动作输出"""
    print("🔍 快速动作测试")
    print("=" * 40)
    
    try:
        # 加载配置
        with open('config/simple_collaboration.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ 配置加载成功")
        
        # 创建环境
        from gcbfplus.env import DoubleIntegratorEnv
        device = torch.device('cpu')
        env = DoubleIntegratorEnv(config['env'])
        
        print(f"✅ 环境创建成功: 观测{env.observation_shape}, 动作{env.action_shape}")
        
        # 创建简单测试状态
        from gcbfplus.env.multi_agent_env import MultiAgentState
        
        num_agents = config['env']['num_agents']
        positions = torch.randn(1, num_agents, 2) * 0.5
        velocities = torch.zeros(1, num_agents, 2)
        goals = torch.randn(1, num_agents, 2) * 0.5
        
        state = MultiAgentState(
            positions=positions,
            velocities=velocities,
            goals=goals,
            batch_size=1
        )
        
        print(f"✅ 测试状态创建成功")
        
        # 测试观测
        obs = env.get_observation(state)
        print(f"✅ 观测生成: {obs.shape}, 范围[{obs.min():.3f}, {obs.max():.3f}]")
        
        # 创建策略网络配置
        policy_config = {
            'type': 'bptt',
            'hidden_dim': 256,
            'input_dim': 6,
            'node_dim': 6,
            'edge_dim': 4,
            'n_layers': 2,
            'msg_hidden_sizes': [256, 256],
            'aggr_hidden_sizes': [256],
            'update_hidden_sizes': [256, 256],
            'predict_alpha': True,
            'perception': {
                'input_dim': 6,
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
            }
        }
        
        # 创建策略网络
        from gcbfplus.policy.bptt_policy import create_policy_from_config
        policy = create_policy_from_config(policy_config)
        
        print(f"✅ 策略网络创建成功")
        
        # 测试随机权重动作
        with torch.no_grad():
            actions_random, alphas_random = policy(obs)
        
        action_mag = torch.norm(actions_random, dim=-1)
        print(f"🎲 随机权重动作:")
        print(f"   形状: {actions_random.shape}")
        print(f"   幅度: 平均={action_mag.mean():.4f}, 最大={action_mag.max():.4f}")
        print(f"   第一个智能体: {actions_random[0, 0]}")
        
        # 尝试加载训练权重
        model_path = "logs/full_collaboration_training/models/500/policy.pt"
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            policy.load_state_dict(state_dict)
            print(f"✅ 训练权重加载成功")
            
            # 测试训练权重动作
            with torch.no_grad():
                actions_trained, alphas_trained = policy(obs)
            
            action_mag_trained = torch.norm(actions_trained, dim=-1)
            print(f"🎯 训练权重动作:")
            print(f"   形状: {actions_trained.shape}")
            print(f"   幅度: 平均={action_mag_trained.mean():.4f}, 最大={action_mag_trained.max():.4f}")
            print(f"   第一个智能体: {actions_trained[0, 0]}")
            
            # 比较
            if action_mag_trained.max() < 1e-6:
                print(f"❌ 问题发现: 训练权重输出极小动作!")
                print(f"   这就是智能体不动的原因!")
            else:
                print(f"✅ 训练权重输出正常动作")
                
        except Exception as e:
            print(f"❌ 训练权重加载失败: {e}")
        
        # 测试环境步进
        test_action = torch.ones(1, num_agents, 2) * 0.1  # 小的测试动作
        try:
            result = env.step(state, test_action, None)
            pos_change = torch.norm(result.next_state.positions - state.positions, dim=-1)
            print(f"🔄 环境步进测试:")
            print(f"   输入动作: {test_action[0, 0]}")
            print(f"   位置变化: 平均={pos_change.mean():.6f}, 最大={pos_change.max():.6f}")
        except Exception as e:
            print(f"❌ 环境步进失败: {e}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_action_test()