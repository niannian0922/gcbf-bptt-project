#!/usr/bin/env python3
"""
🔍 诊断静止模型问题
找出为什么真实模型输出接近零动作
"""

import torch
import numpy as np
import os

print("🔍 诊断静止模型问题")
print("=" * 50)

# 加载模型
model_path = 'logs/full_collaboration_training/models/500/policy.pt'
device = torch.device('cpu')

try:
    policy_dict = torch.load(model_path, map_location=device, weights_only=True)
    print(f"✅ 模型加载成功: {len(policy_dict)} 层")
    
    # 分析模型权重
    print("\n🔍 分析模型权重分布:")
    for key, param in policy_dict.items():
        if 'weight' in key and param.numel() > 100:
            mean_val = torch.mean(torch.abs(param)).item()
            max_val = torch.max(torch.abs(param)).item()
            std_val = torch.std(param).item()
            print(f"  {key}: 均值={mean_val:.6f}, 最大={max_val:.6f}, 标准差={std_val:.6f}")
            
            if mean_val < 1e-6:
                print(f"    ⚠️ 权重过小，可能导致输出接近零")
    
    # 检查输出层
    if 'policy_head.action_mlp.2.weight' in policy_dict:
        output_weights = policy_dict['policy_head.action_mlp.2.weight']
        print(f"\n🎯 输出层权重: {output_weights.shape}")
        print(f"   输出权重统计: 均值={torch.mean(torch.abs(output_weights)):.6f}")
        print(f"   输出权重范围: [{torch.min(output_weights):.6f}, {torch.max(output_weights):.6f}]")
        
        if torch.max(torch.abs(output_weights)) < 0.01:
            print("   ❌ 输出层权重过小，会导致动作接近零!")
        
except Exception as e:
    print(f"❌ 模型分析失败: {e}")
    exit()

# 导入环境并测试
try:
    from gcbfplus.env import DoubleIntegratorEnv
    from gcbfplus.env.multi_agent_env import MultiAgentState
    from gcbfplus.policy.bptt_policy import BPTTPolicy
    print("\n✅ 环境模块导入成功")
except Exception as e:
    print(f"❌ 环境导入失败: {e}")
    exit()

# 创建简单测试环境
env_config = {
    'num_agents': 6,
    'area_size': 4.0,
    'dt': 0.02,
    'mass': 0.5,
    'agent_radius': 0.15,
    'max_force': 1.0,
    'max_steps': 100,
    'obstacles': {
        'enabled': True,
        'count': 2,
        'positions': [[0, 0.7], [0, -0.7]],
        'radii': [0.3, 0.3]
    }
}

try:
    env = DoubleIntegratorEnv(env_config)
    env = env.to(device)
    print(f"✅ 测试环境创建成功")
except Exception as e:
    print(f"❌ 环境创建失败: {e}")
    exit()

# 创建策略网络
input_dim = 9  # 有障碍物环境
try:
    policy_config = {
        'input_dim': input_dim,
        'output_dim': 2,
        'hidden_dim': 256,
        'node_dim': input_dim,
        'edge_dim': 4,
        'n_layers': 2,
        'msg_hidden_sizes': [256, 256],
        'aggr_hidden_sizes': [256],
        'update_hidden_sizes': [256, 256],
        'predict_alpha': True,
        'perception': {
            'input_dim': input_dim,
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
    exit()

# 详细测试不同场景
print("\n🧪 测试不同场景的策略输出:")

test_scenarios = [
    {
        'name': '远距离目标',
        'start_positions': [[-2.0, 0], [-2.0, 0.3], [-2.0, -0.3], [-2.0, 0.6], [-2.0, -0.6], [-2.0, 0.9]],
        'goal_positions': [[2.0, 0], [2.0, 0.3], [2.0, -0.3], [2.0, 0.6], [2.0, -0.6], [2.0, 0.9]]
    },
    {
        'name': '紧急避障',
        'start_positions': [[-0.5, 0], [-0.3, 0], [-0.1, 0], [0.1, 0], [0.3, 0], [0.5, 0]],
        'goal_positions': [[2.0, 0], [2.0, 0.3], [2.0, -0.3], [2.0, 0.6], [2.0, -0.6], [2.0, 0.9]]
    },
    {
        'name': '极端距离',
        'start_positions': [[-3.0, 0], [-3.0, 0.2], [-3.0, -0.2], [-3.0, 0.4], [-3.0, -0.4], [-3.0, 0.6]],
        'goal_positions': [[3.0, 0], [3.0, 0.2], [3.0, -0.2], [3.0, 0.4], [3.0, -0.4], [3.0, 0.6]]
    }
]

with torch.no_grad():
    for scenario in test_scenarios:
        print(f"\n📍 场景: {scenario['name']}")
        
        # 创建测试状态
        num_agents = len(scenario['start_positions'])
        positions = torch.tensor(scenario['start_positions'], device=device).unsqueeze(0).float()
        goals = torch.tensor(scenario['goal_positions'], device=device).unsqueeze(0).float()
        velocities = torch.zeros_like(positions)
        
        state = MultiAgentState(
            positions=positions,
            velocities=velocities,
            goals=goals,
            batch_size=1
        )
        
        try:
            # 获取观测
            observations = env.get_observations(state)
            print(f"   观测形状: {observations.shape}")
            print(f"   观测范围: [{torch.min(observations):.4f}, {torch.max(observations):.4f}]")
            
            # 策略推理
            policy_output = policy(observations, state)
            actions = policy_output.actions[0].cpu().numpy()
            
            # 分析动作
            action_magnitudes = [np.linalg.norm(a) for a in actions]
            avg_action_mag = np.mean(action_magnitudes)
            max_action_mag = np.max(action_magnitudes)
            
            print(f"   动作形状: {actions.shape}")
            print(f"   平均动作强度: {avg_action_mag:.6f}")
            print(f"   最大动作强度: {max_action_mag:.6f}")
            print(f"   动作范围: [{np.min(actions):.6f}, {np.max(actions):.6f}]")
            
            # 计算期望的动作强度
            goal_distances = [np.linalg.norm(scenario['goal_positions'][i] - scenario['start_positions'][i]) for i in range(num_agents)]
            avg_goal_distance = np.mean(goal_distances)
            expected_action = min(0.5, avg_goal_distance * 0.1)  # 简单启发式
            
            print(f"   平均目标距离: {avg_goal_distance:.3f}")
            print(f"   期望动作强度: ~{expected_action:.3f}")
            
            if avg_action_mag < 0.001:
                print("   ❌ 动作强度过小!")
                
                # 详细分析每个智能体
                for i, (pos, goal, action) in enumerate(zip(scenario['start_positions'], scenario['goal_positions'], actions)):
                    direction = np.array(goal) - np.array(pos)
                    distance = np.linalg.norm(direction)
                    unit_direction = direction / distance if distance > 0 else np.array([0, 0])
                    
                    action_magnitude = np.linalg.norm(action)
                    action_direction = action / action_magnitude if action_magnitude > 0 else np.array([0, 0])
                    
                    alignment = np.dot(unit_direction, action_direction) if action_magnitude > 0 else 0
                    
                    print(f"     智能体{i}: 距离={distance:.3f}, 动作强度={action_magnitude:.6f}, 方向对齐={alignment:.3f}")
            else:
                print("   ✅ 检测到有效动作")
                
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")

print("\n🔍 诊断结论:")
print("如果所有场景的动作强度都 < 0.001:")
print("  1. 模型可能收敛到静止策略")
print("  2. 输出层权重可能过小")
print("  3. 训练过程可能有问题")
print("  4. 观测预处理可能不匹配")

print("\n💡 建议解决方案:")
print("  1. 检查训练日志确认损失是否正常下降")
print("  2. 尝试加载训练早期的检查点")
print("  3. 手动设置合理的动作来生成'应该的'可视化")
print("  4. 检查CBF网络是否过度抑制了动作")
 
"""
🔍 诊断静止模型问题
找出为什么真实模型输出接近零动作
"""

import torch
import numpy as np
import os

print("🔍 诊断静止模型问题")
print("=" * 50)

# 加载模型
model_path = 'logs/full_collaboration_training/models/500/policy.pt'
device = torch.device('cpu')

try:
    policy_dict = torch.load(model_path, map_location=device, weights_only=True)
    print(f"✅ 模型加载成功: {len(policy_dict)} 层")
    
    # 分析模型权重
    print("\n🔍 分析模型权重分布:")
    for key, param in policy_dict.items():
        if 'weight' in key and param.numel() > 100:
            mean_val = torch.mean(torch.abs(param)).item()
            max_val = torch.max(torch.abs(param)).item()
            std_val = torch.std(param).item()
            print(f"  {key}: 均值={mean_val:.6f}, 最大={max_val:.6f}, 标准差={std_val:.6f}")
            
            if mean_val < 1e-6:
                print(f"    ⚠️ 权重过小，可能导致输出接近零")
    
    # 检查输出层
    if 'policy_head.action_mlp.2.weight' in policy_dict:
        output_weights = policy_dict['policy_head.action_mlp.2.weight']
        print(f"\n🎯 输出层权重: {output_weights.shape}")
        print(f"   输出权重统计: 均值={torch.mean(torch.abs(output_weights)):.6f}")
        print(f"   输出权重范围: [{torch.min(output_weights):.6f}, {torch.max(output_weights):.6f}]")
        
        if torch.max(torch.abs(output_weights)) < 0.01:
            print("   ❌ 输出层权重过小，会导致动作接近零!")
        
except Exception as e:
    print(f"❌ 模型分析失败: {e}")
    exit()

# 导入环境并测试
try:
    from gcbfplus.env import DoubleIntegratorEnv
    from gcbfplus.env.multi_agent_env import MultiAgentState
    from gcbfplus.policy.bptt_policy import BPTTPolicy
    print("\n✅ 环境模块导入成功")
except Exception as e:
    print(f"❌ 环境导入失败: {e}")
    exit()

# 创建简单测试环境
env_config = {
    'num_agents': 6,
    'area_size': 4.0,
    'dt': 0.02,
    'mass': 0.5,
    'agent_radius': 0.15,
    'max_force': 1.0,
    'max_steps': 100,
    'obstacles': {
        'enabled': True,
        'count': 2,
        'positions': [[0, 0.7], [0, -0.7]],
        'radii': [0.3, 0.3]
    }
}

try:
    env = DoubleIntegratorEnv(env_config)
    env = env.to(device)
    print(f"✅ 测试环境创建成功")
except Exception as e:
    print(f"❌ 环境创建失败: {e}")
    exit()

# 创建策略网络
input_dim = 9  # 有障碍物环境
try:
    policy_config = {
        'input_dim': input_dim,
        'output_dim': 2,
        'hidden_dim': 256,
        'node_dim': input_dim,
        'edge_dim': 4,
        'n_layers': 2,
        'msg_hidden_sizes': [256, 256],
        'aggr_hidden_sizes': [256],
        'update_hidden_sizes': [256, 256],
        'predict_alpha': True,
        'perception': {
            'input_dim': input_dim,
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
    exit()

# 详细测试不同场景
print("\n🧪 测试不同场景的策略输出:")

test_scenarios = [
    {
        'name': '远距离目标',
        'start_positions': [[-2.0, 0], [-2.0, 0.3], [-2.0, -0.3], [-2.0, 0.6], [-2.0, -0.6], [-2.0, 0.9]],
        'goal_positions': [[2.0, 0], [2.0, 0.3], [2.0, -0.3], [2.0, 0.6], [2.0, -0.6], [2.0, 0.9]]
    },
    {
        'name': '紧急避障',
        'start_positions': [[-0.5, 0], [-0.3, 0], [-0.1, 0], [0.1, 0], [0.3, 0], [0.5, 0]],
        'goal_positions': [[2.0, 0], [2.0, 0.3], [2.0, -0.3], [2.0, 0.6], [2.0, -0.6], [2.0, 0.9]]
    },
    {
        'name': '极端距离',
        'start_positions': [[-3.0, 0], [-3.0, 0.2], [-3.0, -0.2], [-3.0, 0.4], [-3.0, -0.4], [-3.0, 0.6]],
        'goal_positions': [[3.0, 0], [3.0, 0.2], [3.0, -0.2], [3.0, 0.4], [3.0, -0.4], [3.0, 0.6]]
    }
]

with torch.no_grad():
    for scenario in test_scenarios:
        print(f"\n📍 场景: {scenario['name']}")
        
        # 创建测试状态
        num_agents = len(scenario['start_positions'])
        positions = torch.tensor(scenario['start_positions'], device=device).unsqueeze(0).float()
        goals = torch.tensor(scenario['goal_positions'], device=device).unsqueeze(0).float()
        velocities = torch.zeros_like(positions)
        
        state = MultiAgentState(
            positions=positions,
            velocities=velocities,
            goals=goals,
            batch_size=1
        )
        
        try:
            # 获取观测
            observations = env.get_observations(state)
            print(f"   观测形状: {observations.shape}")
            print(f"   观测范围: [{torch.min(observations):.4f}, {torch.max(observations):.4f}]")
            
            # 策略推理
            policy_output = policy(observations, state)
            actions = policy_output.actions[0].cpu().numpy()
            
            # 分析动作
            action_magnitudes = [np.linalg.norm(a) for a in actions]
            avg_action_mag = np.mean(action_magnitudes)
            max_action_mag = np.max(action_magnitudes)
            
            print(f"   动作形状: {actions.shape}")
            print(f"   平均动作强度: {avg_action_mag:.6f}")
            print(f"   最大动作强度: {max_action_mag:.6f}")
            print(f"   动作范围: [{np.min(actions):.6f}, {np.max(actions):.6f}]")
            
            # 计算期望的动作强度
            goal_distances = [np.linalg.norm(scenario['goal_positions'][i] - scenario['start_positions'][i]) for i in range(num_agents)]
            avg_goal_distance = np.mean(goal_distances)
            expected_action = min(0.5, avg_goal_distance * 0.1)  # 简单启发式
            
            print(f"   平均目标距离: {avg_goal_distance:.3f}")
            print(f"   期望动作强度: ~{expected_action:.3f}")
            
            if avg_action_mag < 0.001:
                print("   ❌ 动作强度过小!")
                
                # 详细分析每个智能体
                for i, (pos, goal, action) in enumerate(zip(scenario['start_positions'], scenario['goal_positions'], actions)):
                    direction = np.array(goal) - np.array(pos)
                    distance = np.linalg.norm(direction)
                    unit_direction = direction / distance if distance > 0 else np.array([0, 0])
                    
                    action_magnitude = np.linalg.norm(action)
                    action_direction = action / action_magnitude if action_magnitude > 0 else np.array([0, 0])
                    
                    alignment = np.dot(unit_direction, action_direction) if action_magnitude > 0 else 0
                    
                    print(f"     智能体{i}: 距离={distance:.3f}, 动作强度={action_magnitude:.6f}, 方向对齐={alignment:.3f}")
            else:
                print("   ✅ 检测到有效动作")
                
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")

print("\n🔍 诊断结论:")
print("如果所有场景的动作强度都 < 0.001:")
print("  1. 模型可能收敛到静止策略")
print("  2. 输出层权重可能过小")
print("  3. 训练过程可能有问题")
print("  4. 观测预处理可能不匹配")

print("\n💡 建议解决方案:")
print("  1. 检查训练日志确认损失是否正常下降")
print("  2. 尝试加载训练早期的检查点")
print("  3. 手动设置合理的动作来生成'应该的'可视化")
print("  4. 检查CBF网络是否过度抑制了动作")
 
 
 
 