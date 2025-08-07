#!/usr/bin/env python3
"""
🔍 诊断智能体不动的问题
检查策略网络、环境配置、动作生成等关键环节
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import os

def diagnose_static_agents():
    """诊断智能体静止的原因"""
    print("🔍 诊断智能体静止问题")
    print("=" * 60)
    
    try:
        # 1. 检查模型文件
        model_dir = "logs/full_collaboration_training/models/500"
        policy_path = os.path.join(model_dir, "policy.pt")
        
        print("📁 检查模型文件...")
        if not os.path.exists(policy_path):
            print(f"❌ 策略模型未找到: {policy_path}")
            return False
        
        file_size = os.path.getsize(policy_path) / (1024 * 1024)  # MB
        print(f"✅ 策略模型存在: {file_size:.1f}MB")
        
        # 2. 加载配置
        print("\n📋 加载配置...")
        with open('config/simple_collaboration.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 补充网络配置
        config['networks'] = {
            'policy': {
                'type': 'bptt',
                'layers': [256, 256],
                'activation': 'relu',
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
        }
        
        print(f"✅ 配置加载成功")
        print(f"   🤖 智能体数量: {config['env']['num_agents']}")
        print(f"   📐 社交半径: {config['env']['social_radius']}")
        print(f"   ⚡ 最大力: {config['env']['max_force']}")
        print(f"   ⏰ 时间步长: {config['env']['dt']}")
        
        # 3. 创建环境
        print("\n🌍 创建环境...")
        from gcbfplus.env import DoubleIntegratorEnv
        from gcbfplus.env.multi_agent_env import MultiAgentState
        
        device = torch.device('cpu')
        env = DoubleIntegratorEnv(config['env'])
        env = env.to(device)
        
        print(f"✅ 环境创建成功")
        print(f"   📊 观测维度: {env.observation_shape}")
        print(f"   🎯 动作维度: {env.action_shape}")
        print(f"   🤖 智能体半径: {env.agent_radius}")
        print(f"   ⚡ 最大力: {env.max_force}")
        
        # 4. 创建策略网络
        print("\n🧠 创建策略网络...")
        from gcbfplus.policy.bptt_policy import create_policy_from_config
        
        policy_network = create_policy_from_config(config['networks']['policy'])
        policy_network = policy_network.to(device)
        
        print(f"✅ 策略网络创建成功")
        
        # 5. 尝试加载权重
        print("\n💾 加载策略权重...")
        try:
            policy_state_dict = torch.load(policy_path, map_location='cpu', weights_only=True)
            policy_network.load_state_dict(policy_state_dict)
            print(f"✅ 策略权重加载成功")
            weights_loaded = True
        except Exception as e:
            print(f"❌ 策略权重加载失败: {e}")
            print(f"🔧 使用随机权重测试...")
            weights_loaded = False
        
        # 6. 创建测试状态
        print("\n🎬 创建测试状态...")
        batch_size = 1
        num_agents = config['env']['num_agents']
        
        # 创建分散的初始位置
        positions = torch.zeros(batch_size, num_agents, 2, device=device)
        velocities = torch.zeros(batch_size, num_agents, 2, device=device)
        goals = torch.zeros(batch_size, num_agents, 2, device=device)
        
        for i in range(num_agents):
            # 左侧起始位置
            positions[0, i] = torch.tensor([-1.5 + i * 0.3, (i - num_agents/2) * 0.5], device=device)
            # 右侧目标位置
            goals[0, i] = torch.tensor([1.5 + i * 0.3, (i - num_agents/2) * 0.5], device=device)
        
        test_state = MultiAgentState(
            positions=positions,
            velocities=velocities,
            goals=goals,
            batch_size=batch_size
        )
        
        print(f"✅ 测试状态创建成功")
        print(f"   📍 起始位置: {positions[0, 0]}")
        print(f"   🎯 目标位置: {goals[0, 0]}")
        
        # 7. 测试观测生成
        print("\n👁️ 测试观测生成...")
        try:
            observations = env.get_observation(test_state)
            print(f"✅ 观测生成成功")
            print(f"   📊 观测形状: {observations.shape}")
            print(f"   📈 观测范围: [{observations.min():.3f}, {observations.max():.3f}]")
            print(f"   📋 第一个智能体观测: {observations[0, 0, :3]}")  # 显示前3维
        except Exception as e:
            print(f"❌ 观测生成失败: {e}")
            return False
        
        # 8. 测试策略网络前向传播
        print("\n🤖 测试策略网络...")
        try:
            with torch.no_grad():
                actions, alphas = policy_network(observations)
            
            print(f"✅ 策略网络工作正常")
            print(f"   🎯 动作形状: {actions.shape}")
            print(f"   📈 动作范围: [{actions.min():.3f}, {actions.max():.3f}]")
            print(f"   🔄 Alpha形状: {alphas.shape if alphas is not None else 'None'}")
            
            # 检查动作是否为零
            action_magnitude = torch.norm(actions, dim=-1)
            print(f"   ⚡ 动作幅度: 平均={action_magnitude.mean():.6f}, 最大={action_magnitude.max():.6f}")
            
            if action_magnitude.max() < 1e-6:
                print(f"⚠️ 警告: 动作幅度极小，智能体可能不会移动!")
                print(f"   可能原因: 1) 权重未正确加载 2) 网络输出被约束 3) 学习率过小")
            
            # 显示具体动作
            print(f"   📋 第一个智能体动作: {actions[0, 0]}")
            print(f"   📋 第二个智能体动作: {actions[0, 1]}")
            
        except Exception as e:
            print(f"❌ 策略网络测试失败: {e}")
            return False
        
        # 9. 测试环境步进
        print("\n🔄 测试环境步进...")
        try:
            step_result = env.step(test_state, actions, alphas)
            new_state = step_result.next_state
            
            print(f"✅ 环境步进成功")
            
            # 检查位置变化
            position_change = torch.norm(new_state.positions - test_state.positions, dim=-1)
            print(f"   📏 位置变化: 平均={position_change.mean():.6f}, 最大={position_change.max():.6f}")
            
            if position_change.max() < 1e-6:
                print(f"⚠️ 警告: 位置变化极小，智能体实际上没有移动!")
                print(f"   📍 原始位置: {test_state.positions[0, 0]}")
                print(f"   📍 新位置: {new_state.positions[0, 0]}")
                print(f"   ⚡ 应用动作: {actions[0, 0]}")
                print(f"   ⏰ 时间步长: {env.dt}")
                print(f"   💪 最大力: {env.max_force}")
            else:
                print(f"✅ 智能体正常移动")
                print(f"   📍 位置变化: {new_state.positions[0, 0] - test_state.positions[0, 0]}")
            
        except Exception as e:
            print(f"❌ 环境步进失败: {e}")
            return False
        
        # 10. 分析问题
        print(f"\n📊 问题分析总结:")
        print(f"=" * 50)
        
        if not weights_loaded:
            print(f"🔴 主要问题: 策略权重未正确加载")
            print(f"   💡 解决方案: 检查权重文件兼容性")
        elif action_magnitude.max() < 1e-6:
            print(f"🔴 主要问题: 策略网络输出极小动作")
            print(f"   💡 可能原因: 网络权重初始化、学习不充分、或输出被限制")
        elif position_change.max() < 1e-6:
            print(f"🔴 主要问题: 环境未响应动作输入")
            print(f"   💡 可能原因: 时间步长过小、动作缩放问题")
        else:
            print(f"🟢 所有组件正常工作")
            print(f"   💡 问题可能在可视化脚本的动画逻辑")
        
        # 11. 建议的修复方案
        print(f"\n🛠️ 建议修复方案:")
        print(f"1. 使用随机动作测试环境响应性")
        print(f"2. 检查策略网络权重文件")
        print(f"3. 调整动作缩放和时间步长")
        print(f"4. 验证可视化动画逻辑")
        
        return True
        
    except Exception as e:
        print(f"❌ 诊断失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 智能体静止问题诊断系统")
    print("检查策略网络、环境配置、动作生成等关键环节")
    print("=" * 70)
    
    success = diagnose_static_agents()
    
    if success:
        print(f"\n✅ 诊断完成!")
    else:
        print(f"\n❌ 诊断失败，需要进一步调试")