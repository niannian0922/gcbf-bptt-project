#!/usr/bin/env python3
"""
🔧 调试真实模型加载
逐步诊断加载问题
"""

import torch
import os
import sys

def debug_model_loading():
    """逐步调试模型加载"""
    print("🔧 真实模型加载调试器")
    print("=" * 60)
    
    # 步骤1：检查文件存在
    print("📁 步骤1: 检查模型文件...")
    model_path = "logs/full_collaboration_training/models/500/"
    policy_path = os.path.join(model_path, "policy.pt")
    cbf_path = os.path.join(model_path, "cbf.pt")
    config_path = os.path.join(model_path, "config.pt")
    
    files_exist = {
        'policy.pt': os.path.exists(policy_path),
        'cbf.pt': os.path.exists(cbf_path),
        'config.pt': os.path.exists(config_path)
    }
    
    for filename, exists in files_exist.items():
        status = "✅" if exists else "❌"
        if exists:
            size = os.path.getsize(os.path.join(model_path, filename)) / 1024  # KB
            print(f"   {status} {filename}: {size:.1f}KB")
        else:
            print(f"   {status} {filename}: 不存在")
    
    if not all(files_exist.values()):
        print("❌ 模型文件不完整，无法继续")
        return False
    
    # 步骤2：测试基础导入
    print("\n📦 步骤2: 测试模块导入...")
    try:
        from gcbfplus.env import DoubleIntegratorEnv
        print("   ✅ DoubleIntegratorEnv 导入成功")
    except Exception as e:
        print(f"   ❌ DoubleIntegratorEnv 导入失败: {e}")
        return False
    
    try:
        from gcbfplus.env.multi_agent_env import MultiAgentState
        print("   ✅ MultiAgentState 导入成功")
    except Exception as e:
        print(f"   ❌ MultiAgentState 导入失败: {e}")
        return False
    
    try:
        from gcbfplus.policy.bptt_policy import BPTTPolicy
        print("   ✅ BPTTPolicy 导入成功")
    except Exception as e:
        print(f"   ❌ BPTTPolicy 导入失败: {e}")
        return False
    
    # 步骤3：加载配置
    print("\n📋 步骤3: 加载配置文件...")
    try:
        config = torch.load(config_path, map_location='cpu', weights_only=False)
        print(f"   ✅ 配置加载成功")
        print(f"   📊 配置类型: {type(config)}")
        if isinstance(config, dict):
            print(f"   📝 配置键: {list(config.keys())}")
        else:
            print(f"   📝 配置内容: {config}")
    except Exception as e:
        print(f"   ❌ 配置加载失败: {e}")
        print("   🔧 使用备用配置...")
        config = create_fallback_config()
    
    # 步骤4：创建环境
    print("\n🌍 步骤4: 创建环境...")
    try:
        env_config = config.get('env', config) if isinstance(config, dict) else create_fallback_config()['env']
        print(f"   📋 环境配置: {env_config}")
        
        env = DoubleIntegratorEnv(env_config)
        print(f"   ✅ 环境创建成功")
        print(f"   🤖 智能体数量: {env.num_agents}")
        print(f"   👁️ 观测维度: {env.observation_shape}")
        print(f"   🎮 动作维度: {env.action_shape}")
        
    except Exception as e:
        print(f"   ❌ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 步骤5：检查策略模型结构
    print("\n🧠 步骤5: 检查策略模型...")
    try:
        policy_state_dict = torch.load(policy_path, map_location='cpu', weights_only=True)
        print(f"   ✅ 策略权重加载成功")
        print(f"   📊 权重层数: {len(policy_state_dict)}")
        
        # 显示关键层的形状
        key_layers = ['perception.mlp.0.weight', 'policy_head.action_mlp.0.weight']
        for layer_name in key_layers:
            if layer_name in policy_state_dict:
                shape = policy_state_dict[layer_name].shape
                print(f"   📐 {layer_name}: {shape}")
            else:
                print(f"   ⚠️ 未找到层: {layer_name}")
        
        # 推断输入维度
        if 'perception.mlp.0.weight' in policy_state_dict:
            input_dim = policy_state_dict['perception.mlp.0.weight'].shape[1]
            print(f"   🎯 推断的输入维度: {input_dim}")
            
            if input_dim != env.observation_shape:
                print(f"   ⚠️ 警告: 模型输入维度({input_dim}) != 环境观测维度({env.observation_shape})")
                print(f"   🔧 需要调整环境配置...")
                
                # 尝试调整环境
                if input_dim == 6:
                    print("   🔧 调整为无障碍物环境 (6维)")
                    env_config['obstacles'] = {'enabled': False}
                elif input_dim == 9:
                    print("   🔧 调整为有障碍物环境 (9维)")
                    env_config['obstacles'] = {'enabled': True, 'count': 2}
                
                # 重新创建环境
                env = DoubleIntegratorEnv(env_config)
                print(f"   ✅ 环境重新创建: 观测维度={env.observation_shape}")
            
    except Exception as e:
        print(f"   ❌ 策略权重检查失败: {e}")
        return False
    
    # 步骤6：创建策略网络
    print("\n🎭 步骤6: 创建策略网络...")
    try:
        policy_config = create_policy_config(env.observation_shape, env.action_shape)
        print(f"   📋 策略配置创建完成")
        
        policy = BPTTPolicy(policy_config)
        print(f"   ✅ 策略网络创建成功")
        
        # 加载权重
        policy.load_state_dict(policy_state_dict)
        policy.eval()
        print(f"   ✅ 策略权重加载成功")
        
    except Exception as e:
        print(f"   ❌ 策略网络创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 步骤7：测试推理
    print("\n🧪 步骤7: 测试模型推理...")
    try:
        # 创建测试状态
        device = torch.device('cpu')
        test_state = create_test_state(env, device)
        
        # 获取观测
        observations = env.get_observations(test_state)
        print(f"   📊 观测形状: {observations.shape}")
        
        # 测试策略推理
        with torch.no_grad():
            policy_output = policy(observations, test_state)
            actions = policy_output.actions
            print(f"   ✅ 策略推理成功")
            print(f"   🎮 动作形状: {actions.shape}")
            print(f"   📏 动作范围: [{torch.min(actions):.4f}, {torch.max(actions):.4f}]")
            
            if hasattr(policy_output, 'alphas'):
                alphas = policy_output.alphas
                print(f"   ⚖️ Alpha形状: {alphas.shape}")
                print(f"   📏 Alpha范围: [{torch.min(alphas):.4f}, {torch.max(alphas):.4f}]")
        
    except Exception as e:
        print(f"   ❌ 模型推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 所有诊断步骤成功完成!")
    print("✅ 真实模型加载和推理验证通过")
    return True

def create_fallback_config():
    """创建备用配置"""
    return {
        'env': {
            'name': 'DoubleIntegrator',
            'num_agents': 6,
            'area_size': 4.0,
            'dt': 0.02,
            'mass': 0.5,
            'agent_radius': 0.15,
            'comm_radius': 1.0,
            'max_force': 0.5,
            'max_steps': 120,
            'social_radius': 0.4,
            'obstacles': {
                'enabled': True,
                'count': 2,
                'positions': [[0, 0.7], [0, -0.7]],
                'radii': [0.3, 0.3]
            }
        }
    }

def create_policy_config(input_dim, output_dim):
    """创建策略配置"""
    return {
        'type': 'bptt',
        'input_dim': input_dim,
        'output_dim': output_dim,
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
            'output_dim': output_dim,
            'predict_alpha': True,
            'hidden_dims': [256, 256],
            'action_scale': 1.0
        },
        'device': torch.device('cpu')
    }

def create_test_state(env, device):
    """创建测试状态"""
    from gcbfplus.env.multi_agent_env import MultiAgentState
    
    num_agents = env.num_agents
    
    positions = torch.zeros(1, num_agents, 2, device=device)
    velocities = torch.zeros(1, num_agents, 2, device=device)
    goals = torch.zeros(1, num_agents, 2, device=device)
    
    # 简单的测试位置
    for i in range(num_agents):
        positions[0, i] = torch.tensor([-1.0, i * 0.3 - 1.0], device=device)
        goals[0, i] = torch.tensor([1.0, i * 0.3 - 1.0], device=device)
    
    return MultiAgentState(
        positions=positions,
        velocities=velocities,
        goals=goals,
        batch_size=1
    )

if __name__ == "__main__":
    print("🔧 真实模型加载调试器")
    print("逐步诊断并解决加载问题")
    print("=" * 80)
    
    success = debug_model_loading()
    
    if success:
        print(f"\n🎉 调试成功!")
        print(f"✅ 真实模型加载和推理验证通过")
        print(f"🚀 准备生成100%真实的可视化")
    else:
        print(f"\n❌ 调试失败")
        print(f"需要进一步解决问题")
 
 
 
 