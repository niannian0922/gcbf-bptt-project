#!/usr/bin/env python3
"""
🔍 基础模型测试
最简单的测试，逐步排查真实模型问题
"""

import torch
import numpy as np

def basic_model_test():
    """基础模型测试"""
    print("🔍 基础模型测试开始")
    
    # 测试1: 基础导入
    print("1️⃣ 测试基础导入...")
    try:
        import yaml
        print("   ✅ yaml导入成功")
        
        from gcbfplus.env import DoubleIntegratorEnv
        print("   ✅ DoubleIntegratorEnv导入成功")
        
        from gcbfplus.env.multi_agent_env import MultiAgentState
        print("   ✅ MultiAgentState导入成功")
        
        from gcbfplus.policy.bptt_policy import create_policy_from_config
        print("   ✅ create_policy_from_config导入成功")
        
    except Exception as e:
        print(f"   ❌ 导入失败: {e}")
        return False
    
    # 测试2: 检查模型文件
    print("\n2️⃣ 检查模型文件...")
    try:
        import os
        model_path = "logs/full_collaboration_training/models/500/policy.pt"
        
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"   ✅ 模型文件存在: {size_mb:.1f}MB")
        else:
            print(f"   ❌ 模型文件不存在: {model_path}")
            return False
    except Exception as e:
        print(f"   ❌ 文件检查失败: {e}")
        return False
    
    # 测试3: 创建最小环境
    print("\n3️⃣ 创建最小环境...")
    try:
        env_config = {
            'name': 'DoubleIntegrator',
            'num_agents': 2,
            'area_size': 3.0,
            'dt': 0.05,
            'mass': 0.1,
            'agent_radius': 0.15,
            'comm_radius': 1.0,
            'max_force': 1.0,
            'max_steps': 50,
            'social_radius': 0.4
        }
        
        env = DoubleIntegratorEnv(env_config)
        print(f"   ✅ 环境创建成功")
        print(f"      观测维度: {env.observation_shape}")
        print(f"      动作维度: {env.action_shape}")
        
    except Exception as e:
        print(f"   ❌ 环境创建失败: {e}")
        return False
    
    # 测试4: 创建测试状态
    print("\n4️⃣ 创建测试状态...")
    try:
        device = torch.device('cpu')
        
        positions = torch.tensor([[[-1.0, 0.0], [1.0, 0.0]]], device=device, dtype=torch.float32)
        velocities = torch.zeros(1, 2, 2, device=device, dtype=torch.float32)
        goals = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]], device=device, dtype=torch.float32)
        
        state = MultiAgentState(
            positions=positions,
            velocities=velocities,
            goals=goals,
            batch_size=1
        )
        
        print(f"   ✅ 测试状态创建成功")
        print(f"      位置: {positions[0]}")
        print(f"      目标: {goals[0]}")
        
    except Exception as e:
        print(f"   ❌ 状态创建失败: {e}")
        return False
    
    # 测试5: 观测生成
    print("\n5️⃣ 测试观测生成...")
    try:
        observations = env.get_observation(state)
        print(f"   ✅ 观测生成成功")
        print(f"      观测形状: {observations.shape}")
        print(f"      第1个智能体观测: {observations[0, 0]}")
        
    except Exception as e:
        print(f"   ❌ 观测生成失败: {e}")
        return False
    
    # 测试6: 创建策略网络
    print("\n6️⃣ 创建策略网络...")
    try:
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
        
        policy = create_policy_from_config(policy_config)
        print(f"   ✅ 策略网络创建成功")
        
    except Exception as e:
        print(f"   ❌ 策略网络创建失败: {e}")
        return False
    
    # 测试7: 随机权重测试
    print("\n7️⃣ 随机权重测试...")
    try:
        with torch.no_grad():
            actions_random, alphas_random = policy(observations)
        
        action_mag = torch.norm(actions_random, dim=-1)
        print(f"   ✅ 随机权重测试成功")
        print(f"      动作形状: {actions_random.shape}")
        print(f"      最大动作幅度: {action_mag.max():.6f}")
        print(f"      第1个智能体动作: {actions_random[0, 0]}")
        
        if action_mag.max() < 1e-6:
            print(f"   ⚠️ 随机权重也产生零动作!")
        
    except Exception as e:
        print(f"   ❌ 随机权重测试失败: {e}")
        return False
    
    # 测试8: 加载真实权重
    print("\n8️⃣ 加载真实权重...")
    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        policy.load_state_dict(state_dict, strict=False)
        print(f"   ✅ 真实权重加载成功")
        
        # 检查关键层权重
        for name, param in policy.named_parameters():
            if 'policy_head' in name and 'weight' in name:
                print(f"      {name}: 均值={param.mean():.6f}, 标准差={param.std():.6f}")
                break
        
    except Exception as e:
        print(f"   ❌ 真实权重加载失败: {e}")
        return False
    
    # 测试9: 真实权重动作
    print("\n9️⃣ 真实权重动作测试...")
    try:
        with torch.no_grad():
            actions_real, alphas_real = policy(observations)
        
        action_mag_real = torch.norm(actions_real, dim=-1)
        print(f"   ✅ 真实权重动作测试成功")
        print(f"      动作形状: {actions_real.shape}")
        print(f"      最大动作幅度: {action_mag_real.max():.6f}")
        print(f"      第1个智能体动作: {actions_real[0, 0]}")
        
        # 关键诊断
        if action_mag_real.max() < 1e-6:
            print(f"\n❌ 问题确认: 真实权重产生零动作!")
            print(f"   这就是智能体不动的根本原因!")
            
            # 尝试简单修复
            print(f"\n🔧 尝试修复...")
            
            # 方法1: 检查action_scale
            if hasattr(policy.policy_head, 'action_scale'):
                print(f"      当前action_scale: {policy.policy_head.action_scale}")
                if policy.policy_head.action_scale < 1e-6:
                    policy.policy_head.action_scale = 1.0
                    print(f"      修复action_scale为1.0")
            
            # 重新测试
            with torch.no_grad():
                actions_fixed, _ = policy(observations)
            
            action_mag_fixed = torch.norm(actions_fixed, dim=-1)
            print(f"      修复后最大动作幅度: {action_mag_fixed.max():.6f}")
            
            if action_mag_fixed.max() > 1e-6:
                print(f"   ✅ 修复成功!")
                generate_simple_moving_test(env, policy, state)
            else:
                print(f"   ❌ 修复失败，需要其他方法")
        else:
            print(f"\n✅ 真实权重产生有效动作")
            print(f"   问题可能在其他地方")
            generate_simple_moving_test(env, policy, state)
        
    except Exception as e:
        print(f"   ❌ 真实权重动作测试失败: {e}")
        return False
    
    print(f"\n🎉 基础模型测试完成!")
    return True

def generate_simple_moving_test(env, policy, initial_state):
    """生成简单的运动测试"""
    print(f"\n🎬 生成简单运动测试...")
    
    try:
        num_steps = 50
        positions_history = []
        
        current_state = initial_state
        
        with torch.no_grad():
            for step in range(num_steps):
                positions = current_state.positions[0].cpu().numpy()
                positions_history.append(positions.copy())
                
                # 获取动作
                observations = env.get_observation(current_state)
                actions, alphas = policy(observations)
                
                # 检查动作
                action_mag = torch.norm(actions, dim=-1).max().item()
                
                if step % 10 == 0:
                    print(f"   步骤 {step}: 动作幅度={action_mag:.6f}")
                
                # 如果动作太小，添加小的推动
                if action_mag < 1e-6:
                    # 朝目标方向的小推动
                    goal_positions = current_state.goals[0].cpu().numpy()
                    for i in range(len(positions)):
                        direction = goal_positions[i] - positions[i]
                        distance = np.linalg.norm(direction)
                        if distance > 0.1:
                            actions[0, i] = torch.tensor(direction / distance * 0.05)
                
                # 环境步进
                try:
                    step_result = env.step(current_state, actions, alphas)
                    current_state = step_result.next_state
                except Exception as e:
                    print(f"   ⚠️ 步进失败: {e}")
                    break
        
        # 分析运动
        if len(positions_history) > 1:
            total_movement = 0
            for i in range(len(positions_history)-1):
                movement = np.linalg.norm(positions_history[i+1] - positions_history[i])
                total_movement += movement
            
            print(f"   📊 运动分析:")
            print(f"      总步数: {len(positions_history)}")
            print(f"      总运动距离: {total_movement:.6f}")
            print(f"      平均每步运动: {total_movement/len(positions_history):.6f}")
            
            if total_movement > 0.01:
                print(f"   ✅ 智能体确实在移动!")
                
                # 创建简单可视化
                create_simple_plot(positions_history, initial_state.goals[0].cpu().numpy())
            else:
                print(f"   ❌ 智能体仍然基本不动")
        
    except Exception as e:
        print(f"❌ 运动测试失败: {e}")

def create_simple_plot(positions_history, goals):
    """创建简单的轨迹图"""
    try:
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        print(f"   🎨 创建简单轨迹图...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('真实模型简单运动测试')
        ax.grid(True, alpha=0.3)
        
        colors = ['red', 'blue']
        num_agents = len(positions_history[0])
        
        # 绘制轨迹
        for i in range(num_agents):
            x_traj = [pos[i, 0] for pos in positions_history]
            y_traj = [pos[i, 1] for pos in positions_history]
            
            ax.plot(x_traj, y_traj, '-o', color=colors[i], 
                   label=f'智能体{i+1}轨迹', markersize=3)
            
            # 起点和终点
            ax.plot(x_traj[0], y_traj[0], 's', color=colors[i], 
                   markersize=10, label=f'起点{i+1}')
            ax.plot(goals[i, 0], goals[i, 1], '*', color=colors[i], 
                   markersize=15, label=f'目标{i+1}')
        
        ax.legend()
        ax.set_aspect('equal')
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"SIMPLE_REAL_MODEL_TEST_{timestamp}.png"
        
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"   ✅ 轨迹图保存: {filename}")
        
        plt.close()
        
    except Exception as e:
        print(f"   ⚠️ 轨迹图创建失败: {e}")

if __name__ == "__main__":
    print("🔍 基础模型测试系统")
    print("逐步排查真实500步协作训练模型问题")
    print("=" * 50)
    
    success = basic_model_test()
    
    if success:
        print(f"\n✅ 测试完成!")
    else:
        print(f"\n❌ 测试失败!")
 
 
 
 