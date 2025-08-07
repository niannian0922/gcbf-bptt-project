#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的动态Alpha调试脚本 - 专注于核心调试功能

这个脚本实现了您要求的所有关键功能：
1. 加载预训练的黄金基准模型
2. 设置两智能体对撞场景
3. 冻结除alpha_head外的所有参数
4. 逐步训练alpha预测，提供详细输出
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path

def main():
    """主调试函数"""
    print("🚨 动态Alpha调试器 - 简化版")
    print("=" * 50)
    
    # 基本设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 步骤1: 加载黄金基准模型
    print("\n📂 步骤1: 加载黄金基准模型")
    model_path = Path("logs/bptt/models/1000")
    
    if not model_path.exists():
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    # 加载配置
    config_path = model_path / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ 加载配置: {config_path}")
    print(f"   原始训练步数: {config['training']['training_steps']}")
    print(f"   原始安全权重: {config['training']['safety_weight']}")
    
    # 步骤2: 修改配置启用动态alpha
    print("\n🔧 步骤2: 修改配置启用动态alpha")
    
    # 确保策略头配置存在
    if 'policy_head' not in config['networks']['policy']:
        config['networks']['policy']['policy_head'] = {}
    
    # 启用alpha预测
    config['networks']['policy']['policy_head'].update({
        'predict_alpha': True,
        'alpha_hidden_dim': 64,
        'output_dim': 2,
        'input_dim': config['networks']['policy'].get('hidden_dim', 64),
        'hidden_dims': [128, 64],  # 添加隐藏层配置
        'activation': 'relu'
    })
    
    print("✅ 配置已修改为支持动态alpha预测")
    
    # 步骤3: 创建策略网络
    print("\n🧠 步骤3: 创建策略网络")
    
    try:
        from gcbfplus.policy import BPTTPolicy
        policy = BPTTPolicy(config['networks']['policy']).to(device)
        print("✅ 策略网络创建成功")
        
        # 检查是否有alpha网络
        if hasattr(policy.policy_head, 'alpha_network') and policy.policy_head.alpha_network is not None:
            print("✅ Alpha预测网络已存在")
        else:
            print("🔧 添加Alpha预测网络...")
            # 手动添加alpha网络
            input_dim = policy.policy_head.input_dim
            policy.policy_head.alpha_network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softplus()
            ).to(device)
            policy.policy_head.predict_alpha = True
            print(f"✅ 创建Alpha网络: {input_dim} -> 64 -> 1")
        
    except Exception as e:
        print(f"❌ 策略网络创建失败: {e}")
        return
    
    # 步骤4: 加载预训练权重（部分）
    print("\n📥 步骤4: 加载预训练权重")
    
    policy_file = model_path / "policy.pt"
    if policy_file.exists():
        try:
            state_dict = torch.load(policy_file, map_location=device)
            # 只加载兼容的权重，忽略不匹配的
            policy.load_state_dict(state_dict, strict=False)
            print("✅ 部分权重加载成功（忽略不兼容部分）")
        except Exception as e:
            print(f"⚠️  权重加载异常: {e}")
            print("继续使用随机初始化的权重...")
    else:
        print("⚠️  权重文件不存在，使用随机初始化")
    
    # 步骤5: 冻结非alpha参数
    print("\n❄️  步骤5: 冻结非alpha参数")
    
    frozen_params = 0
    trainable_params = 0
    alpha_param_names = []
    
    for name, param in policy.named_parameters():
        if 'alpha_network' in name:
            param.requires_grad = True
            trainable_params += param.numel()
            alpha_param_names.append(name)
            print(f"   🔓 可训练: {name} ({param.numel()} 参数)")
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    print(f"✅ 冻结参数: {frozen_params:,}")
    print(f"✅ 可训练参数: {trainable_params:,}")
    
    if trainable_params == 0:
        print("❌ 没有可训练的alpha参数！")
        return
    
    # 步骤6: 创建优化器
    print("\n⚙️  步骤6: 创建优化器")
    
    alpha_params = [p for n, p in policy.named_parameters() if 'alpha_network' in n and p.requires_grad]
    optimizer = optim.Adam(alpha_params, lr=0.001)
    
    print(f"✅ 优化器创建完成 (Adam, lr=0.001)")
    print(f"   优化参数数量: {len(alpha_params)}")
    
    # 步骤7: 创建简化的两智能体对撞场景
    print("\n🚗💥 步骤7: 创建两智能体对撞场景")
    
    # 模拟简单的观测（两个智能体）
    batch_size = 1
    n_agents = 2
    obs_dim = config['networks']['policy'].get('input_dim', 9)
    
    # 创建对撞场景的观测
    # 智能体1: 位置(-0.8, 0), 速度(0.5, 0)
    # 智能体2: 位置(0.8, 0), 速度(-0.5, 0)
    observations = torch.zeros(batch_size, n_agents, obs_dim, device=device)
    
    # 简化的观测设置（位置和速度）
    if obs_dim >= 4:
        # 智能体1
        observations[0, 0, 0] = -0.8  # x位置
        observations[0, 0, 1] = 0.0   # y位置
        observations[0, 0, 2] = 0.5   # x速度
        observations[0, 0, 3] = 0.0   # y速度
        
        # 智能体2
        observations[0, 1, 0] = 0.8   # x位置
        observations[0, 1, 1] = 0.0   # y位置
        observations[0, 1, 2] = -0.5  # x速度
        observations[0, 1, 3] = 0.0   # y速度
    
    print("✅ 对撞场景设置完成")
    print(f"   智能体1: pos=(-0.8, 0.0), vel=(0.5, 0.0)")
    print(f"   智能体2: pos=(0.8, 0.0), vel=(-0.5, 0.0)")
    print(f"   预计碰撞时间: {1.6/1.0:.1f}秒（无干预情况）")
    
    # 步骤8: 调试训练循环
    print("\n🚀 步骤8: 开始调试训练循环")
    print("=" * 50)
    
    # 存储调试数据
    debug_data = {
        'steps': [],
        'alpha_values': [],
        'safety_losses': [],
        'alpha_reg_losses': [],
        'total_losses': []
    }
    
    max_steps = 100
    
    for step in range(max_steps):
        # 前向传播
        with torch.no_grad():
            policy.eval()  # 感知和记忆部分不训练
        
        # 只让alpha网络处于训练模式
        if hasattr(policy.policy_head, 'alpha_network'):
            policy.policy_head.alpha_network.train()
        
        # 获取动作和预测的alpha
        actions, predicted_alpha = policy(observations)
        
        # 提取alpha值
        if predicted_alpha is not None:
            alpha_mean = predicted_alpha.mean().item()
        else:
            alpha_mean = 1.0  # 默认值
            print(f"   ⚠️  步骤{step}: Alpha预测为None!")
        
        # 计算简化的安全损失
        # 基于智能体间距离的简单安全损失
        pos1 = observations[0, 0, :2]  # 智能体1位置
        pos2 = observations[0, 1, :2]  # 智能体2位置
        distance = torch.norm(pos1 - pos2)
        
        safety_radius = 0.1  # 安全半径
        safety_loss = torch.clamp(safety_radius - distance, min=0).pow(2)
        
        # Alpha正则化损失
        target_alpha = 1.5
        if predicted_alpha is not None:
            alpha_reg_loss = (predicted_alpha.mean() - target_alpha).pow(2) * 0.01
        else:
            alpha_reg_loss = torch.tensor(0.0, device=device)
        
        # 总损失
        total_loss = safety_loss + alpha_reg_loss
        
        # 反向传播（只更新alpha网络）
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 存储调试数据
        debug_data['steps'].append(step)
        debug_data['alpha_values'].append(alpha_mean)
        debug_data['safety_losses'].append(safety_loss.item())
        debug_data['alpha_reg_losses'].append(alpha_reg_loss.item())
        debug_data['total_losses'].append(total_loss.item())
        
        # 打印详细调试信息
        print(f"步骤 {step:3d} | "
              f"预测Alpha: {alpha_mean:.4f} | "
              f"安全损失: {safety_loss.item():.6f} | "
              f"Alpha正则: {alpha_reg_loss.item():.6f} | "
              f"总损失: {total_loss.item():.6f} | "
              f"距离: {distance.item():.3f}m")
        
        # 每20步打印分隔线
        if (step + 1) % 20 == 0:
            print("-" * 50)
    
    # 步骤9: 分析结果
    print("\n📊 调试结果分析:")
    print("=" * 50)
    
    initial_alpha = debug_data['alpha_values'][0]
    final_alpha = debug_data['alpha_values'][-1]
    alpha_change = final_alpha - initial_alpha
    
    avg_safety_loss = np.mean(debug_data['safety_losses'])
    avg_alpha_reg_loss = np.mean(debug_data['alpha_reg_losses'])
    
    print(f"Alpha学习结果:")
    print(f"  初始Alpha: {initial_alpha:.4f}")
    print(f"  最终Alpha: {final_alpha:.4f}")
    print(f"  Alpha变化: {alpha_change:+.4f}")
    
    print(f"\n损失分析:")
    print(f"  平均安全损失: {avg_safety_loss:.6f}")
    print(f"  平均Alpha正则损失: {avg_alpha_reg_loss:.6f}")
    
    # 学习趋势分析
    if len(debug_data['steps']) > 20:
        early_alpha = np.mean(debug_data['alpha_values'][:10])
        late_alpha = np.mean(debug_data['alpha_values'][-10:])
        trend = "上升" if late_alpha > early_alpha else "下降"
        
        print(f"\n学习趋势:")
        print(f"  前10步平均Alpha: {early_alpha:.4f}")
        print(f"  后10步平均Alpha: {late_alpha:.4f}")
        print(f"  整体趋势: Alpha {trend}")
    
    print("\n🎉 调试完成!")
    print("="*50)
    
    # 总结关键发现
    print("🔍 关键发现:")
    if abs(alpha_change) > 0.01:
        print(f"  ✅ Alpha成功学习 (变化: {alpha_change:+.4f})")
    else:
        print(f"  ⚠️  Alpha变化很小 (变化: {alpha_change:+.4f})")
        
    if avg_safety_loss > 0.001:
        print(f"  ⚠️  安全损失较高 ({avg_safety_loss:.6f})")
    else:
        print(f"  ✅ 安全损失较低 ({avg_safety_loss:.6f})")
    
    print("\n💡 接下来可以:")
    print("  1. 调整学习率和训练步数")
    print("  2. 修改安全损失函数")
    print("  3. 调整Alpha正则化权重")
    print("  4. 测试不同的初始场景")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断调试")
    except Exception as e:
        print(f"\n❌ 调试过程中出现异常: {e}")
        import traceback
        traceback.print_exc()