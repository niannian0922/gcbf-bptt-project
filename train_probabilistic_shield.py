#!/usr/bin/env python3
"""
🛡️ PROBABILISTIC SAFETY SHIELD 概率安全防护罩训练脚本

重大架构重构：将GCBF+模块从直接控制输出转换为概率安全防护罩。
这个创新解耦了"安全"和"效率"目标，允许策略网络在安全区域自由探索，
同时在危险情况下提供安全回退保证。

核心创新：
1. GCBF+模块输出安全信心分数 alpha_safety (0-1)
2. 最终动作 = alpha_safety * 策略动作 + (1-alpha_safety) * 安全动作
3. 新的CBF损失函数训练风险评估器，不是约束满足器
"""

import argparse
import torch
import yaml
from pathlib import Path

from gcbfplus.trainer.bptt_trainer import BPTTTrainer
from gcbfplus.env.double_integrator import DoubleIntegratorEnv
from gcbfplus.policy.bptt_policy import BPTTPolicy
from gcbfplus.env.gcbf_safety_layer import GCBFSafetyLayer
import os


def create_environment(config):
    """创建环境"""
    env_config = config.get('env', {})
    return DoubleIntegratorEnv(env_config)


def create_policy_network(config):
    """创建策略网络"""
    return BPTTPolicy(config)


def create_cbf_network(config):
    """创建CBF网络（安全防护罩）"""
    cbf_config = config.get('env', {}).get('safety_layer', {})
    return GCBFSafetyLayer(cbf_config)


def main():
    parser = argparse.ArgumentParser(
        description="🛡️ 训练概率安全防护罩模型"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/probabilistic_safety_shield.yaml",
        help="配置文件路径"
    )
    args = parser.parse_args()
    
    # 加载配置
    print(f"🛡️ 加载概率安全防护罩配置: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")
    
    print("\n" + "="*60)
    print("🛡️ PROBABILISTIC SAFETY SHIELD 概率安全防护罩")
    print("="*60)
    print("核心创新：")
    print("📊 GCBF+模块 → 安全信心分数输出 (0-1)")
    print("🎯 最终动作 = 信心 × 策略动作 + (1-信心) × 安全动作")
    print("🧠 CBF损失 → 风险评估器训练，不是约束满足")
    print("🔄 解耦安全与效率目标")
    print("="*60)
    
    # 创建环境
    print("\n🌍 创建环境...")
    env = create_environment(config)
    print(f"✅ 环境创建完成: {env.num_agents} 智能体")
    
    # 创建网络
    print("🏗️ 创建策略网络...")
    policy_network = create_policy_network(config).to(device)
    print("✅ 策略网络创建完成")
    
    print("🛡️ 创建概率安全防护罩...")
    cbf_network = create_cbf_network(config).to(device)
    print("✅ 概率安全防护罩创建完成")
    
    # 将CBF网络设置到环境的安全层
    env.safety_layer = cbf_network
    env.to(device)
    
    # 初始化训练器
    print("\n🚀 初始化概率安全防护罩训练器...")
    trainer = BPTTTrainer(env, policy_network, cbf_network, config=config)
    
    # 验证安全层配置
    if hasattr(trainer.env, 'safety_layer') and trainer.env.safety_layer is not None:
        print("✅ 安全防护罩已启用")
        print(f"   - 安全锐利度参数 k: {trainer.env.safety_layer.k}")
        print(f"   - 安全裕度: {trainer.env.safety_layer.safety_margin}")
    else:
        print("⚠️  警告：未检测到安全防护罩")
    
    # 验证配置参数
    if config.get('training', {}).get('use_probabilistic_shield', False):
        print("✅ 概率安全防护罩模式已启用")
    else:
        print("⚠️  注意：未明确启用概率防护罩模式")
        
    print(f"\n📋 训练配置:")
    print(f"   - 训练步数: {config['training']['training_steps']}")
    print(f"   - 时域长度: {config['training']['horizon_length']}")
    print(f"   - 安全损失权重: {config['training']['safety_weight']}")
    print(f"   - CBF学习率: {config['training']['cbf_lr']}")
    
    # 开始训练
    print(f"\n🎯 开始概率安全防护罩训练...")
    print("   这将训练GCBF网络成为准确的风险评估器！")
    
    try:
        trainer.train()
        print("\n🎉 概率安全防护罩训练完成！")
        print("🔍 模型现在应该能够：")
        print("   ✓ 在安全区域输出高信心分数 (接近1)")
        print("   ✓ 在危险区域输出低信心分数 (接近0)")
        print("   ✓ 动态混合策略动作和安全动作")
        print("   ✓ 解耦安全与效率的优化目标")
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()
