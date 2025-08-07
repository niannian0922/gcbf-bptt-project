#!/usr/bin/env python3
"""
最小化真实模型测试
逐个测试每个组件
"""

import sys
import os

print("🎯 最小化真实模型测试")
print("=" * 50)
sys.stdout.flush()

# 测试1：基础导入
print("📦 测试1: 基础导入...")
sys.stdout.flush()

try:
    import torch
    print("✅ torch导入成功")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ torch导入失败: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✅ numpy导入成功")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ numpy导入失败: {e}")
    sys.exit(1)

# 测试2：检查模型文件
print("\n📁 测试2: 检查模型文件...")
sys.stdout.flush()

model_path = "logs/full_collaboration_training/models/500/"
policy_path = os.path.join(model_path, "policy.pt")

if os.path.exists(policy_path):
    size = os.path.getsize(policy_path) / (1024 * 1024)  # MB
    print(f"✅ 策略文件存在: {size:.1f}MB")
    sys.stdout.flush()
else:
    print("❌ 策略文件不存在")
    sys.exit(1)

# 测试3：加载策略权重
print("\n📥 测试3: 加载策略权重...")
sys.stdout.flush()

try:
    policy_state_dict = torch.load(policy_path, map_location='cpu', weights_only=True)
    print(f"✅ 策略权重加载成功 ({len(policy_state_dict)} 层)")
    
    # 查看关键层
    if 'perception.mlp.0.weight' in policy_state_dict:
        shape = policy_state_dict['perception.mlp.0.weight'].shape
        print(f"📐 感知层形状: {shape}")
        input_dim = shape[1]
        print(f"🔍 推断输入维度: {input_dim}")
    else:
        print("⚠️ 未找到感知层")
    
    sys.stdout.flush()
    
except Exception as e:
    print(f"❌ 策略权重加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4：尝试导入环境
print("\n🌍 测试4: 导入环境模块...")
sys.stdout.flush()

try:
    from gcbfplus.env import DoubleIntegratorEnv
    print("✅ DoubleIntegratorEnv导入成功")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ DoubleIntegratorEnv导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from gcbfplus.env.multi_agent_env import MultiAgentState
    print("✅ MultiAgentState导入成功")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ MultiAgentState导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5：尝试导入策略
print("\n🧠 测试5: 导入策略模块...")
sys.stdout.flush()

try:
    from gcbfplus.policy.bptt_policy import BPTTPolicy
    print("✅ BPTTPolicy导入成功")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ BPTTPolicy导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 所有基础测试通过!")
print("✅ 准备创建真实模型可视化")
sys.stdout.flush()
 
 
 
 