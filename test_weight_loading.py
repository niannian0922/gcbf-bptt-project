#!/usr/bin/env python3
"""
专门测试权重加载
"""

import torch
import os
import sys

print("🔍 权重加载专项测试")
print("=" * 40)
sys.stdout.flush()

model_path = "logs/full_collaboration_training/models/500/"
policy_path = os.path.join(model_path, "policy.pt")

print(f"📁 策略文件路径: {policy_path}")
print(f"📁 文件存在: {os.path.exists(policy_path)}")
sys.stdout.flush()

if os.path.exists(policy_path):
    size = os.path.getsize(policy_path) / (1024 * 1024)
    print(f"📊 文件大小: {size:.2f}MB")
    sys.stdout.flush()

print("🔄 开始加载权重...")
sys.stdout.flush()

try:
    # 尝试加载
    print("   步骤1: 调用torch.load...")
    sys.stdout.flush()
    
    policy_state_dict = torch.load(policy_path, map_location='cpu', weights_only=True)
    
    print("   步骤2: 加载完成")
    sys.stdout.flush()
    
    print(f"   📊 权重字典大小: {len(policy_state_dict)}")
    sys.stdout.flush()
    
    print("   步骤3: 列出前几个键...")
    sys.stdout.flush()
    
    keys = list(policy_state_dict.keys())[:5]
    for key in keys:
        print(f"      {key}: {policy_state_dict[key].shape}")
        sys.stdout.flush()
    
    print("   步骤4: 查找感知层...")
    sys.stdout.flush()
    
    if 'perception.mlp.0.weight' in policy_state_dict:
        shape = policy_state_dict['perception.mlp.0.weight'].shape
        print(f"   📐 感知层形状: {shape}")
        print(f"   🎯 输入维度: {shape[1]}")
        sys.stdout.flush()
    else:
        print("   ⚠️ 未找到感知层")
        print("   🔍 查找所有包含'weight'的键:")
        for key in policy_state_dict.keys():
            if 'weight' in key:
                print(f"      {key}: {policy_state_dict[key].shape}")
        sys.stdout.flush()
    
    print("✅ 权重加载测试成功")
    sys.stdout.flush()

except Exception as e:
    print(f"❌ 权重加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()

print("🎉 权重加载测试完成")
sys.stdout.flush()
 
 
 
 