#!/usr/bin/env python3
"""
直接模型测试 - 最简化版本
"""
print("开始直接模型测试")

try:
    import torch
    print("✅ torch导入成功")
    
    import os
    model_path = 'logs/full_collaboration_training/models/500/policy.pt'
    print(f"📁 检查文件: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024*1024)
        print(f"📊 文件大小: {size:.1f}MB")
        
        print("🔄 尝试加载权重...")
        policy_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        print(f"✅ 权重加载成功: {len(policy_dict)} 层")
        
        # 查看第一个权重的形状
        first_key = list(policy_dict.keys())[0]
        first_shape = policy_dict[first_key].shape
        print(f"📐 第一层形状: {first_key} -> {first_shape}")
        
        # 查找感知层
        if 'perception.mlp.0.weight' in policy_dict:
            perception_shape = policy_dict['perception.mlp.0.weight'].shape
            print(f"🧠 感知层形状: {perception_shape}")
            print(f"🎯 输入维度: {perception_shape[1]}")
        
        print("🎉 模型测试成功!")
        
    else:
        print("❌ 模型文件不存在")
        
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("测试完成")
 
 
 
 