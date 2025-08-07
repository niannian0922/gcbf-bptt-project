#!/usr/bin/env python3
"""
简化的最新模型可视化
绕过复杂的依赖问题
"""

def main():
    print("简化的最新模型可视化")
    print("=" * 40)
    
    # 首先只尝试基础操作
    try:
        import torch
        print("✅ PyTorch可用")
    except:
        print("❌ PyTorch不可用")
        return
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib可用")
    except:
        print("❌ Matplotlib不可用")
        return
    
    try:
        import numpy as np
        print("✅ NumPy可用")
    except:
        print("❌ NumPy不可用")
        return
    
    # 检查模型文件
    import os
    model_path = 'logs/full_collaboration_training/models/500/policy.pt'
    
    if not os.path.exists(model_path):
        print("❌ 最新模型文件不存在")
        return
        
    print(f"✅ 最新模型文件存在 ({os.path.getsize(model_path)/(1024*1024):.1f}MB)")
    
    # 尝试加载模型
    try:
        policy_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        print(f"✅ 模型加载成功 ({len(policy_dict)} 层)")
        
        # 获取输入维度
        if 'perception.mlp.0.weight' in policy_dict:
            input_dim = policy_dict['perception.mlp.0.weight'].shape[1]
            print(f"🎯 模型输入维度: {input_dim}")
        else:
            print("⚠️ 未找到感知层，使用默认配置")
            input_dim = 9
            
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    print("\n🎉 基础检查通过!")
    print("您的最新模型文件完好，可以用于可视化")
    print("\n📋 建议:")
    print("1. 重启Python环境可能解决依赖冲突")
    print("2. 或者使用这个模型路径在其他环境中生成可视化")
    print(f"3. 模型路径: {model_path}")
    print(f"4. 输入维度: {input_dim}")

if __name__ == "__main__":
    main()
 
"""
简化的最新模型可视化
绕过复杂的依赖问题
"""

def main():
    print("简化的最新模型可视化")
    print("=" * 40)
    
    # 首先只尝试基础操作
    try:
        import torch
        print("✅ PyTorch可用")
    except:
        print("❌ PyTorch不可用")
        return
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib可用")
    except:
        print("❌ Matplotlib不可用")
        return
    
    try:
        import numpy as np
        print("✅ NumPy可用")
    except:
        print("❌ NumPy不可用")
        return
    
    # 检查模型文件
    import os
    model_path = 'logs/full_collaboration_training/models/500/policy.pt'
    
    if not os.path.exists(model_path):
        print("❌ 最新模型文件不存在")
        return
        
    print(f"✅ 最新模型文件存在 ({os.path.getsize(model_path)/(1024*1024):.1f}MB)")
    
    # 尝试加载模型
    try:
        policy_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        print(f"✅ 模型加载成功 ({len(policy_dict)} 层)")
        
        # 获取输入维度
        if 'perception.mlp.0.weight' in policy_dict:
            input_dim = policy_dict['perception.mlp.0.weight'].shape[1]
            print(f"🎯 模型输入维度: {input_dim}")
        else:
            print("⚠️ 未找到感知层，使用默认配置")
            input_dim = 9
            
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    print("\n🎉 基础检查通过!")
    print("您的最新模型文件完好，可以用于可视化")
    print("\n📋 建议:")
    print("1. 重启Python环境可能解决依赖冲突")
    print("2. 或者使用这个模型路径在其他环境中生成可视化")
    print(f"3. 模型路径: {model_path}")
    print(f"4. 输入维度: {input_dim}")

if __name__ == "__main__":
    main()
 
 
 
 