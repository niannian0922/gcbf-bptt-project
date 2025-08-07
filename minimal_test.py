#!/usr/bin/env python3
"""
最小化测试脚本
"""

print("开始执行...")

try:
    import torch
    print("1. torch导入成功")
    
    import yaml
    print("2. yaml导入成功")
    
    from pathlib import Path
    print("3. pathlib导入成功")
    
    # 测试模型路径
    model_path = Path("logs/bptt/models/1000")
    print(f"4. 模型路径: {model_path}")
    print(f"   存在: {model_path.exists()}")
    
    if model_path.exists():
        config_path = model_path / "config.yaml"
        print(f"5. 配置路径: {config_path}")
        print(f"   存在: {config_path.exists()}")
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("6. 配置加载成功")
            print(f"   训练步数: {config.get('training', {}).get('training_steps', 'N/A')}")
    
    # 测试策略导入
    print("7. 尝试导入策略...")
    from gcbfplus.policy import BPTTPolicy
    print("8. 策略导入成功")
    
    print("✅ 所有基本测试通过")
    
except Exception as e:
    print(f"❌ 异常: {e}")
    import traceback
    traceback.print_exc()

print("测试完成")