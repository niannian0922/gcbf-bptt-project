#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试动态Alpha调试脚本的基本功能
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """测试导入"""
    print("🔧 测试导入...")
    try:
        import torch
        import yaml
        import numpy as np
        from gcbfplus.env import DoubleIntegratorEnv
        from gcbfplus.policy import BPTTPolicy
        print("✅ 所有导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_model_path():
    """测试模型路径"""
    print("\n📂 测试模型路径...")
    model_path = Path("logs/bptt/models/1000")
    
    if model_path.exists():
        print(f"✅ 模型目录存在: {model_path}")
        
        config_file = model_path / "config.yaml"
        policy_file = model_path / "policy.pt"
        
        if config_file.exists():
            print(f"✅ 配置文件存在: {config_file}")
        else:
            print(f"❌ 配置文件缺失: {config_file}")
            
        if policy_file.exists():
            print(f"✅ 策略文件存在: {policy_file}")
        else:
            print(f"❌ 策略文件缺失: {policy_file}")
            
        return config_file.exists() and policy_file.exists()
    else:
        print(f"❌ 模型目录不存在: {model_path}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n🚀 测试基本功能...")
    try:
        # 导入主要的调试类
        from debug_dynamic_alpha import AlphaDebugger
        
        # 检查可用模型
        logs_dir = Path("logs")
        if not logs_dir.exists():
            print("❌ logs目录不存在")
            return False
            
        available_models = []
        for subdir in logs_dir.iterdir():
            if subdir.is_dir():
                models_dir = subdir / "models"
                if models_dir.exists():
                    for model_dir in models_dir.iterdir():
                        if model_dir.is_dir() and (model_dir / "policy.pt").exists():
                            available_models.append(str(model_dir))
        
        print(f"发现 {len(available_models)} 个可用模型:")
        for model in available_models:
            print(f"  - {model}")
            
        if available_models:
            print("✅ 找到可用模型")
            return True
        else:
            print("❌ 未找到可用模型")
            return False
            
    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🧪 动态Alpha调试脚本测试")
    print("=" * 40)
    
    # 运行测试
    tests = [
        ("导入测试", test_imports),
        ("模型路径测试", test_model_path),
        ("基本功能测试", test_basic_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 40)
    print("📊 测试结果总结:")
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n通过率: {passed}/{len(tests)} ({passed/len(tests)*100:.0f}%)")
    
    if passed == len(tests):
        print("🎉 所有测试通过！可以运行完整的调试脚本")
    else:
        print("⚠️  存在问题，需要先解决再运行完整脚本")

if __name__ == "__main__":
    main()