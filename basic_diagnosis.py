#!/usr/bin/env python3
"""
基礎診斷測試
"""

print("🔬 基礎診斷開始")
print("=" * 40)

try:
    print("1. 測試Python基礎...")
    import sys
    print(f"   Python版本: {sys.version}")
    
    print("2. 測試PyTorch...")
    import torch
    print(f"   PyTorch版本: {torch.__version__}")
    
    print("3. 測試YAML...")
    import yaml
    print("   YAML導入成功")
    
    print("4. 測試項目導入...")
    try:
        from gcbfplus.env import DoubleIntegratorEnv
        print("   DoubleIntegratorEnv導入成功")
    except Exception as e:
        print(f"   DoubleIntegratorEnv導入失敗: {e}")
        import traceback
        traceback.print_exc()
        
    try:
        from gcbfplus.policy import create_policy_from_config
        print("   Policy導入成功")
    except Exception as e:
        print(f"   Policy導入失敗: {e}")
        import traceback
        traceback.print_exc()
        
    try:
        from gcbfplus.trainer.bptt_trainer import BPTTTrainer
        print("   BPTTTrainer導入成功")
    except Exception as e:
        print(f"   BPTTTrainer導入失敗: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n5. 測試配置文件...")
    try:
        with open('config/simple_collaboration_pretrain.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("   配置文件加載成功")
        print(f"   智能體數量: {config['env']['num_agents']}")
    except Exception as e:
        print(f"   配置文件加載失敗: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 基礎診斷完成")
    
except Exception as e:
    print(f"❌ 基礎診斷失敗: {e}")
    import traceback
    traceback.print_exc()
 
"""
基礎診斷測試
"""

print("🔬 基礎診斷開始")
print("=" * 40)

try:
    print("1. 測試Python基礎...")
    import sys
    print(f"   Python版本: {sys.version}")
    
    print("2. 測試PyTorch...")
    import torch
    print(f"   PyTorch版本: {torch.__version__}")
    
    print("3. 測試YAML...")
    import yaml
    print("   YAML導入成功")
    
    print("4. 測試項目導入...")
    try:
        from gcbfplus.env import DoubleIntegratorEnv
        print("   DoubleIntegratorEnv導入成功")
    except Exception as e:
        print(f"   DoubleIntegratorEnv導入失敗: {e}")
        import traceback
        traceback.print_exc()
        
    try:
        from gcbfplus.policy import create_policy_from_config
        print("   Policy導入成功")
    except Exception as e:
        print(f"   Policy導入失敗: {e}")
        import traceback
        traceback.print_exc()
        
    try:
        from gcbfplus.trainer.bptt_trainer import BPTTTrainer
        print("   BPTTTrainer導入成功")
    except Exception as e:
        print(f"   BPTTTrainer導入失敗: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n5. 測試配置文件...")
    try:
        with open('config/simple_collaboration_pretrain.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("   配置文件加載成功")
        print(f"   智能體數量: {config['env']['num_agents']}")
    except Exception as e:
        print(f"   配置文件加載失敗: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 基礎診斷完成")
    
except Exception as e:
    print(f"❌ 基礎診斷失敗: {e}")
    import traceback
    traceback.print_exc()
 
 
 
 