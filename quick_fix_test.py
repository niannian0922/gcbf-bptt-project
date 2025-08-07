#!/usr/bin/env python3
"""
快速修復測試
"""

import torch
import yaml

print("🔧 快速修復測試")
print("=" * 30)

try:
    # 測試修復後的配置
    with open('config/simple_collaboration_pretrain.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    from gcbfplus.env import DoubleIntegratorEnv
    env = DoubleIntegratorEnv(config['env'])
    
    state = env.reset()
    obs = env.get_observation(state)
    
    print(f"✅ 環境觀測形狀: {obs.shape}")
    
    expected_input = config['networks']['policy']['perception']['input_dim']
    actual_input = obs.shape[-1]
    
    print(f"✅ 策略期望輸入: {expected_input}")
    print(f"✅ 實際觀測維度: {actual_input}")
    
    if expected_input == actual_input:
        print("🎉 維度匹配修復成功！")
        
        # 測試策略創建
        from gcbfplus.policy import create_policy_from_config
        policy = create_policy_from_config(config['networks']['policy'])
        
        # 測試前向傳播
        batch_obs = obs.unsqueeze(0)
        with torch.no_grad():
            actions, alpha = policy(batch_obs)
            print(f"✅ 前向傳播成功: {actions.shape}")
        
        print("🚀 修復完成，可以運行完整實驗！")
        
    else:
        print(f"❌ 維度仍不匹配: {expected_input} vs {actual_input}")
        
except Exception as e:
    print(f"❌ 測試失敗: {e}")
    import traceback
    traceback.print_exc()
 
"""
快速修復測試
"""

import torch
import yaml

print("🔧 快速修復測試")
print("=" * 30)

try:
    # 測試修復後的配置
    with open('config/simple_collaboration_pretrain.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    from gcbfplus.env import DoubleIntegratorEnv
    env = DoubleIntegratorEnv(config['env'])
    
    state = env.reset()
    obs = env.get_observation(state)
    
    print(f"✅ 環境觀測形狀: {obs.shape}")
    
    expected_input = config['networks']['policy']['perception']['input_dim']
    actual_input = obs.shape[-1]
    
    print(f"✅ 策略期望輸入: {expected_input}")
    print(f"✅ 實際觀測維度: {actual_input}")
    
    if expected_input == actual_input:
        print("🎉 維度匹配修復成功！")
        
        # 測試策略創建
        from gcbfplus.policy import create_policy_from_config
        policy = create_policy_from_config(config['networks']['policy'])
        
        # 測試前向傳播
        batch_obs = obs.unsqueeze(0)
        with torch.no_grad():
            actions, alpha = policy(batch_obs)
            print(f"✅ 前向傳播成功: {actions.shape}")
        
        print("🚀 修復完成，可以運行完整實驗！")
        
    else:
        print(f"❌ 維度仍不匹配: {expected_input} vs {actual_input}")
        
except Exception as e:
    print(f"❌ 測試失敗: {e}")
    import traceback
    traceback.print_exc()
 
 
 
 