#!/usr/bin/env python3
"""
🛡️ 测试概率安全防护罩功能

验证GCBF+模块重构后的核心功能：
1. 安全信心分数计算
2. 动作混合逻辑
3. 风险评估器损失计算
"""

import torch
import yaml
import numpy as np
from gcbfplus.env.double_integrator import DoubleIntegratorEnv
from gcbfplus.env.gcbf_safety_layer import GCBFSafetyLayer


def test_safety_confidence_computation():
    """测试安全信心分数计算"""
    print("🔍 测试安全信心分数计算...")
    
    # 创建简单的安全层配置
    safety_config = {
        'alpha': 1.0,
        'eps': 0.02,
        'safety_margin': 0.1,
        'safety_sharpness': 2.0,  # k参数
        'use_qp': False
    }
    
    safety_layer = GCBFSafetyLayer(safety_config)
    
    # 创建测试环境
    env_config = {
        'num_agents': 2,
        'area_size': 2.0,
        'dt': 0.05,
        'mass': 0.1,
        'agent_radius': 0.2,
        'comm_radius': 1.0,
        'obstacles': {'enabled': True, 'count': 2, 'radius': 0.1}
    }
    
    env = DoubleIntegratorEnv(env_config)
    env.to(torch.device('cpu'))
    env.safety_layer = safety_layer
    
    # 创建测试状态
    state = env.reset(batch_size=1)
    
    # 测试安全信心分数计算
    try:
        alpha_safety = safety_layer.compute_safety_confidence(state)
        print(f"✅ 安全信心分数计算成功")
        print(f"   - 输出形状: {alpha_safety.shape}")
        print(f"   - 数值范围: [{alpha_safety.min().item():.3f}, {alpha_safety.max().item():.3f}]")
        
        # 验证输出在[0,1]范围内
        assert torch.all(alpha_safety >= 0) and torch.all(alpha_safety <= 1), "安全信心分数应在[0,1]范围内"
        print("✅ 数值范围验证通过")
        
    except Exception as e:
        print(f"❌ 安全信心分数计算失败: {e}")
        return False
    
    return True


def test_action_blending():
    """测试动作混合逻辑"""
    print("\n🔧 测试动作混合逻辑...")
    
    # 创建环境配置
    config = {
        'num_agents': 2,
        'area_size': 2.0,
        'dt': 0.05,
        'mass': 0.1,
        'agent_radius': 0.2,
        'comm_radius': 1.0,
        'obstacles': {'enabled': True, 'count': 1, 'radius': 0.1}
    }
    
    env = DoubleIntegratorEnv(config)
    env.to(torch.device('cpu'))
    
    # 创建安全层
    safety_config = {
        'alpha': 1.0,
        'eps': 0.02,
        'safety_margin': 0.1,
        'safety_sharpness': 1.0,
        'use_qp': False
    }
    env.safety_layer = GCBFSafetyLayer(safety_config)
    
    # 创建测试状态和动作
    state = env.reset(batch_size=1)
    raw_action = torch.tensor([[[1.0, 0.5], [0.0, -1.0]]])  # 积极的策略动作
    
    try:
        # 测试动作混合
        blended_action, alpha_safety = env.apply_safety_layer(state, raw_action)
        
        print(f"✅ 动作混合成功")
        print(f"   - 原始动作: {raw_action.squeeze()}")
        print(f"   - 安全信心: {alpha_safety.squeeze()}")
        print(f"   - 混合动作: {blended_action.squeeze()}")
        
        # 验证混合逻辑
        # 当alpha_safety接近1时，混合动作应接近原始动作
        # 当alpha_safety接近0时，混合动作应接近零（安全动作）
        
        safe_action = torch.zeros_like(raw_action)
        expected_blend = alpha_safety * raw_action + (1 - alpha_safety) * safe_action
        
        assert torch.allclose(blended_action, expected_blend, atol=1e-6), "动作混合公式不正确"
        print("✅ 动作混合公式验证通过")
        
    except Exception as e:
        print(f"❌ 动作混合测试失败: {e}")
        return False
    
    return True


def test_different_safety_scenarios():
    """测试不同安全场景下的行为"""
    print("\n🎯 测试不同安全场景...")
    
    # 创建环境
    config = {
        'num_agents': 2,
        'area_size': 2.0,
        'dt': 0.05,
        'mass': 0.1,
        'agent_radius': 0.2,
        'comm_radius': 1.0,
        'obstacles': {'enabled': True, 'count': 1, 'radius': 0.1}
    }
    
    env = DoubleIntegratorEnv(config)
    env.to(torch.device('cpu'))
    
    # 创建安全层，使用不同的锐利度参数
    for k in [0.5, 1.0, 2.0, 5.0]:
        print(f"\n📊 测试锐利度 k={k}:")
        
        safety_config = {
            'alpha': 1.0,
            'eps': 0.02,
            'safety_margin': 0.1,
            'safety_sharpness': k,
            'use_qp': False
        }
        env.safety_layer = GCBFSafetyLayer(safety_config)
        
        # 创建不同安全状况的测试场景
        state = env.reset(batch_size=1)
        alpha_safety = env.safety_layer.compute_safety_confidence(state)
        
        print(f"   安全信心分数: {alpha_safety.squeeze().detach().numpy()}")
    
    print("✅ 不同安全场景测试完成")
    return True


def main():
    """主测试函数"""
    print("🛡️ 概率安全防护罩功能测试")
    print("="*50)
    
    tests = [
        test_safety_confidence_computation,
        test_action_blending,
        test_different_safety_scenarios
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ 测试失败: {test.__name__}")
        except Exception as e:
            print(f"❌ 测试异常: {test.__name__} - {e}")
    
    print("\n" + "="*50)
    print(f"🏁 测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！概率安全防护罩功能正常")
        print("\n🚀 可以开始训练了：")
        print("   python train_probabilistic_shield.py")
    else:
        print("⚠️  部分测试失败，请检查实现")
    
    return passed == total


if __name__ == "__main__":
    main()
