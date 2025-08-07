#!/usr/bin/env python3
"""
檢查不同環境配置的觀測維度
"""

import yaml
from gcbfplus.env import DoubleIntegratorEnv

def check_obs_dimensions():
    """檢查觀測維度"""
    print("🔍 檢查觀測維度")
    print("=" * 40)
    
    # 1. 有障礙物環境
    env_config_with_obs = {
        'num_agents': 4,
        'area_size': 2.0,
        'car_radius': 0.1,
        'comm_radius': 0.5,
        'dt': 0.05,
        'obstacles': {
            'enabled': True,
            'count': 2,
            'positions': [[0, 1], [0, -1]],
            'radii': [0.8, 0.8]
        }
    }
    
    env1 = DoubleIntegratorEnv(env_config_with_obs)
    print(f"✅ 有障礙物環境觀測維度: {env1.observation_shape}")
    
    # 2. 無障礙物環境
    env_config_no_obs = {
        'num_agents': 4,
        'area_size': 2.0,
        'car_radius': 0.1,
        'comm_radius': 0.5,
        'dt': 0.05,
        'obstacles': {
            'enabled': False
        }
    }
    
    env2 = DoubleIntegratorEnv(env_config_no_obs)
    print(f"✅ 無障礙物環境觀測維度: {env2.observation_shape}")
    
    # 3. 測試預訓練配置
    with open('config/simple_collaboration_pretrain.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    env3 = DoubleIntegratorEnv(config['env'])
    print(f"✅ 預訓練配置觀測維度: {env3.observation_shape}")

if __name__ == "__main__":
    check_obs_dimensions()
 
"""
檢查不同環境配置的觀測維度
"""

import yaml
from gcbfplus.env import DoubleIntegratorEnv

def check_obs_dimensions():
    """檢查觀測維度"""
    print("🔍 檢查觀測維度")
    print("=" * 40)
    
    # 1. 有障礙物環境
    env_config_with_obs = {
        'num_agents': 4,
        'area_size': 2.0,
        'car_radius': 0.1,
        'comm_radius': 0.5,
        'dt': 0.05,
        'obstacles': {
            'enabled': True,
            'count': 2,
            'positions': [[0, 1], [0, -1]],
            'radii': [0.8, 0.8]
        }
    }
    
    env1 = DoubleIntegratorEnv(env_config_with_obs)
    print(f"✅ 有障礙物環境觀測維度: {env1.observation_shape}")
    
    # 2. 無障礙物環境
    env_config_no_obs = {
        'num_agents': 4,
        'area_size': 2.0,
        'car_radius': 0.1,
        'comm_radius': 0.5,
        'dt': 0.05,
        'obstacles': {
            'enabled': False
        }
    }
    
    env2 = DoubleIntegratorEnv(env_config_no_obs)
    print(f"✅ 無障礙物環境觀測維度: {env2.observation_shape}")
    
    # 3. 測試預訓練配置
    with open('config/simple_collaboration_pretrain.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    env3 = DoubleIntegratorEnv(config['env'])
    print(f"✅ 預訓練配置觀測維度: {env3.observation_shape}")

if __name__ == "__main__":
    check_obs_dimensions()
 
 
 
 