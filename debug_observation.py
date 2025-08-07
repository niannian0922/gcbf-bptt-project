#!/usr/bin/env python3
"""
調試觀測維度問題
"""

import yaml
import torch
from gcbfplus.env import DoubleIntegratorEnv

def debug_observation():
    """調試觀測維度"""
    print("🔍 調試觀測維度問題")
    print("=" * 50)
    
    # 測試無障礙物配置
    print("1️⃣ 測試無障礙物配置...")
    with open('config/simple_collaboration_pretrain.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"配置中的障礙物設置: {config['env']['obstacles']}")
    
    env = DoubleIntegratorEnv(config['env'])
    state = env.reset()
    obs = env.get_observation(state)
    
    print(f"環境觀測形狀: {obs.shape}")
    print(f"第一個智能體觀測: {obs[0]}")
    print(f"觀測維度數: {obs.shape[-1]}")
    
    # 分析觀測內容
    print("\n🔍 觀測內容分析:")
    first_agent_obs = obs[0]
    print(f"位置 (x, y): {first_agent_obs[:2]}")
    print(f"速度 (vx, vy): {first_agent_obs[2:4]}")
    print(f"目標 (gx, gy): {first_agent_obs[4:6]}")
    if len(first_agent_obs) > 6:
        print(f"額外維度: {first_agent_obs[6:]}")
    
    # 測試有障礙物配置
    print("\n2️⃣ 測試有障礙物配置...")
    with open('config/simple_collaboration.yaml', 'r', encoding='utf-8') as f:
        config_with_obstacles = yaml.safe_load(f)
    
    print(f"配置中的障礙物設置: {config_with_obstacles['env']['obstacles']}")
    
    env_with_obstacles = DoubleIntegratorEnv(config_with_obstacles['env'])
    state_with_obstacles = env_with_obstacles.reset()
    obs_with_obstacles = env_with_obstacles.get_observation(state_with_obstacles)
    
    print(f"有障礙物環境觀測形狀: {obs_with_obstacles.shape}")
    print(f"第一個智能體觀測: {obs_with_obstacles[0]}")
    print(f"觀測維度數: {obs_with_obstacles.shape[-1]}")
    
    # 分析觀測內容
    print("\n🔍 有障礙物觀測內容分析:")
    first_agent_obs_with_obstacles = obs_with_obstacles[0]
    print(f"位置 (x, y): {first_agent_obs_with_obstacles[:2]}")
    print(f"速度 (vx, vy): {first_agent_obs_with_obstacles[2:4]}")
    print(f"目標 (gx, gy): {first_agent_obs_with_obstacles[4:6]}")
    if len(first_agent_obs_with_obstacles) > 6:
        print(f"障礙物相關維度: {first_agent_obs_with_obstacles[6:]}")
    
    print(f"\n📊 總結:")
    print(f"無障礙物環境: {obs.shape[-1]} 維觀測")
    print(f"有障礙物環境: {obs_with_obstacles.shape[-1]} 維觀測")

if __name__ == "__main__":
    debug_observation()
 
"""
調試觀測維度問題
"""

import yaml
import torch
from gcbfplus.env import DoubleIntegratorEnv

def debug_observation():
    """調試觀測維度"""
    print("🔍 調試觀測維度問題")
    print("=" * 50)
    
    # 測試無障礙物配置
    print("1️⃣ 測試無障礙物配置...")
    with open('config/simple_collaboration_pretrain.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"配置中的障礙物設置: {config['env']['obstacles']}")
    
    env = DoubleIntegratorEnv(config['env'])
    state = env.reset()
    obs = env.get_observation(state)
    
    print(f"環境觀測形狀: {obs.shape}")
    print(f"第一個智能體觀測: {obs[0]}")
    print(f"觀測維度數: {obs.shape[-1]}")
    
    # 分析觀測內容
    print("\n🔍 觀測內容分析:")
    first_agent_obs = obs[0]
    print(f"位置 (x, y): {first_agent_obs[:2]}")
    print(f"速度 (vx, vy): {first_agent_obs[2:4]}")
    print(f"目標 (gx, gy): {first_agent_obs[4:6]}")
    if len(first_agent_obs) > 6:
        print(f"額外維度: {first_agent_obs[6:]}")
    
    # 測試有障礙物配置
    print("\n2️⃣ 測試有障礙物配置...")
    with open('config/simple_collaboration.yaml', 'r', encoding='utf-8') as f:
        config_with_obstacles = yaml.safe_load(f)
    
    print(f"配置中的障礙物設置: {config_with_obstacles['env']['obstacles']}")
    
    env_with_obstacles = DoubleIntegratorEnv(config_with_obstacles['env'])
    state_with_obstacles = env_with_obstacles.reset()
    obs_with_obstacles = env_with_obstacles.get_observation(state_with_obstacles)
    
    print(f"有障礙物環境觀測形狀: {obs_with_obstacles.shape}")
    print(f"第一個智能體觀測: {obs_with_obstacles[0]}")
    print(f"觀測維度數: {obs_with_obstacles.shape[-1]}")
    
    # 分析觀測內容
    print("\n🔍 有障礙物觀測內容分析:")
    first_agent_obs_with_obstacles = obs_with_obstacles[0]
    print(f"位置 (x, y): {first_agent_obs_with_obstacles[:2]}")
    print(f"速度 (vx, vy): {first_agent_obs_with_obstacles[2:4]}")
    print(f"目標 (gx, gy): {first_agent_obs_with_obstacles[4:6]}")
    if len(first_agent_obs_with_obstacles) > 6:
        print(f"障礙物相關維度: {first_agent_obs_with_obstacles[6:]}")
    
    print(f"\n📊 總結:")
    print(f"無障礙物環境: {obs.shape[-1]} 維觀測")
    print(f"有障礙物環境: {obs_with_obstacles.shape[-1]} 維觀測")

if __name__ == "__main__":
    debug_observation()
 
 
 
 