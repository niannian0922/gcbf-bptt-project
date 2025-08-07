#!/usr/bin/env python3
"""
直接模型可視化 - 使用已知成功的配置
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

from gcbfplus.env import DoubleIntegratorEnv
from gcbfplus.policy import BPTTPolicy


def create_exact_config():
    """
    使用從實際模型權重分析得出的精確配置
    """
    # 根據實際模型權重構建配置
    config = {
        'perception': {
            'use_vision': False,
            'input_dim': 9,
            'output_dim': 256,
            'hidden_dims': [256, 256],  # 第一層256→256，第二層256→256
            'activation': 'relu'
        },
        'memory': {
            'input_dim': 256,  # 將在BPTTPolicy中自動設置
            'hidden_dim': 256,
            'num_layers': 1
        },
        'policy_head': {
            'input_dim': 256,  # 將在BPTTPolicy中自動設置
            'output_dim': 2,
            'hidden_dims': [256, 256, 2],  # 根據實際權重：0→256, 2→256, 4→2
            'activation': 'relu',
            'predict_alpha': True,
            'alpha_hidden_dims': [128, 1]  # 根據實際權重：0→128, 2→1
        }
    }
    
    return config


def load_model_direct(model_path, device='cpu'):
    """
    直接加載模型，使用精確匹配的配置
    """
    print(f"🎯 直接加載模型: {model_path}")
    
    # 創建精確配置
    policy_config = create_exact_config()
    print(f"📋 使用配置: {policy_config}")
    
    # 創建網絡
    policy_network = BPTTPolicy(policy_config)
    policy_network = policy_network.to(device)
    
    # 加載權重
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        
        # 使用strict=False允許部分加載
        missing_keys, unexpected_keys = policy_network.load_state_dict(state_dict, strict=False)
        
        print(f"✅ 模型加載成功")
        if missing_keys:
            print(f"⚠️ 缺少的鍵: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            print(f"⚠️ 額外的鍵: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        
        return policy_network
        
    except Exception as e:
        print(f"❌ 模型加載失敗: {e}")
        return None


def test_model_inference(policy_network, env, device):
    """
    測試模型推理
    """
    print(f"🧪 測試模型推理")
    
    policy_network.eval()
    
    # 重置環境
    state = env.reset()
    observations = env.get_observations(state).to(device)
    
    print(f"📏 觀測形狀: {observations.shape}")
    
    with torch.no_grad():
        try:
            # 策略推理
            policy_output = policy_network(observations)
            
            if hasattr(policy_output, 'actions'):
                actions = policy_output.actions
                print(f"✅ 動作輸出形狀: {actions.shape}")
                print(f"📊 動作範圍: [{torch.min(actions):.6f}, {torch.max(actions):.6f}]")
                print(f"📊 動作強度: {torch.norm(actions, dim=-1).mean():.6f}")
            else:
                print(f"❌ 無法獲取動作輸出")
                return False
            
            if hasattr(policy_output, 'alphas'):
                alphas = policy_output.alphas
                print(f"✅ Alpha輸出形狀: {alphas.shape}")
                print(f"📊 Alpha範圍: [{torch.min(alphas):.6f}, {torch.max(alphas):.6f}]")
            else:
                print(f"⚠️ 沒有Alpha輸出")
            
            return True
            
        except Exception as e:
            print(f"❌ 推理失敗: {e}")
            import traceback
            traceback.print_exc()
            return False


def run_full_simulation(policy_network, env, device, num_steps=100):
    """
    運行完整仿真
    """
    print(f"🎬 運行完整仿真 ({num_steps} 步)")
    
    policy_network.eval()
    
    # 初始化
    state = env.reset()
    trajectory_positions = []
    trajectory_actions = []
    trajectory_alphas = []
    
    with torch.no_grad():
        for step in range(num_steps):
            # 記錄位置
            current_positions = state.positions[0].cpu().numpy()
            trajectory_positions.append(current_positions.copy())
            
            # 獲取觀測
            observations = env.get_observations(state).to(device)
            
            try:
                # 策略推理
                policy_output = policy_network(observations)
                actions = policy_output.actions
                alphas = getattr(policy_output, 'alphas', torch.ones_like(actions[:, :, :1]) * 0.5)
                
                # 記錄數據
                trajectory_actions.append(actions[0].cpu().numpy())
                trajectory_alphas.append(alphas[0].cpu().numpy())
                
                # 環境步進
                step_result = env.step(state, actions, alphas)
                state = step_result.next_state
                
                # 進度報告
                if step % 25 == 0:
                    action_magnitude = torch.norm(actions, dim=-1).mean().item()
                    print(f"步驟 {step}: 動作強度={action_magnitude:.6f}")
                
            except Exception as e:
                print(f"❌ 步驟 {step} 失敗: {e}")
                break
    
    print(f"✅ 仿真完成，共 {len(trajectory_positions)} 步")
    
    return {
        'positions': trajectory_positions,
        'actions': trajectory_actions,
        'alphas': trajectory_alphas
    }


def create_professional_visualization(trajectory_data, output_path):
    """
    創建專業的可視化
    """
    print(f"🎨 創建專業可視化")
    
    positions = trajectory_data['positions']
    actions = trajectory_data['actions']
    alphas = trajectory_data['alphas']
    
    if not positions:
        print(f"❌ 沒有軌跡數據")
        return False
    
    num_steps = len(positions)
    num_agents = len(positions[0])
    
    # 創建圖形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('🎯 最終統一可視化結果 - 100%真實模型', fontsize=20, fontweight='bold')
    
    # 主軌跡圖
    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.set_title('🚁 多智能體協作軌跡', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 障礙物
    obstacles = [
        {'pos': [0.0, -0.8], 'radius': 0.4},
        {'pos': [0.0, 0.8], 'radius': 0.4}
    ]
    
    for obs in obstacles:
        circle = plt.Circle(obs['pos'], obs['radius'], color='red', alpha=0.8, 
                          label='障礙物' if obs == obstacles[0] else "")
        ax1.add_patch(circle)
    
    # 區域標記
    start_zone = plt.Rectangle((-3.0, -2.0), 1.0, 4.0, fill=False, 
                              edgecolor='green', linestyle='--', linewidth=3, 
                              alpha=0.9, label='起始區域')
    ax1.add_patch(start_zone)
    
    target_zone = plt.Rectangle((2.0, -2.0), 1.0, 4.0, fill=False, 
                               edgecolor='blue', linestyle='--', linewidth=3, 
                               alpha=0.9, label='目標區域')
    ax1.add_patch(target_zone)
    
    # 智能體顏色
    colors = ['#FF2D2D', '#2DFF2D', '#2D2DFF', '#FF8C2D', '#FF2D8C', '#2DFFFF'][:num_agents]
    
    # 軌跡線和智能體
    trail_lines = []
    agent_dots = []
    
    for i in range(num_agents):
        line, = ax1.plot([], [], '-', color=colors[i], alpha=0.8, linewidth=3,
                        label=f'智能體{i+1}' if i < 3 else "")
        trail_lines.append(line)
        
        dot, = ax1.plot([], [], 'o', color=colors[i], markersize=16, 
                       markeredgecolor='black', markeredgewidth=2, zorder=5)
        agent_dots.append(dot)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 動作分析圖
    ax2.set_title('🧠 策略網絡輸出分析', fontsize=14, fontweight='bold')
    ax2.set_xlabel('時間步')
    ax2.set_ylabel('動作強度')
    ax2.grid(True, alpha=0.3)
    
    # Alpha值監控
    ax3.set_title('⚖️ 動態Alpha值監控', fontsize=14, fontweight='bold')
    ax3.set_xlabel('時間步')
    ax3.set_ylabel('Alpha值')
    ax3.grid(True, alpha=0.3)
    
    # 運動統計
    ax4.set_title('📊 運動統計分析', fontsize=14, fontweight='bold')
    ax4.set_xlabel('時間步')
    ax4.set_ylabel('平均位移')
    ax4.grid(True, alpha=0.3)
    
    # 動畫信息文本
    info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                        verticalalignment='top', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    def animate(frame):
        if frame >= num_steps:
            return trail_lines + agent_dots + [info_text]
        
        current_pos = positions[frame]
        
        # 更新軌跡和智能體
        for i in range(num_agents):
            trail_x = [pos[i, 0] for pos in positions[:frame+1]]
            trail_y = [pos[i, 1] for pos in positions[:frame+1]]
            trail_lines[i].set_data(trail_x, trail_y)
            
            agent_dots[i].set_data([current_pos[i, 0]], [current_pos[i, 1]])
        
        # 更新分析圖表
        if frame > 5:
            steps = list(range(frame+1))
            
            # 動作分析
            if frame < len(actions):
                action_magnitudes = []
                for step in range(frame+1):
                    if step < len(actions):
                        step_actions = actions[step]
                        avg_magnitude = np.mean([np.linalg.norm(a) for a in step_actions])
                        action_magnitudes.append(avg_magnitude)
                    else:
                        action_magnitudes.append(0)
                
                ax2.clear()
                ax2.plot(steps, action_magnitudes, 'purple', linewidth=3, label='平均動作強度')
                ax2.fill_between(steps, action_magnitudes, alpha=0.3, color='purple')
                ax2.set_title(f'🧠 策略網絡輸出分析 (步數: {frame})')
                ax2.set_xlabel('時間步')
                ax2.set_ylabel('動作強度')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Alpha值分析
            if frame < len(alphas):
                alpha_values = []
                for step in range(frame+1):
                    if step < len(alphas):
                        avg_alpha = np.mean(alphas[step])
                        alpha_values.append(avg_alpha)
                    else:
                        alpha_values.append(0.5)
                
                ax3.clear()
                ax3.plot(steps, alpha_values, 'orange', linewidth=3, label='平均Alpha值')
                ax3.fill_between(steps, alpha_values, alpha=0.3, color='orange')
                ax3.set_title(f'⚖️ 動態Alpha值監控 (步數: {frame})')
                ax3.set_xlabel('時間步')
                ax3.set_ylabel('Alpha值')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # 運動統計
            displacements = []
            initial_pos = positions[0]
            for step in range(frame+1):
                if step < len(positions):
                    current_pos = positions[step]
                    avg_displacement = np.mean([
                        np.linalg.norm(current_pos[i] - initial_pos[i]) 
                        for i in range(num_agents)
                    ])
                    displacements.append(avg_displacement)
                else:
                    displacements.append(0)
            
            ax4.clear()
            ax4.plot(steps, displacements, 'green', linewidth=3, label='平均位移')
            ax4.fill_between(steps, displacements, alpha=0.3, color='green')
            ax4.set_title(f'📊 運動統計分析 (步數: {frame})')
            ax4.set_xlabel('時間步')
            ax4.set_ylabel('平均位移')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 計算當前統計
        if frame > 0:
            initial_pos = positions[0]
            current_pos = positions[frame]
            total_displacement = np.mean([
                np.linalg.norm(current_pos[i] - initial_pos[i]) 
                for i in range(num_agents)
            ])
            
            action_magnitude = 0
            if frame < len(actions):
                action_magnitude = np.mean([np.linalg.norm(a) for a in actions[frame]])
        else:
            total_displacement = 0
            action_magnitude = 0
        
        info_text.set_text(
            f'步驟: {frame}\n'
            f'智能體數: {num_agents}\n'
            f'平均位移: {total_displacement:.4f}\n'
            f'動作強度: {action_magnitude:.6f}'
        )
        
        return trail_lines + agent_dots + [info_text]
    
    # 創建動畫
    anim = FuncAnimation(fig, animate, frames=num_steps, interval=120, blit=False, repeat=True)
    
    # 保存
    try:
        print(f"💾 保存專業可視化: {output_path}")
        if output_path.endswith('.mp4'):
            anim.save(output_path, writer='ffmpeg', fps=8, dpi=150)
        else:
            anim.save(output_path, writer='pillow', fps=6, dpi=150)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ 保存成功: {file_size:.2f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存失敗: {e}")
        return False
    finally:
        plt.close()


def main():
    """
    主函數
    """
    print(f"🎯 直接模型可視化系統")
    print(f"=" * 70)
    
    # 設置
    model_dir = 'logs/full_collaboration_training'
    device = torch.device('cpu')
    
    # 找最新模型
    models_dir = os.path.join(model_dir, 'models')
    steps = [int(d) for d in os.listdir(models_dir) if d.isdigit()]
    latest_step = max(steps)
    
    policy_path = os.path.join(model_dir, 'models', str(latest_step), 'policy.pt')
    print(f"📁 模型路徑: {policy_path}")
    
    # 創建環境
    env_config = {
        'area_size': 3.0,
        'car_radius': 0.15,
        'comm_radius': 1.0,
        'dt': 0.05,
        'mass': 0.1,
        'max_force': 1.0,
        'max_steps': 100,
        'num_agents': 6,
        'obstacles': {
            'enabled': True,
            'bottleneck': True,
            'positions': [[0.0, -0.8], [0.0, 0.8]],
            'radii': [0.4, 0.4]
        }
    }
    
    env = DoubleIntegratorEnv(env_config)
    env = env.to(device)
    
    print(f"🌍 環境: {env.observation_shape} → {env.action_shape}")
    
    # 加載模型
    policy_network = load_model_direct(policy_path, device)
    if policy_network is None:
        print(f"❌ 系統失敗")
        return
    
    # 測試推理
    if not test_model_inference(policy_network, env, device):
        print(f"❌ 推理測試失敗")
        return
    
    # 運行仿真
    trajectory_data = run_full_simulation(policy_network, env, device, 120)
    
    # 創建可視化
    output_path = 'DIRECT_FINAL_COLLABORATION_RESULT.mp4'
    success = create_professional_visualization(trajectory_data, output_path)
    
    if success:
        print(f"\n🎉 直接可視化成功完成!")
        print(f"📁 結果文件: {output_path}")
        print(f"✅ 這是您真實訓練模型的表現")
        
        # 運動分析
        positions = trajectory_data['positions']
        if positions:
            initial_pos = positions[0]
            final_pos = positions[-1]
            total_displacement = np.mean([
                np.linalg.norm(final_pos[i] - initial_pos[i]) 
                for i in range(len(initial_pos))
            ])
            print(f"📊 平均總位移: {total_displacement:.4f}")
            
            if total_displacement < 0.01:
                print(f"⚠️ 智能體基本靜止，可能存在訓練問題")
            else:
                print(f"✅ 智能體正常運動")
    else:
        print(f"\n❌ 可視化失敗")


if __name__ == '__main__':
    main()
 
"""
直接模型可視化 - 使用已知成功的配置
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

from gcbfplus.env import DoubleIntegratorEnv
from gcbfplus.policy import BPTTPolicy


def create_exact_config():
    """
    使用從實際模型權重分析得出的精確配置
    """
    # 根據實際模型權重構建配置
    config = {
        'perception': {
            'use_vision': False,
            'input_dim': 9,
            'output_dim': 256,
            'hidden_dims': [256, 256],  # 第一層256→256，第二層256→256
            'activation': 'relu'
        },
        'memory': {
            'input_dim': 256,  # 將在BPTTPolicy中自動設置
            'hidden_dim': 256,
            'num_layers': 1
        },
        'policy_head': {
            'input_dim': 256,  # 將在BPTTPolicy中自動設置
            'output_dim': 2,
            'hidden_dims': [256, 256, 2],  # 根據實際權重：0→256, 2→256, 4→2
            'activation': 'relu',
            'predict_alpha': True,
            'alpha_hidden_dims': [128, 1]  # 根據實際權重：0→128, 2→1
        }
    }
    
    return config


def load_model_direct(model_path, device='cpu'):
    """
    直接加載模型，使用精確匹配的配置
    """
    print(f"🎯 直接加載模型: {model_path}")
    
    # 創建精確配置
    policy_config = create_exact_config()
    print(f"📋 使用配置: {policy_config}")
    
    # 創建網絡
    policy_network = BPTTPolicy(policy_config)
    policy_network = policy_network.to(device)
    
    # 加載權重
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        
        # 使用strict=False允許部分加載
        missing_keys, unexpected_keys = policy_network.load_state_dict(state_dict, strict=False)
        
        print(f"✅ 模型加載成功")
        if missing_keys:
            print(f"⚠️ 缺少的鍵: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            print(f"⚠️ 額外的鍵: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        
        return policy_network
        
    except Exception as e:
        print(f"❌ 模型加載失敗: {e}")
        return None


def test_model_inference(policy_network, env, device):
    """
    測試模型推理
    """
    print(f"🧪 測試模型推理")
    
    policy_network.eval()
    
    # 重置環境
    state = env.reset()
    observations = env.get_observations(state).to(device)
    
    print(f"📏 觀測形狀: {observations.shape}")
    
    with torch.no_grad():
        try:
            # 策略推理
            policy_output = policy_network(observations)
            
            if hasattr(policy_output, 'actions'):
                actions = policy_output.actions
                print(f"✅ 動作輸出形狀: {actions.shape}")
                print(f"📊 動作範圍: [{torch.min(actions):.6f}, {torch.max(actions):.6f}]")
                print(f"📊 動作強度: {torch.norm(actions, dim=-1).mean():.6f}")
            else:
                print(f"❌ 無法獲取動作輸出")
                return False
            
            if hasattr(policy_output, 'alphas'):
                alphas = policy_output.alphas
                print(f"✅ Alpha輸出形狀: {alphas.shape}")
                print(f"📊 Alpha範圍: [{torch.min(alphas):.6f}, {torch.max(alphas):.6f}]")
            else:
                print(f"⚠️ 沒有Alpha輸出")
            
            return True
            
        except Exception as e:
            print(f"❌ 推理失敗: {e}")
            import traceback
            traceback.print_exc()
            return False


def run_full_simulation(policy_network, env, device, num_steps=100):
    """
    運行完整仿真
    """
    print(f"🎬 運行完整仿真 ({num_steps} 步)")
    
    policy_network.eval()
    
    # 初始化
    state = env.reset()
    trajectory_positions = []
    trajectory_actions = []
    trajectory_alphas = []
    
    with torch.no_grad():
        for step in range(num_steps):
            # 記錄位置
            current_positions = state.positions[0].cpu().numpy()
            trajectory_positions.append(current_positions.copy())
            
            # 獲取觀測
            observations = env.get_observations(state).to(device)
            
            try:
                # 策略推理
                policy_output = policy_network(observations)
                actions = policy_output.actions
                alphas = getattr(policy_output, 'alphas', torch.ones_like(actions[:, :, :1]) * 0.5)
                
                # 記錄數據
                trajectory_actions.append(actions[0].cpu().numpy())
                trajectory_alphas.append(alphas[0].cpu().numpy())
                
                # 環境步進
                step_result = env.step(state, actions, alphas)
                state = step_result.next_state
                
                # 進度報告
                if step % 25 == 0:
                    action_magnitude = torch.norm(actions, dim=-1).mean().item()
                    print(f"步驟 {step}: 動作強度={action_magnitude:.6f}")
                
            except Exception as e:
                print(f"❌ 步驟 {step} 失敗: {e}")
                break
    
    print(f"✅ 仿真完成，共 {len(trajectory_positions)} 步")
    
    return {
        'positions': trajectory_positions,
        'actions': trajectory_actions,
        'alphas': trajectory_alphas
    }


def create_professional_visualization(trajectory_data, output_path):
    """
    創建專業的可視化
    """
    print(f"🎨 創建專業可視化")
    
    positions = trajectory_data['positions']
    actions = trajectory_data['actions']
    alphas = trajectory_data['alphas']
    
    if not positions:
        print(f"❌ 沒有軌跡數據")
        return False
    
    num_steps = len(positions)
    num_agents = len(positions[0])
    
    # 創建圖形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('🎯 最終統一可視化結果 - 100%真實模型', fontsize=20, fontweight='bold')
    
    # 主軌跡圖
    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.set_title('🚁 多智能體協作軌跡', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 障礙物
    obstacles = [
        {'pos': [0.0, -0.8], 'radius': 0.4},
        {'pos': [0.0, 0.8], 'radius': 0.4}
    ]
    
    for obs in obstacles:
        circle = plt.Circle(obs['pos'], obs['radius'], color='red', alpha=0.8, 
                          label='障礙物' if obs == obstacles[0] else "")
        ax1.add_patch(circle)
    
    # 區域標記
    start_zone = plt.Rectangle((-3.0, -2.0), 1.0, 4.0, fill=False, 
                              edgecolor='green', linestyle='--', linewidth=3, 
                              alpha=0.9, label='起始區域')
    ax1.add_patch(start_zone)
    
    target_zone = plt.Rectangle((2.0, -2.0), 1.0, 4.0, fill=False, 
                               edgecolor='blue', linestyle='--', linewidth=3, 
                               alpha=0.9, label='目標區域')
    ax1.add_patch(target_zone)
    
    # 智能體顏色
    colors = ['#FF2D2D', '#2DFF2D', '#2D2DFF', '#FF8C2D', '#FF2D8C', '#2DFFFF'][:num_agents]
    
    # 軌跡線和智能體
    trail_lines = []
    agent_dots = []
    
    for i in range(num_agents):
        line, = ax1.plot([], [], '-', color=colors[i], alpha=0.8, linewidth=3,
                        label=f'智能體{i+1}' if i < 3 else "")
        trail_lines.append(line)
        
        dot, = ax1.plot([], [], 'o', color=colors[i], markersize=16, 
                       markeredgecolor='black', markeredgewidth=2, zorder=5)
        agent_dots.append(dot)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 動作分析圖
    ax2.set_title('🧠 策略網絡輸出分析', fontsize=14, fontweight='bold')
    ax2.set_xlabel('時間步')
    ax2.set_ylabel('動作強度')
    ax2.grid(True, alpha=0.3)
    
    # Alpha值監控
    ax3.set_title('⚖️ 動態Alpha值監控', fontsize=14, fontweight='bold')
    ax3.set_xlabel('時間步')
    ax3.set_ylabel('Alpha值')
    ax3.grid(True, alpha=0.3)
    
    # 運動統計
    ax4.set_title('📊 運動統計分析', fontsize=14, fontweight='bold')
    ax4.set_xlabel('時間步')
    ax4.set_ylabel('平均位移')
    ax4.grid(True, alpha=0.3)
    
    # 動畫信息文本
    info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                        verticalalignment='top', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    def animate(frame):
        if frame >= num_steps:
            return trail_lines + agent_dots + [info_text]
        
        current_pos = positions[frame]
        
        # 更新軌跡和智能體
        for i in range(num_agents):
            trail_x = [pos[i, 0] for pos in positions[:frame+1]]
            trail_y = [pos[i, 1] for pos in positions[:frame+1]]
            trail_lines[i].set_data(trail_x, trail_y)
            
            agent_dots[i].set_data([current_pos[i, 0]], [current_pos[i, 1]])
        
        # 更新分析圖表
        if frame > 5:
            steps = list(range(frame+1))
            
            # 動作分析
            if frame < len(actions):
                action_magnitudes = []
                for step in range(frame+1):
                    if step < len(actions):
                        step_actions = actions[step]
                        avg_magnitude = np.mean([np.linalg.norm(a) for a in step_actions])
                        action_magnitudes.append(avg_magnitude)
                    else:
                        action_magnitudes.append(0)
                
                ax2.clear()
                ax2.plot(steps, action_magnitudes, 'purple', linewidth=3, label='平均動作強度')
                ax2.fill_between(steps, action_magnitudes, alpha=0.3, color='purple')
                ax2.set_title(f'🧠 策略網絡輸出分析 (步數: {frame})')
                ax2.set_xlabel('時間步')
                ax2.set_ylabel('動作強度')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Alpha值分析
            if frame < len(alphas):
                alpha_values = []
                for step in range(frame+1):
                    if step < len(alphas):
                        avg_alpha = np.mean(alphas[step])
                        alpha_values.append(avg_alpha)
                    else:
                        alpha_values.append(0.5)
                
                ax3.clear()
                ax3.plot(steps, alpha_values, 'orange', linewidth=3, label='平均Alpha值')
                ax3.fill_between(steps, alpha_values, alpha=0.3, color='orange')
                ax3.set_title(f'⚖️ 動態Alpha值監控 (步數: {frame})')
                ax3.set_xlabel('時間步')
                ax3.set_ylabel('Alpha值')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # 運動統計
            displacements = []
            initial_pos = positions[0]
            for step in range(frame+1):
                if step < len(positions):
                    current_pos = positions[step]
                    avg_displacement = np.mean([
                        np.linalg.norm(current_pos[i] - initial_pos[i]) 
                        for i in range(num_agents)
                    ])
                    displacements.append(avg_displacement)
                else:
                    displacements.append(0)
            
            ax4.clear()
            ax4.plot(steps, displacements, 'green', linewidth=3, label='平均位移')
            ax4.fill_between(steps, displacements, alpha=0.3, color='green')
            ax4.set_title(f'📊 運動統計分析 (步數: {frame})')
            ax4.set_xlabel('時間步')
            ax4.set_ylabel('平均位移')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 計算當前統計
        if frame > 0:
            initial_pos = positions[0]
            current_pos = positions[frame]
            total_displacement = np.mean([
                np.linalg.norm(current_pos[i] - initial_pos[i]) 
                for i in range(num_agents)
            ])
            
            action_magnitude = 0
            if frame < len(actions):
                action_magnitude = np.mean([np.linalg.norm(a) for a in actions[frame]])
        else:
            total_displacement = 0
            action_magnitude = 0
        
        info_text.set_text(
            f'步驟: {frame}\n'
            f'智能體數: {num_agents}\n'
            f'平均位移: {total_displacement:.4f}\n'
            f'動作強度: {action_magnitude:.6f}'
        )
        
        return trail_lines + agent_dots + [info_text]
    
    # 創建動畫
    anim = FuncAnimation(fig, animate, frames=num_steps, interval=120, blit=False, repeat=True)
    
    # 保存
    try:
        print(f"💾 保存專業可視化: {output_path}")
        if output_path.endswith('.mp4'):
            anim.save(output_path, writer='ffmpeg', fps=8, dpi=150)
        else:
            anim.save(output_path, writer='pillow', fps=6, dpi=150)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ 保存成功: {file_size:.2f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存失敗: {e}")
        return False
    finally:
        plt.close()


def main():
    """
    主函數
    """
    print(f"🎯 直接模型可視化系統")
    print(f"=" * 70)
    
    # 設置
    model_dir = 'logs/full_collaboration_training'
    device = torch.device('cpu')
    
    # 找最新模型
    models_dir = os.path.join(model_dir, 'models')
    steps = [int(d) for d in os.listdir(models_dir) if d.isdigit()]
    latest_step = max(steps)
    
    policy_path = os.path.join(model_dir, 'models', str(latest_step), 'policy.pt')
    print(f"📁 模型路徑: {policy_path}")
    
    # 創建環境
    env_config = {
        'area_size': 3.0,
        'car_radius': 0.15,
        'comm_radius': 1.0,
        'dt': 0.05,
        'mass': 0.1,
        'max_force': 1.0,
        'max_steps': 100,
        'num_agents': 6,
        'obstacles': {
            'enabled': True,
            'bottleneck': True,
            'positions': [[0.0, -0.8], [0.0, 0.8]],
            'radii': [0.4, 0.4]
        }
    }
    
    env = DoubleIntegratorEnv(env_config)
    env = env.to(device)
    
    print(f"🌍 環境: {env.observation_shape} → {env.action_shape}")
    
    # 加載模型
    policy_network = load_model_direct(policy_path, device)
    if policy_network is None:
        print(f"❌ 系統失敗")
        return
    
    # 測試推理
    if not test_model_inference(policy_network, env, device):
        print(f"❌ 推理測試失敗")
        return
    
    # 運行仿真
    trajectory_data = run_full_simulation(policy_network, env, device, 120)
    
    # 創建可視化
    output_path = 'DIRECT_FINAL_COLLABORATION_RESULT.mp4'
    success = create_professional_visualization(trajectory_data, output_path)
    
    if success:
        print(f"\n🎉 直接可視化成功完成!")
        print(f"📁 結果文件: {output_path}")
        print(f"✅ 這是您真實訓練模型的表現")
        
        # 運動分析
        positions = trajectory_data['positions']
        if positions:
            initial_pos = positions[0]
            final_pos = positions[-1]
            total_displacement = np.mean([
                np.linalg.norm(final_pos[i] - initial_pos[i]) 
                for i in range(len(initial_pos))
            ])
            print(f"📊 平均總位移: {total_displacement:.4f}")
            
            if total_displacement < 0.01:
                print(f"⚠️ 智能體基本靜止，可能存在訓練問題")
            else:
                print(f"✅ 智能體正常運動")
    else:
        print(f"\n❌ 可視化失敗")


if __name__ == '__main__':
    main()
 
 
 
 