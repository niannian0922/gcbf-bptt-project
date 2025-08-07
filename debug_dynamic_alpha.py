#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态Alpha调试脚本 - 自适应安全边距的专门调试工具

此脚本专门用于隔离和分析动态alpha机制的行为，通过：
1. 加载预训练的"黄金基准"固定alpha模型
2. 创建简化的两智能体对撞场景
3. 仅训练alpha预测头，观察学习动态
4. 提供详细的调试输出和分析

作者: GCBF-BPTT项目组
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

from gcbfplus.env import DoubleIntegratorEnv
from gcbfplus.env.gcbf_safety_layer import GCBFSafetyLayer
from gcbfplus.policy import BPTTPolicy


class AlphaDebugger:
    """动态Alpha调试器 - 专门用于分析自适应安全边距学习"""
    
    def __init__(self, model_path: str = "logs/bptt/models/1000"):
        """
        初始化调试器
        
        参数:
            model_path: 预训练模型路径（黄金基准）
        """
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 动态Alpha调试器初始化")
        print(f"   使用设备: {self.device}")
        print(f"   模型路径: {self.model_path}")
        
        # 加载配置和模型
        self._load_golden_model()
        
        # 创建简化环境
        self._create_debug_environment()
        
        # 设置调试训练
        self._setup_debug_training()
    
    def _load_golden_model(self):
        """加载预训练的黄金基准模型"""
        print("\n📂 加载黄金基准模型...")
        
        # 加载配置
        config_path = self.model_path / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print(f"   配置文件: {config_path}")
        print(f"   原始训练步数: {self.config['training']['training_steps']}")
        print(f"   原始安全权重: {self.config['training']['safety_weight']}")
        
        # 修改配置以支持动态alpha
        self._modify_config_for_alpha_debug()
        
        # 创建策略网络
        self.policy = BPTTPolicy(self.config['networks']['policy']).to(self.device)
        
        # 加载预训练权重
        policy_state_path = self.model_path / "policy.pt"
        if policy_state_path.exists():
            print(f"   加载策略权重: {policy_state_path}")
            state_dict = torch.load(policy_state_path, map_location=self.device)
            
            # 处理可能的键不匹配（旧模型兼容性）
            self._load_compatible_state_dict(state_dict)
        else:
            print("   ⚠️  未找到预训练权重，使用随机初始化")
    
    def _modify_config_for_alpha_debug(self):
        """修改配置以支持动态alpha调试"""
        print("   🔧 修改配置以支持动态alpha...")
        
        # 确保策略头启用alpha预测
        if 'policy_head' not in self.config['networks']['policy']:
            self.config['networks']['policy']['policy_head'] = {}
        
        self.config['networks']['policy']['policy_head'].update({
            'predict_alpha': True,
            'alpha_hidden_dim': 64,
            'output_dim': 2,  # 2D动作空间
            'alpha_bounds': [0.1, 5.0]  # alpha范围
        })
        
        # 设置调试环境参数
        self.config['env'].update({
            'num_agents': 2,  # 简化为两智能体
            'area_size': 2.0,  # 增大空间
            'cbf_alpha': 1.0,  # 基础alpha值
            'comm_radius': 1.5,  # 增大通信范围
            'max_steps': 100   # 减少步数以快速调试
        })
        
        print(f"      ✅ 智能体数量: {self.config['env']['num_agents']}")
        print(f"      ✅ 启用动态alpha: {self.config['networks']['policy']['policy_head']['predict_alpha']}")
    
    def _load_compatible_state_dict(self, state_dict):
        """加载兼容的状态字典，处理键名不匹配"""
        try:
            # 尝试直接加载
            self.policy.load_state_dict(state_dict, strict=False)
            print("   ✅ 成功加载预训练权重")
        except Exception as e:
            print(f"   ⚠️  权重加载异常: {e}")
            print("   🔄 尝试兼容性映射...")
            
            # 创建兼容性映射
            compatible_dict = {}
            policy_state = self.policy.state_dict()
            
            # 映射已知的键名变化
            key_mappings = {
                'head.': 'policy_head.',
                'memory.gru_cell': 'memory.gru',
                # 可以根据需要添加更多映射
            }
            
            for old_key, tensor in state_dict.items():
                new_key = old_key
                for old_pattern, new_pattern in key_mappings.items():
                    new_key = new_key.replace(old_pattern, new_pattern)
                
                if new_key in policy_state:
                    compatible_dict[new_key] = tensor
                    print(f"      映射: {old_key} -> {new_key}")
            
            # 加载映射后的权重
            self.policy.load_state_dict(compatible_dict, strict=False)
            print("   ✅ 使用兼容性映射成功加载部分权重")
    
    def _create_debug_environment(self):
        """创建简化的调试环境"""
        print("\n🌍 创建调试环境...")
        
        # 创建双积分器环境
        self.env = DoubleIntegratorEnv(self.config['env'])
        
        # 创建安全层
        self.safety_layer = GCBFSafetyLayer(
            alpha=self.config['env']['cbf_alpha'],
            device=self.device
        )
        
        print(f"   环境尺寸: {self.config['env']['area_size']}m x {self.config['env']['area_size']}m")
        print(f"   智能体数量: {self.config['env']['num_agents']}")
        print(f"   基础CBF alpha: {self.config['env']['cbf_alpha']}")
        
        # 创建对撞初始状态
        self._create_collision_scenario()
    
    def _create_collision_scenario(self):
        """创建两智能体正面对撞场景"""
        print("   🚗💥 设置正面对撞场景...")
        
        # 智能体1: 从左侧向右移动
        agent1_pos = np.array([-0.8, 0.0])  # 左侧起始
        agent1_vel = np.array([0.5, 0.0])   # 向右移动
        
        # 智能体2: 从右侧向左移动  
        agent2_pos = np.array([0.8, 0.0])   # 右侧起始
        agent2_vel = np.array([-0.5, 0.0])  # 向左移动
        
        # 构建初始状态
        self.initial_positions = np.stack([agent1_pos, agent2_pos])
        self.initial_velocities = np.stack([agent1_vel, agent2_vel])
        
        print(f"      智能体1: pos={agent1_pos}, vel={agent1_vel}")
        print(f"      智能体2: pos={agent2_pos}, vel={agent2_vel}")
        print(f"      预计碰撞时间: {1.6/1.0:.1f}秒 (如无干预)")
    
    def _setup_debug_training(self):
        """设置调试训练"""
        print("\n🎯 设置调试训练...")
        
        # 冻结除alpha_head外的所有参数
        self._freeze_non_alpha_parameters()
        
        # 创建只针对alpha参数的优化器
        alpha_params = []
        for name, param in self.policy.named_parameters():
            if 'alpha_network' in name and param.requires_grad:
                alpha_params.append(param)
        
        if len(alpha_params) == 0:
            print("   ⚠️  未找到可训练的alpha参数！")
            # 如果没有alpha网络，需要添加
            self._add_alpha_network()
            alpha_params = [p for n, p in self.policy.named_parameters() 
                          if 'alpha_network' in n and p.requires_grad]
        
        self.alpha_optimizer = optim.Adam(alpha_params, lr=0.001)
        
        print(f"   可训练alpha参数数量: {sum(p.numel() for p in alpha_params)}")
        print(f"   优化器: Adam (lr=0.001)")
        
        # 调试指标存储
        self.debug_metrics = {
            'steps': [],
            'predicted_alpha': [],
            'safety_loss': [],
            'alpha_reg_loss': [],
            'total_loss': [],
            'min_distance': [],
            'collision_occurred': []
        }
    
    def _freeze_non_alpha_parameters(self):
        """冻结除alpha网络外的所有参数"""
        frozen_params = 0
        trainable_params = 0
        
        for name, param in self.policy.named_parameters():
            if 'alpha_network' in name:
                param.requires_grad = True
                trainable_params += param.numel()
                print(f"   🔓 可训练: {name} ({param.numel()} 参数)")
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"   ❄️  冻结参数: {frozen_params:,}")
        print(f"   🔥 可训练参数: {trainable_params:,}")
    
    def _add_alpha_network(self):
        """如果缺失，添加alpha网络"""
        print("   🔧 添加alpha预测网络...")
        
        if not hasattr(self.policy.policy_head, 'alpha_network') or self.policy.policy_head.alpha_network is None:
            # 获取输入维度
            input_dim = self.policy.policy_head.input_dim
            alpha_hidden_dim = 64
            
            # 创建alpha网络
            self.policy.policy_head.alpha_network = nn.Sequential(
                nn.Linear(input_dim, alpha_hidden_dim),
                nn.ReLU(),
                nn.Linear(alpha_hidden_dim, 1),
                nn.Softplus()  # 确保alpha > 0
            ).to(self.device)
            
            # 启用alpha预测
            self.policy.policy_head.predict_alpha = True
            
            print(f"      ✅ 创建alpha网络: {input_dim} -> {alpha_hidden_dim} -> 1")
    
    def run_debug_loop(self, max_steps: int = 100):
        """运行调试训练循环"""
        print(f"\n🚀 开始调试训练循环 (最大步数: {max_steps})")
        print("=" * 60)
        
        for step in range(max_steps):
            # 重置环境到初始对撞状态
            state = self._reset_to_collision_scenario()
            
            # 前向传播
            obs_tensor = torch.FloatTensor(state.observations).unsqueeze(0).to(self.device)
            actions, predicted_alpha = self.policy(obs_tensor)
            
            # 提取预测的alpha值
            if predicted_alpha is not None:
                alpha_value = predicted_alpha.mean().item()
            else:
                alpha_value = self.config['env']['cbf_alpha']  # 使用默认值
                print(f"   ⚠️  步骤 {step}: 未预测alpha，使用默认值 {alpha_value}")
            
            # 应用安全层（使用预测的alpha）
            safe_actions = self._apply_safety_layer(state, actions.squeeze(0), alpha_value)
            
            # 计算损失
            safety_loss, alpha_reg_loss, total_loss = self._calculate_debug_losses(
                state, safe_actions, predicted_alpha, alpha_value
            )
            
            # 反向传播（仅更新alpha网络）
            self.alpha_optimizer.zero_grad()
            total_loss.backward()
            self.alpha_optimizer.step()
            
            # 计算额外指标
            min_distance = self._calculate_min_distance(state.positions)
            collision_occurred = min_distance < (2 * self.config['env']['agent_radius'])
            
            # 存储调试指标
            self._store_debug_metrics(step, alpha_value, safety_loss.item(), 
                                    alpha_reg_loss.item(), total_loss.item(), 
                                    min_distance, collision_occurred)
            
            # 打印详细调试信息
            self._print_debug_info(step, alpha_value, safety_loss.item(), 
                                 alpha_reg_loss.item(), total_loss.item(), 
                                 min_distance, collision_occurred)
            
            # 每10步打印分隔线
            if (step + 1) % 10 == 0:
                print("-" * 60)
        
        print("🏁 调试训练完成!")
        self._analyze_results()
    
    def _reset_to_collision_scenario(self):
        """重置环境到对撞场景"""
        # 重置环境
        state = self.env.reset()
        
        # 设置固定的对撞初始状态
        state.positions = self.initial_positions.copy()
        state.velocities = self.initial_velocities.copy()
        
        # 重置策略记忆
        if hasattr(self.policy, 'memory') and hasattr(self.policy.memory, 'reset'):
            self.policy.memory.reset()
        
        return state
    
    def _apply_safety_layer(self, state, actions, alpha_value):
        """应用安全层约束"""
        # 转换状态为安全层格式
        positions = torch.FloatTensor(state.positions).to(self.device)
        velocities = torch.FloatTensor(state.velocities).to(self.device)
        
        # 应用安全约束（使用预测的alpha）
        safe_actions = self.safety_layer.apply_safety_constraint(
            positions, velocities, actions, alpha=alpha_value
        )
        
        return safe_actions
    
    def _calculate_debug_losses(self, state, safe_actions, predicted_alpha, alpha_value):
        """计算调试损失"""
        # 安全损失：基于CBF约束
        positions = torch.FloatTensor(state.positions).to(self.device)
        velocities = torch.FloatTensor(state.velocities).to(self.device)
        
        # 计算最小距离
        distances = torch.cdist(positions, positions)
        distances = distances + torch.eye(len(positions), device=self.device) * 1e6  # 忽略自身
        min_distance = distances.min()
        
        # 安全损失：距离越小，损失越大
        safety_radius = 2 * self.config['env']['agent_radius']
        safety_loss = torch.clamp(safety_radius - min_distance, min=0).pow(2)
        
        # Alpha正则化损失：鼓励适中的alpha值
        if predicted_alpha is not None:
            target_alpha = 1.5  # 目标alpha值
            alpha_reg_loss = (predicted_alpha.mean() - target_alpha).pow(2) * 0.01
        else:
            alpha_reg_loss = torch.tensor(0.0, device=self.device)
        
        # 总损失
        total_loss = safety_loss + alpha_reg_loss
        
        return safety_loss, alpha_reg_loss, total_loss
    
    def _calculate_min_distance(self, positions):
        """计算智能体间最小距离"""
        if len(positions) < 2:
            return float('inf')
        
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        return min(distances)
    
    def _store_debug_metrics(self, step, alpha_value, safety_loss, alpha_reg_loss, 
                           total_loss, min_distance, collision_occurred):
        """存储调试指标"""
        self.debug_metrics['steps'].append(step)
        self.debug_metrics['predicted_alpha'].append(alpha_value)
        self.debug_metrics['safety_loss'].append(safety_loss)
        self.debug_metrics['alpha_reg_loss'].append(alpha_reg_loss)
        self.debug_metrics['total_loss'].append(total_loss)
        self.debug_metrics['min_distance'].append(min_distance)
        self.debug_metrics['collision_occurred'].append(collision_occurred)
    
    def _print_debug_info(self, step, alpha_value, safety_loss, alpha_reg_loss, 
                         total_loss, min_distance, collision_occurred):
        """打印详细调试信息"""
        collision_status = "🚨 碰撞!" if collision_occurred else "✅ 安全"
        
        print(f"步骤 {step:3d} | "
              f"Alpha: {alpha_value:.3f} | "
              f"安全损失: {safety_loss:.4f} | "
              f"Alpha正则: {alpha_reg_loss:.4f} | "
              f"总损失: {total_loss:.4f} | "
              f"最小距离: {min_distance:.3f}m | "
              f"{collision_status}")
    
    def _analyze_results(self):
        """分析调试结果"""
        print("\n📊 调试结果分析:")
        print("=" * 50)
        
        metrics = self.debug_metrics
        
        # 基本统计
        final_alpha = metrics['predicted_alpha'][-1]
        initial_alpha = metrics['predicted_alpha'][0]
        alpha_change = final_alpha - initial_alpha
        
        avg_safety_loss = np.mean(metrics['safety_loss'])
        avg_alpha_reg_loss = np.mean(metrics['alpha_reg_loss'])
        min_distance_achieved = min(metrics['min_distance'])
        collision_rate = sum(metrics['collision_occurred']) / len(metrics['steps'])
        
        print(f"Alpha学习:")
        print(f"  初始Alpha: {initial_alpha:.3f}")
        print(f"  最终Alpha: {final_alpha:.3f}")
        print(f"  Alpha变化: {alpha_change:+.3f}")
        
        print(f"\n损失分析:")
        print(f"  平均安全损失: {avg_safety_loss:.4f}")
        print(f"  平均Alpha正则损失: {avg_alpha_reg_loss:.4f}")
        
        print(f"\n安全性能:")
        print(f"  最小距离: {min_distance_achieved:.3f}m")
        print(f"  碰撞率: {collision_rate:.1%}")
        
        # 学习趋势分析
        if len(metrics['steps']) > 10:
            early_alpha = np.mean(metrics['predicted_alpha'][:10])
            late_alpha = np.mean(metrics['predicted_alpha'][-10:])
            learning_trend = "上升" if late_alpha > early_alpha else "下降"
            
            print(f"\n学习趋势:")
            print(f"  前10步平均Alpha: {early_alpha:.3f}")
            print(f"  后10步平均Alpha: {late_alpha:.3f}")
            print(f"  整体趋势: Alpha {learning_trend}")
        
        # 生成可视化
        self._plot_debug_results()
    
    def _plot_debug_results(self):
        """绘制调试结果"""
        print("\n📈 生成调试可视化图表...")
        
        metrics = self.debug_metrics
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('动态Alpha调试结果分析', fontsize=16, fontweight='bold')
        
        # 1. Alpha预测变化
        ax1.plot(metrics['steps'], metrics['predicted_alpha'], 'b-', linewidth=2, label='预测Alpha')
        ax1.axhline(y=1.5, color='r', linestyle='--', alpha=0.7, label='目标Alpha')
        ax1.set_xlabel('训练步数')
        ax1.set_ylabel('Alpha值')
        ax1.set_title('Alpha预测学习曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 损失变化
        ax2.plot(metrics['steps'], metrics['safety_loss'], 'r-', label='安全损失', alpha=0.8)
        ax2.plot(metrics['steps'], metrics['alpha_reg_loss'], 'g-', label='Alpha正则损失', alpha=0.8)
        ax2.plot(metrics['steps'], metrics['total_loss'], 'k-', label='总损失', linewidth=2)
        ax2.set_xlabel('训练步数')
        ax2.set_ylabel('损失值')
        ax2.set_title('损失函数变化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 3. 最小距离变化
        ax3.plot(metrics['steps'], metrics['min_distance'], 'purple', linewidth=2)
        safety_radius = 2 * self.config['env']['agent_radius']
        ax3.axhline(y=safety_radius, color='r', linestyle='--', alpha=0.7, label=f'安全阈值 ({safety_radius:.2f}m)')
        ax3.set_xlabel('训练步数')
        ax3.set_ylabel('最小距离 (m)')
        ax3.set_title('智能体间最小距离')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 碰撞状态
        collision_indicator = [1 if c else 0 for c in metrics['collision_occurred']]
        ax4.scatter(metrics['steps'], collision_indicator, c=['red' if c else 'green' for c in collision_indicator], alpha=0.6)
        ax4.set_xlabel('训练步数')
        ax4.set_ylabel('碰撞状态')
        ax4.set_title('碰撞发生情况')
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['安全', '碰撞'])
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('debug_dynamic_alpha_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("   ✅ 调试图表已保存: debug_dynamic_alpha_results.png")


def main():
    """主函数"""
    print("🚨 动态Alpha调试器启动")
    print("=" * 60)
    
    # 检查可用模型
    available_models = []
    logs_dir = Path("logs")
    if logs_dir.exists():
        for subdir in logs_dir.iterdir():
            if subdir.is_dir():
                models_dir = subdir / "models"
                if models_dir.exists():
                    for model_dir in models_dir.iterdir():
                        if model_dir.is_dir() and (model_dir / "policy.pt").exists():
                            available_models.append(str(model_dir))
    
    if not available_models:
        print("❌ 未找到可用的预训练模型!")
        print("   请先运行训练脚本生成模型")
        return
    
    print(f"🔍 发现 {len(available_models)} 个可用模型:")
    for i, model in enumerate(available_models):
        print(f"   {i+1}. {model}")
    
    # 选择最新的模型（默认）
    selected_model = "logs/bptt/models/1000"  # 黄金基准
    if os.path.exists(selected_model):
        print(f"\n🎯 使用黄金基准模型: {selected_model}")
    else:
        # 如果默认模型不存在，使用第一个可用的
        selected_model = available_models[0]
        print(f"\n🎯 使用模型: {selected_model}")
    
    try:
        # 创建调试器
        debugger = AlphaDebugger(selected_model)
        
        # 运行调试循环
        debugger.run_debug_loop(max_steps=100)
        
        print("\n🎉 调试完成!")
        print("   查看 debug_dynamic_alpha_results.png 以获得详细分析")
        
    except Exception as e:
        print(f"\n❌ 调试过程中出现异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()