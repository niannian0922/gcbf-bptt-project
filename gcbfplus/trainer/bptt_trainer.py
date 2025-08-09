# End-to-end BPTT trainer with bottleneck scenario analysis

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, List, Union, Callable

# 尝试导入wandb，但设为可选
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("警告：未找到wandb。训练将继续进行，但不会记录到wandb。")

from ..env.base_env import BaseEnv, EnvState
from ..env.multi_agent_env import MultiAgentEnv, MultiAgentState
from ..policy.bptt_policy import BPTTPolicy
from ..utils.episode_logger import EpisodeLogger, compute_min_distances_to_obstacles, compute_goal_distances


class BPTTTrainer:
    """
    实现时序反向传播（BPTT）的训练器，通过可微分物理仿真器进行策略和CBF网络的端到端优化。
    
    该训练器通过仿真器的梯度直接优化两个网络，无需Q学习、专家策略和重放缓冲区。
    支持自适应安全边距（动态Alpha）的创新训练方法。
    """
    
    def __init__(
        self,
        env: BaseEnv,
        policy_network: nn.Module,
        cbf_network: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化BPTT训练器。
        
        参数:
            env: 可微分环境实例
            policy_network: 要训练的策略网络
            cbf_network: 可选的CBF安全网络
            optimizer: 可选的优化器（如果为None将创建默认值）
            config: 配置字典
        """
        # 存储环境和网络
        self.env = env
        self.policy_network = policy_network
        self.cbf_network = cbf_network
        
        # 从策略网络获取设备
        self.device = next(policy_network.parameters()).device
        
        # 如果没有提供配置，则设置默认配置
        self.config = {} if config is None else config
        
        # 从配置中提取参数
        self.run_name = self.config.get('run_name', 'BPTT_Run')
        # 基于run_name构建唯一的日志目录
        self.log_dir = f"logs/{self.run_name}"
        self.num_agents = self.config.get('num_agents', 8)
        self.area_size = self.config.get('area_size', 1.0)
        
        # 训练参数
        # 🚀 修复：从training子部分正确读取训练步数
        training_config = self.config.get('training', {})
        self.training_steps = training_config.get('training_steps', self.config.get('training_steps', 10000))
        self.eval_interval = training_config.get('eval_interval', self.config.get('eval_interval', 100))
        self.save_interval = training_config.get('save_interval', self.config.get('save_interval', 1000))
        self.horizon_length = training_config.get('horizon_length', self.config.get('horizon_length', 50))
        self.eval_horizon = training_config.get('eval_horizon', self.config.get('eval_horizon', 100))
        self.max_grad_norm = training_config.get('max_grad_norm', self.config.get('max_grad_norm', 1.0))
        
        # 损失权重 - 支持新的控制正则化
        loss_weights = self.config.get('loss_weights', {})
        self.goal_weight = loss_weights.get('goal_weight', self.config.get('goal_weight', 1.0))
        self.safety_weight = loss_weights.get('safety_weight', self.config.get('safety_weight', 10.0))
        self.control_weight = loss_weights.get('control_weight', self.config.get('control_weight', 0.1))
        self.jerk_weight = loss_weights.get('jerk_weight', self.config.get('jerk_weight', 0.05))
        self.alpha_reg_weight = loss_weights.get('alpha_reg_weight', self.config.get('alpha_reg_weight', 0.01))
        self.progress_weight = loss_weights.get('progress_weight', self.config.get('progress_weight', 0.0))
        
        # 新增控制正则化权重
        self.acceleration_loss_weight = loss_weights.get('acceleration_loss_weight', 0.01)
        
        self.cbf_alpha = self.config.get('cbf_alpha', 1.0)
        
        # 创建日志目录
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.model_dir = os.path.join(self.log_dir, 'models')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # 如果没有提供优化器，则初始化
        if optimizer is None:
            params = list(self.policy_network.parameters())
            if self.cbf_network is not None:
                params += list(self.cbf_network.parameters())
                
            self.optimizer = optim.Adam(
                params,
                lr=self.config.get('learning_rate', 0.001)
            )
        else:
            self.optimizer = optimizer
        
        # 如果指定了学习率调度器，则初始化
        if self.config.get('use_lr_scheduler', False):
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('lr_step_size', 2000),
                gamma=self.config.get('lr_gamma', 0.5)
            )
        else:
            self.scheduler = None
        
        # 初始化数据记录器
        self.enable_episode_logging = self.config.get('enable_episode_logging', False)
        if self.enable_episode_logging:
            log_dir = os.path.join(self.log_dir, 'episode_logs')
            self.episode_logger = EpisodeLogger(log_dir=log_dir, prefix="bptt_episode")
            print(f"📊 数据记录已启用: {log_dir}")
        else:
            self.episode_logger = None
    
    def initialize_scenario(self, batch_size: int = 1) -> MultiAgentState:
        """
        初始化一个新的场景，带有随机初始状态和目标。
        
        参数:
            batch_size: 要初始化的并行环境数量
            
        返回:
            环境状态
        """
        # 使用环境的重置方法初始化状态
        return self.env.reset(batch_size=batch_size, randomize=True)
    
    def train(self) -> None:
        """
        实现BPTT优化的主训练循环。
        """
        print(f"Starting BPTT training with configuration:")
        print(f"  Run name: {self.run_name}")
        print(f"  Steps: {self.training_steps}")
        print(f"  Horizon: {self.horizon_length}")
        print(f"  Log dir: {self.log_dir}")
        
        # 在离线模式下初始化wandb
        if WANDB_AVAILABLE:
            wandb.init(name=self.run_name, project='gcbf-bptt', dir=self.log_dir, config=self.config, mode="offline")
        
        start_time = time.time()
        
        # 将环境设置为训练模式以启用梯度衰减
        self.env.train()
        
        pbar = tqdm(total=self.training_steps)
        for step in range(self.training_steps):
            # 训练模式
            self.policy_network.train()
            if self.cbf_network is not None:
                self.cbf_network.train()
            
            # 在每次反向传播之前清零梯度
            self.optimizer.zero_grad()
            
            # 初始化场景
            state = self.initialize_scenario()
            
            # BPTT Rollout
            trajectory_states = []
            trajectory_actions = []
            trajectory_alphas = []
            trajectory_rewards = []
            trajectory_costs = []
            safety_losses = []
            
            # 运行前向仿真并收集轨迹数据
            current_state = state
            for t in range(self.horizon_length):
                # 保存当前状态
                trajectory_states.append(current_state)
                
                # 从状态获取观测值
                observations = self.env.get_observation(current_state)
                
                # 在将观测值传递给网络之前，将观测值移动到正确的设备（CPU或GPU）
                observations = observations.to(self.device)
                actions, alpha, raw_dynamic_margins = self.policy_network(observations)
                
                # 🚀 CORE INNOVATION: 处理动态安全裕度
                dynamic_margins = None
                if raw_dynamic_margins is not None and self.config.get("use_adaptive_margin", False):
                    # 将Sigmoid输出(0,1)映射到配置的[min_margin, max_margin]范围
                    min_margin = self.config.get("min_safety_margin", 0.15)
                    max_margin = self.config.get("max_safety_margin", 0.4)
                    # 关键修复：确保dynamic_margins张量在正确的设备上
                    dynamic_margins = (min_margin + raw_dynamic_margins * (max_margin - min_margin)).to(self.device)
                
                # 处理alpha为None的情况（固定alpha配置）
                if alpha is None:
                    # 使用环境默认的alpha值
                    batch_size, num_agents = actions.shape[:2]
                    alpha = torch.full((batch_size, num_agents, 1), 
                                     self.env.cbf_alpha, 
                                     device=self.device, 
                                     dtype=actions.dtype)
                
                # 存储用于反向传播的分离副本
                trajectory_actions.append(actions.clone())
                trajectory_alphas.append(alpha.clone())
                # 🚀 CORE INNOVATION: 存储动态安全裕度
                if dynamic_margins is not None:
                    if not hasattr(self, 'trajectory_margins'):
                        self.trajectory_margins = []
                    self.trajectory_margins.append(dynamic_margins.clone())
                
                # 如果提供了CBF网络，则计算屏障函数值用于损失计算
                if self.cbf_network is not None:
                    # 🚀 CORE INNOVATION: CBF网络使用动态安全裕度
                    cbf_values = self.cbf_network.barrier_function(current_state, dynamic_margins)
                    
                    # 根据CBF值计算安全损失
                    # 负值表示不安全状态
                    safety_loss = torch.mean(torch.relu(-cbf_values))
                    safety_losses.append(safety_loss)
                
                # 🛡️ PROBABILISTIC SAFETY SHIELD: 在环境中使用动态alpha进行一步
                step_result = self.env.step(current_state, actions, alpha)
                next_state = step_result.next_state
                rewards = step_result.reward
                costs = step_result.cost
                
                # 🛡️ 计算安全信心分数用于新的风险评估损失
                if hasattr(self.env, 'safety_layer') and self.env.safety_layer is not None:
                    alpha_safety = self.env.safety_layer.compute_safety_confidence(current_state, dynamic_margins)
                    # 检查是否发生碰撞（用于新的CBF损失计算）
                    is_collision = costs > 0  # 假设cost > 0表示碰撞
                else:
                    alpha_safety = None
                    is_collision = costs > 0
                
                # 🛡️ 存储安全信心分数和碰撞标志用于新的风险评估损失
                if alpha_safety is not None:
                    if not hasattr(self, 'trajectory_alpha_safety'):
                        self.trajectory_alpha_safety = []
                    if not hasattr(self, 'trajectory_collisions'):
                        self.trajectory_collisions = []
                    self.trajectory_alpha_safety.append(alpha_safety.clone())
                    self.trajectory_collisions.append(is_collision.clone())
                
                # 保存奖励和成本（分离以防止在反向传播期间修改）
                trajectory_rewards.append(rewards.clone())
                trajectory_costs.append(costs.clone())
                
                # 更新当前状态以进行下一次迭代（分离以防止就地修改）
                current_state = next_state
            
            # 计算损失
            
            # 目标到达损失（使用奖励）
            if trajectory_rewards:
                stacked_rewards = torch.stack(trajectory_rewards)
                goal_loss = -torch.mean(stacked_rewards)
            else:
                # 回退：使用到目标的距离
                goal_distances = self.env.get_goal_distance(current_state)
                goal_loss = torch.mean(goal_distances)
            
            # 控制努力损失
            stacked_actions = torch.stack(trajectory_actions)
            control_effort = torch.mean(stacked_actions ** 2)
            
            # 加速度损失 - L2范数的动作（控制输入的大小）
            acceleration_loss = torch.mean(torch.square(stacked_actions))
            
            # 抖动损失（Jerk Loss）- 连续动作之间差异的L2范数
            jerk_loss = 0.0
            if len(trajectory_actions) > 1:
                action_diffs = []
                for i in range(1, len(trajectory_actions)):
                    action_diff = trajectory_actions[i] - trajectory_actions[i-1]
                    action_diffs.append(action_diff)
                if action_diffs:
                    stacked_diffs = torch.stack(action_diffs)
                    jerk_loss = torch.mean(torch.square(stacked_diffs))
            
            # 🛡️ PROBABILISTIC SAFETY SHIELD: 新的风险评估器损失函数
            # CBF的目的不再是简单地强制h(x) > 0，而是训练GCBF模块成为准确的"风险评估器"
            if hasattr(self, 'trajectory_alpha_safety') and self.trajectory_alpha_safety:
                # 实现新的loss_cbf：如果模型在碰撞前"过度自信"（高alpha_safety），则严重惩罚
                stacked_alpha_safety = torch.stack(self.trajectory_alpha_safety)
                stacked_collisions = torch.stack(self.trajectory_collisions)
                
                # 计算风险评估损失：仅在发生碰撞时惩罚高confidence
                # loss_cbf = alpha_safety if collision else 0.0
                collision_mask = stacked_collisions.float()  # 转换布尔值为浮点数
                # Debug shapes to ensure alignment
                print(f"Shape of collision_mask: {collision_mask.shape}")
                print(f"Shape of stacked_alpha_safety: {stacked_alpha_safety.shape}")
                # Align alpha_safety shape to match collision mask if needed
                if stacked_alpha_safety.dim() < collision_mask.dim():
                    # Assume [T] or [T,1] => expand along agent dimension
                    num_agents = collision_mask.shape[1]
                    alpha_expanded = stacked_alpha_safety.unsqueeze(1).expand(-1, num_agents)
                else:
                    alpha_expanded = stacked_alpha_safety
                risk_assessment_loss = torch.mean(collision_mask * alpha_expanded)
                total_safety_loss = risk_assessment_loss
                
            elif safety_losses:
                # 回退到传统CBF损失（如果没有使用概率防护罩）
                stacked_safety = torch.stack(safety_losses)
                total_safety_loss = torch.mean(stacked_safety)
            else:
                # 如果没有CBF网络，则使用环境成本
                stacked_costs = torch.stack(trajectory_costs)
                total_safety_loss = torch.mean(stacked_costs)
            
            # Alpha正则化损失（鼓励更小的alpha值以提高效率）
            stacked_alphas = torch.stack(trajectory_alphas)
            alpha_regularization_loss = torch.mean(stacked_alphas)
            
            # 进度奖励损失（基于潜力的奖励塑形）
            progress_loss = 0.0
            if self.progress_weight > 0.0 and len(trajectory_states) > 1:
                progress_loss = self._calculate_progress_loss(trajectory_states)
            
            # 计算基础总损失 - 包含新的控制正则化项
            total_loss = (
                self.goal_weight * goal_loss +
                self.safety_weight * total_safety_loss +
                self.control_weight * control_effort +
                self.acceleration_loss_weight * acceleration_loss +  # 新增：加速度损失
                self.jerk_weight * jerk_loss +  # 抖动损失（连续动作差异）
                self.progress_weight * progress_loss
            )
            
            # 🚀 CORE INNOVATION: 安全门控Alpha正则化
            # 检查是否启用我们的创新逻辑
            alpha_reg_applied = False
            if self.config.get("use_safety_gated_alpha_reg", False):
                # 如果启用了门控机制
                # 只有当平均安全损失低于我们设定的阈值时，才添加alpha正则化惩罚
                safety_threshold = self.config.get("safety_loss_threshold", 0.01)
                if total_safety_loss.item() < safety_threshold:
                    total_loss += self.alpha_reg_weight * alpha_regularization_loss
                    alpha_reg_applied = True
                    # 记录惩罚被激活了
                    if hasattr(self, 'writer') and self.writer:
                        self.writer.add_scalar('innovation/alpha_reg_activated', 1.0, self.global_step)
                        self.writer.add_scalar('innovation/safety_loss_vs_threshold', total_safety_loss.item(), self.global_step)
                else:
                    # 如果安全损失较高，则不添加惩罚，让alpha自由增大以确保安全
                    if hasattr(self, 'writer') and self.writer:
                        self.writer.add_scalar('innovation/alpha_reg_activated', 0.0, self.global_step)
                        self.writer.add_scalar('innovation/safety_loss_vs_threshold', total_safety_loss.item(), self.global_step)
            else:
                # 如果没有启用，则使用常规方式（总是添加惩罚）
                total_loss += self.alpha_reg_weight * alpha_regularization_loss
                alpha_reg_applied = True
                
            # 🚀 CORE INNOVATION: 动态安全裕度正则化损失
            margin_regularization_loss = torch.tensor(0.0, device=self.device)
            if self.config.get("use_adaptive_margin", False) and hasattr(self, 'trajectory_margins') and self.trajectory_margins:
                # 实现核心约束逻辑：
                # 1. 基础惩罚：我们不希望裕度太小，所以惩罚 (最大裕度 - 当前裕度)，鼓励它变大。
                # 2. 安全加权：当安全损失很高时，这种惩罚应该被放大。
                
                # 堆叠所有轨迹的动态裕度
                stacked_margins = torch.stack(self.trajectory_margins)
                
                # 从 detached 的 safety_loss 获取权重
                safety_weighting = total_safety_loss.detach() + 0.1
                
                # 计算裕度损失：鼓励更大的裕度，特别是在不安全时
                max_margin = self.config.get("max_safety_margin", 0.4)
                margin_cost = max_margin - stacked_margins
                
                margin_regularization_loss = torch.mean(safety_weighting * torch.mean(margin_cost))
                
                # 将其加入总损失
                margin_reg_weight = self.config.get("margin_reg_weight", 0.0)
                total_loss += margin_reg_weight * margin_regularization_loss
            
            # 通过整个计算图反向传播损失
            # 始终使用retain_graph=True进行BPTT以防止计算图问题
            total_loss.backward(retain_graph=True)
            
            # 剪辑梯度以防止梯度爆炸
            parameters = list(self.policy_network.parameters())
            if self.cbf_network is not None:
                parameters += list(self.cbf_network.parameters())
                
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
            
            # 更新参数
            self.optimizer.step()
            
            # 如果启用了调度器，则更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 计算日志指标 - 包含新的加速度损失
            metrics = {
                "train/total_loss": total_loss.item(),
                "train/goal_loss": goal_loss.item(),
                "train/safety_loss": total_safety_loss.item(),
                "train/control_loss": control_effort.item(),
                "train/acceleration_loss": acceleration_loss.item(),  # 新增：加速度损失
                "train/jerk_loss": jerk_loss if isinstance(jerk_loss, float) else jerk_loss.item(),
                "train/alpha_reg_loss": alpha_regularization_loss.item(),
                "train/alpha_reg_applied": float(alpha_reg_applied),  # 新增：门控状态
                "train/margin_reg_loss": margin_regularization_loss.item(),  # 🚀 NEW: 裕度正则化损失
                "train/progress_loss": progress_loss if isinstance(progress_loss, float) else progress_loss.item(),
                "train/avg_alpha": torch.mean(stacked_alphas).item(),
                "train/lr": self.optimizer.param_groups[0]['lr'],
                "step": step,
            }
            
            # 🚀 CORE INNOVATION: 添加动态裕度相关指标
            if hasattr(self, 'trajectory_margins') and self.trajectory_margins:
                avg_margin = torch.mean(torch.stack(self.trajectory_margins)).item()
                metrics["train/avg_dynamic_margin"] = avg_margin
            
            # 记录指标
            if WANDB_AVAILABLE:
                wandb.log(metrics)
            
            # 评估和模型保存
            if (step + 1) % self.eval_interval == 0:
                eval_metrics = self.evaluate()
                if WANDB_AVAILABLE:
                    wandb.log(eval_metrics)
                
                # 打印进度
                time_elapsed = time.time() - start_time
                print(f"\nStep {step+1}/{self.training_steps}, Time: {time_elapsed:.2f}s")
                print(f"  Total Loss: {total_loss.item():.4f}")
                print(f"  Goal Loss: {goal_loss.item():.4f}")
                print(f"  Safety Loss: {total_safety_loss.item():.4f}")
                print(f"  Control Loss: {control_effort.item():.4f}")
                print(f"  Acceleration Loss: {acceleration_loss.item():.4f}")  # 新增
                print(f"  Jerk Loss: {jerk_loss if isinstance(jerk_loss, float) else jerk_loss.item():.4f}")
                print(f"  Alpha Reg Loss: {alpha_regularization_loss.item():.4f}")
                print(f"  Alpha Reg Applied: {'Yes' if alpha_reg_applied else 'No'}")  # 新增：门控状态
                print(f"  Margin Reg Loss: {margin_regularization_loss.item():.4f}")  # 🚀 NEW: 裕度正则化损失
                print(f"  Progress Loss: {progress_loss if isinstance(progress_loss, float) else progress_loss.item():.4f}")
                print(f"  Avg Alpha: {torch.mean(stacked_alphas).item():.4f}")
                # 🚀 CORE INNOVATION: 显示动态裕度信息
                if hasattr(self, 'trajectory_margins') and self.trajectory_margins:
                    avg_margin = torch.mean(torch.stack(self.trajectory_margins)).item()
                    print(f"  Avg Dynamic Margin: {avg_margin:.4f}")
                print(f"  Evaluation Success Rate: {eval_metrics['eval/success_rate']:.2f}")
                print(f"  Evaluation Collision Rate: {eval_metrics['eval/collision_rate']:.2f}")
            
            # 保存模型
            if (step + 1) % self.save_interval == 0:
                self.save_models(step + 1)
            
            pbar.update(1)
            
            # 🚀 CORE INNOVATION: 清理轨迹裕度列表以准备下一个训练步骤
            if hasattr(self, 'trajectory_margins'):
                self.trajectory_margins = []
            # 🛡️ PROBABILISTIC SAFETY SHIELD: 清理安全防护罩相关数据
            if hasattr(self, 'trajectory_alpha_safety'):
                self.trajectory_alpha_safety = []
            if hasattr(self, 'trajectory_collisions'):
                self.trajectory_collisions = []
        
        pbar.close()
        print("Training completed.")
        
        # 保存最终模型
        self.save_models(self.training_steps)
        
        # 最终评估
        final_metrics = self.evaluate(num_episodes=20)
        print("\nFinal Evaluation Results:")
        print(f"  Success Rate: {final_metrics['eval/success_rate']:.2f}")
        print(f"  Collision Rate: {final_metrics['eval/collision_rate']:.2f}")
        print(f"  Avg Goal Distance: {final_metrics['eval/avg_goal_distance']:.4f}")
        
        return final_metrics
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        评估当前策略和CBF网络。
        
        参数:
            num_episodes: 评估的剧集数量
            
        返回:
            评估指标字典
        """
        success_count = 0
        collision_count = 0
        avg_goal_distance = 0
        avg_min_cbf = float('inf')
        
        # 将网络和环境设置为评估模式
        self.policy_network.eval()
        if self.cbf_network is not None:
            self.cbf_network.eval()
        
        # 将环境设置为评估模式（禁用梯度衰减）
        if hasattr(self.env, 'eval'):
            self.env.eval()
        
        for _ in range(num_episodes):
            # 初始化场景
            state = self.initialize_scenario()
            
            # 运行剧集，不跟踪梯度
            with torch.no_grad():
                # 重置环境
                current_state = state
                
                # 运行前向仿真
                for _ in range(self.eval_horizon):
                    # 获取观测值
                    observations = self.env.get_observation(current_state)
                    
                    # 在将观测值传递给网络之前，将观测值移动到正确的设备（CPU或GPU）
                    observations = observations.to(self.device)
                    
                    # 如果提供了CBF网络，则获取CBF值
                    if self.cbf_network is not None:
                        cbf_values = self.cbf_network.barrier_function(current_state)
                        min_cbf_val = cbf_values.min().item()
                        avg_min_cbf = min(avg_min_cbf, min_cbf_val)
                    
                    # 从策略网络获取动作、alpha和动态裕度
                    actions, alpha, raw_dynamic_margins = self.policy_network(observations)
                    
                    # 🚀 CORE INNOVATION: 处理动态安全裕度（评估时）
                    dynamic_margins = None
                    if raw_dynamic_margins is not None and self.config.get("use_adaptive_margin", False):
                        min_margin = self.config.get("min_safety_margin", 0.15)
                        max_margin = self.config.get("max_safety_margin", 0.4)
                        # 关键修复：确保dynamic_margins张量在正确的设备上（评估时）
                        dynamic_margins = (min_margin + raw_dynamic_margins * (max_margin - min_margin)).to(self.device)
                    
                    # 处理alpha为None的情况（固定alpha配置）
                    if alpha is None:
                        # 使用环境默认的alpha值
                        batch_size, num_agents = actions.shape[:2]
                        alpha = torch.full((batch_size, num_agents, 1), 
                                         self.env.cbf_alpha, 
                                         device=self.device, 
                                         dtype=actions.dtype)
                    
                    # 在环境中使用动态alpha进行一步
                    step_result = self.env.step(current_state, actions, alpha)
                    next_state = step_result.next_state
                    
                    # 检查是否发生碰撞
                    if torch.any(step_result.cost > 0):
                        collision_count += 1
                        break
                    
                    # 更新状态
                    current_state = next_state
                
                # 检查是否达到目标（使用环境的目标距离）
                goal_distances = self.env.get_goal_distance(current_state)
                avg_distance = goal_distances.mean().item()
                avg_goal_distance += avg_distance
                
                # 如果所有代理都接近其目标，则计为成功
                if torch.all(goal_distances < self.env.agent_radius * 2):
                    success_count += 1
        
        # 将网络和环境设置回训练模式
        self.policy_network.train()
        if self.cbf_network is not None:
            self.cbf_network.train()
            
        # 将环境设置回训练模式
        if hasattr(self.env, 'train'):
            self.env.train()
        
        # 计算平均指标
        success_rate = success_count / num_episodes
        collision_rate = collision_count / num_episodes
        avg_goal_distance /= num_episodes
        
        # 准备评估指标
        metrics = {
            "eval/success_rate": success_rate,
            "eval/collision_rate": collision_rate,
            "eval/avg_goal_distance": avg_goal_distance,
        }
        
        # 如果提供了CBF网络，则添加CBF指标
        if self.cbf_network is not None and avg_min_cbf != float('inf'):
            metrics["eval/avg_min_cbf"] = avg_min_cbf
        
        return metrics
    
    def evaluate_with_logging(self, num_episodes: int = 1, log_episodes: bool = True) -> Dict[str, float]:
        """
        评估当前策略并记录详细的episode数据，计算全面的KPIs。
        
        参数:
            num_episodes: 评估的剧集数量
            log_episodes: 是否记录episode数据到文件
            
        返回:
            包含详细KPIs的评估指标字典
        """
        success_count = 0
        collision_count = 0
        timeout_count = 0
        avg_goal_distance = 0
        avg_min_cbf = float('inf')
        episode_files = []
        
        # 🏆 **NEW: 冠军评估体系 - KPI统计聚合器**
        stats_aggregator = {
            'success_episodes': [],      # 成功的episode详细数据
            'collision_episodes': [],    # 碰撞的episode详细数据  
            'timeout_episodes': [],      # 超时的episode详细数据
            'all_episodes': []           # 所有episode的基础统计
        }
        
        # 将网络和环境设置为评估模式
        self.policy_network.eval()
        if self.cbf_network is not None:
            self.cbf_network.eval()
        
        # 将环境设置为评估模式（禁用梯度衰减）
        if hasattr(self.env, 'eval'):
            self.env.eval()
        
        for episode_idx in range(num_episodes):
            print(f"\n🎯 运行评估 Episode {episode_idx + 1}/{num_episodes}")
            
            # 初始化场景
            state = self.initialize_scenario()
            
            # 初始化数据记录
            episode_logger = None
            if log_episodes and (self.episode_logger is not None or num_episodes == 1):
                if self.episode_logger is not None:
                    episode_logger = self.episode_logger
                else:
                    # 为单次评估创建临时记录器
                    log_dir = os.path.join(self.log_dir, 'eval_logs')
                    episode_logger = EpisodeLogger(log_dir=log_dir, prefix="eval_episode")
                
                # 开始记录episode
                episode_id = episode_logger.start_episode(
                    batch_size=state.batch_size,
                    n_agents=state.n_agents,
                    obstacles=state.obstacles,
                    goals=state.goals,
                    safety_radius=getattr(self.env, 'agent_radius', 0.2),
                    area_size=getattr(self.env, 'area_size', 2.0)
                )
            
            # 运行剧集，不跟踪梯度
            episode_status = "TIMEOUT"
            step_count = 0
            
            with torch.no_grad():
                # 重置环境
                current_state = state
                
                # 运行前向仿真
                for step in range(self.eval_horizon):
                    step_count = step + 1
                    
                    # 获取观测值
                    observations = self.env.get_observation(current_state)
                    
                    # 确保观测在正确设备上
                    if hasattr(observations, 'to'):
                        observations = observations.to(self.get_device())
                    
                    # 从策略网络获取动作、alpha和动态安全裕度
                    # 🚀 修复：策略网络现在返回三个值 (actions, alpha, dynamic_margins)
                    policy_output = self.policy_network(observations)
                    if len(policy_output) == 3:
                        # 新的自适应裕度模型：返回 (actions, alpha, dynamic_margins)
                        actions, alpha, dynamic_margins = policy_output
                    else:
                        # 旧模型：只返回 (actions, alpha)
                        actions, alpha = policy_output
                        dynamic_margins = None
                    
                    # 如果提供了CBF网络，则获取CBF值
                    h_values = None
                    if self.cbf_network is not None:
                        # CBF网络需要state和可选的dynamic_margins作为输入
                        # 🚀 修复：传递dynamic_margins以支持自适应安全裕度
                        if dynamic_margins is not None:
                            # 确保dynamic_margins在正确的设备上
                            dynamic_margins = dynamic_margins.to(self.get_device())
                        h_values = self.cbf_network.barrier_function(current_state, dynamic_margins)
                        min_cbf_val = h_values.min().item()
                        avg_min_cbf = min(avg_min_cbf, min_cbf_val)
                    
                    # 处理alpha为None的情况（固定alpha配置）
                    if alpha is None:
                        # 使用环境默认的alpha值
                        batch_size, num_agents = actions.shape[:2]
                        alpha = torch.full((batch_size, num_agents, 1), 
                                         self.cbf_alpha, 
                                         device=self.get_device(), 
                                         dtype=actions.dtype)
                    
                    # 在环境中使用动态alpha进行一步
                    step_result = self.env.step(current_state, actions, alpha)
                    next_state = step_result.next_state
                    
                    # 记录step数据
                    if episode_logger is not None:
                        # 计算最小距离和目标距离
                        min_distances = None
                        goal_distances = None
                        
                        if hasattr(current_state, 'obstacles') and current_state.obstacles is not None:
                            obstacles_np = current_state.obstacles.detach().cpu().numpy()
                            positions_np = current_state.positions.detach().cpu().numpy()
                            min_distances = torch.from_numpy(
                                compute_min_distances_to_obstacles(positions_np, obstacles_np)
                            )
                        
                        if hasattr(current_state, 'goals'):
                            positions_np = current_state.positions.detach().cpu().numpy()
                            goals_np = current_state.goals.detach().cpu().numpy()
                            goal_distances = torch.from_numpy(
                                compute_goal_distances(positions_np, goals_np)
                            )
                        
                        # 记录数据
                        episode_logger.log_step(
                            positions=current_state.positions,
                            velocities=current_state.velocities,
                            actions=step_result.info.get('action', actions),
                            raw_actions=step_result.info.get('raw_action', actions),
                            alpha_values=alpha,
                            h_values=h_values,
                            min_distances=min_distances,
                            goal_distances=goal_distances,
                            rewards=step_result.reward,
                            costs=step_result.cost
                        )
                    
                    # 检查是否发生碰撞
                    if torch.any(step_result.cost > 0):
                        collision_count += 1
                        episode_status = "COLLISION"
                        print(f"   ❌ 碰撞发生在第 {step + 1} 步")
                        break
                    
                    # 更新状态
                    current_state = next_state
                    
                    # 检查是否达到目标
                    goal_distances = self.env.get_goal_distance(current_state)
                    if torch.all(goal_distances < getattr(self.env, 'agent_radius', 0.2) * 2):
                        success_count += 1
                        episode_status = "SUCCESS"
                        print(f"   ✅ 成功完成任务在第 {step + 1} 步")
                        break
                
                # 计算最终目标距离
                final_goal_distances = self.env.get_goal_distance(current_state)
                avg_distance = final_goal_distances.mean().item()
                avg_goal_distance += avg_distance
                
                if episode_status == "TIMEOUT":
                    timeout_count += 1
                    print(f"   ⏰ Episode超时 ({self.eval_horizon} 步)")
            
            # 结束episode记录
            if episode_logger is not None:
                filename = episode_logger.end_episode(episode_status)
                episode_files.append(filename)
                print(f"   💾 Episode数据已保存: {filename}")
            
            # 🏆 **NEW: 收集episode统计数据到KPI聚合器**
            episode_stats = self._collect_episode_kpis(
                episode_status=episode_status,
                step_count=step_count,
                final_goal_distance=avg_distance,
                min_cbf_value=avg_min_cbf if avg_min_cbf != float('inf') else None,
                episode_file=episode_files[-1] if episode_files else None
            )
            
            # 按状态分类存储episode数据  
            stats_aggregator['all_episodes'].append(episode_stats)
            if episode_status == "SUCCESS":
                stats_aggregator['success_episodes'].append(episode_stats)
            elif episode_status == "COLLISION":
                stats_aggregator['collision_episodes'].append(episode_stats)
            elif episode_status == "TIMEOUT":
                stats_aggregator['timeout_episodes'].append(episode_stats)
        
        # 将网络和环境设置回训练模式
        self.policy_network.train()
        if self.cbf_network is not None:
            self.cbf_network.train()
            
        # 将环境设置回训练模式
        if hasattr(self.env, 'train'):
            self.env.train()
        
        # 🏆 **NEW: 计算详细的KPI指标**
        kpi_metrics = self._compute_champion_kpis(stats_aggregator, num_episodes)
        
        # 传统指标 (向后兼容)
        success_rate = success_count / num_episodes
        collision_rate = collision_count / num_episodes
        timeout_rate = timeout_count / num_episodes
        avg_goal_distance /= num_episodes
        
        # 准备评估指标 (结合传统和新KPI)
        metrics = {
            "eval/success_rate": success_rate,
            "eval/collision_rate": collision_rate,
            "eval/timeout_rate": timeout_rate,
            "eval/avg_goal_distance": avg_goal_distance,
            "eval/total_episodes": num_episodes,
            "eval/avg_episode_length": step_count,
        }
        
        if self.cbf_network is not None and avg_min_cbf != float('inf'):
            metrics["eval/avg_min_cbf"] = avg_min_cbf
        
        # 🏆 **NEW: 添加详细KPI到返回结果**
        metrics.update(kpi_metrics)
        
        # 添加episode文件路径信息
        if episode_files:
            metrics["eval/episode_files"] = episode_files
            
            # 🏆 **NEW: 显示冠军级别的KPI总结**
            self._print_champion_summary(kpi_metrics, episode_files)
        
        return metrics
    
    def _calculate_progress_loss(self, trajectory_states) -> torch.Tensor:
        """
        計算進度損失（基於潜力的奖励塑形）。
        
        參數:
            trajectory_states: 軌跡狀態列表
            
        返回:
            進度損失張量
        """
        if len(trajectory_states) < 2:
            return torch.tensor(0.0, device=self.get_device())
        
        # 計算初始和最終目標距離
        initial_state = trajectory_states[0]
        final_state = trajectory_states[-1]
        
        initial_distances = self.env.get_goal_distance(initial_state)
        final_distances = self.env.get_goal_distance(final_state)
        
        # 進度 = 初始距離 - 最終距離（正值表示朝目標前進）
        progress = initial_distances - final_distances
        
        # 負進度表示遠離目標，應該被懲罰
        progress_loss = -torch.mean(progress)
        
        return progress_loss
    
    def get_device(self) -> torch.device:
        """获取设备信息（CPU或CUDA）。"""
        if hasattr(self.policy_network, 'parameters'):
            params = list(self.policy_network.parameters())
            if params:
                return params[0].device
        return torch.device('cpu')
    
    def save_models(self, step: int) -> None:
        """
        保存策略和CBF网络模型。
        
        参数:
            step: 当前训练步数
        """
        step_dir = os.path.join(self.model_dir, str(step))
        if not os.path.exists(step_dir):
            os.makedirs(step_dir)
        
        # 保存策略网络
        policy_path = os.path.join(step_dir, "policy.pt")
        torch.save(self.policy_network.state_dict(), policy_path)
        
        # 如果提供了CBF网络，则保存
        if self.cbf_network is not None:
            cbf_path = os.path.join(step_dir, "cbf.pt")
            torch.save(self.cbf_network.state_dict(), cbf_path)
        
        # 保存优化器状态
        optim_path = os.path.join(step_dir, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optim_path)
        
        # 保存配置
        config_path = os.path.join(step_dir, "config.pt")
        torch.save(self.config, config_path)
        
        print(f"Models saved at step {step}")
    
    def load_models(self, step: int) -> None:
        """
        加载策略和CBF网络模型。
        
        参数:
            step: 要加载的训练步数
        """
        step_dir = os.path.join(self.model_dir, str(step))
        
        if not os.path.exists(step_dir):
            raise FileNotFoundError(f"No saved models found at step {step}")
        
        # 加载策略网络
        policy_path = os.path.join(step_dir, "policy.pt")
        if os.path.exists(policy_path):
            self.policy_network.load_state_dict(torch.load(policy_path))
            print(f"Policy network loaded from {policy_path}")
        
        # 如果提供了CBF网络，则加载
        if self.cbf_network is not None:
            cbf_path = os.path.join(step_dir, "cbf.pt")
            if os.path.exists(cbf_path):
                self.cbf_network.load_state_dict(torch.load(cbf_path))
                print(f"CBF network loaded from {cbf_path}")
        
        # 加载优化器状态
        optim_path = os.path.join(step_dir, "optimizer.pt")
        if os.path.exists(optim_path):
            self.optimizer.load_state_dict(torch.load(optim_path))
            print(f"Optimizer state loaded from {optim_path}")
    
    # 🏆 **NEW: 冠军评估体系的KPI支持方法**
    
    def _collect_episode_kpis(self, episode_status: str, step_count: int, 
                            final_goal_distance: float, min_cbf_value: float = None,
                            episode_file: str = None) -> Dict[str, float]:
        """
        为单个episode收集详细的KPI统计数据。
        
        参数:
            episode_status: episode结束状态 ("SUCCESS", "COLLISION", "TIMEOUT")
            step_count: episode总步数
            final_goal_distance: 最终目标距离
            min_cbf_value: 最小CBF值
            episode_file: episode数据文件路径
            
        返回:
            包含该episode所有KPI的字典
        """
        import numpy as np
        
        stats = {
            'status': episode_status,
            'completion_time': step_count,
            'final_goal_distance': final_goal_distance,
            'success': 1 if episode_status == "SUCCESS" else 0,
            'collision': 1 if episode_status == "COLLISION" else 0,
            'timeout': 1 if episode_status == "TIMEOUT" else 0,
            'episode_file': episode_file
        }
        
        if min_cbf_value is not None:
            stats['min_safety_distance'] = min_cbf_value
            
        # 如果有episode文件，尝试加载更详细的统计
        if episode_file and os.path.exists(episode_file):
            try:
                from gcbfplus.utils.episode_logger import load_episode_data
                episode_data = load_episode_data(episode_file)
                
                # 计算平均加速度和抖动 (jerk)
                if 'actions' in episode_data and episode_data['actions'] is not None:
                    actions = episode_data['actions']
                    if len(actions) > 1:
                        # 计算加速度 (action differences)
                        accelerations = np.diff(actions, axis=0)
                        avg_acceleration = np.mean(np.linalg.norm(accelerations, axis=-1))
                        stats['avg_acceleration'] = float(avg_acceleration)
                        
                        # 计算抖动 (acceleration differences)
                        if len(accelerations) > 1:
                            jerks = np.diff(accelerations, axis=0)
                            avg_jerk = np.mean(np.linalg.norm(jerks, axis=-1))
                            stats['avg_jerk'] = float(avg_jerk)
                
                # 计算最小安全距离统计
                if 'min_distances' in episode_data and episode_data['min_distances'] is not None:
                    min_distances = episode_data['min_distances']
                    if len(min_distances) > 0:
                        stats['min_safety_distance'] = float(np.min(min_distances))
                        stats['avg_safety_distance'] = float(np.mean(min_distances))
                        stats['safety_violations'] = int(np.sum(min_distances < 0.1))  # 违规次数
                        
            except Exception as e:
                print(f"Warning: Failed to extract detailed stats from {episode_file}: {e}")
        
        return stats
    
    def _compute_champion_kpis(self, stats_aggregator: Dict, num_episodes: int) -> Dict[str, float]:
        """
        基于聚合的统计数据计算最终的冠军级别KPIs。
        
        参数:
            stats_aggregator: 包含所有episode统计的聚合器
            num_episodes: 总episode数
            
        返回:
            包含所有冠军KPI的字典
        """
        import numpy as np
        
        kpis = {}
        
        # 基础成功率统计
        success_episodes = stats_aggregator['success_episodes']
        collision_episodes = stats_aggregator['collision_episodes']
        timeout_episodes = stats_aggregator['timeout_episodes']
        all_episodes = stats_aggregator['all_episodes']
        
        kpis['champion/success_rate'] = len(success_episodes) / num_episodes
        kpis['champion/collision_rate'] = len(collision_episodes) / num_episodes
        kpis['champion/timeout_rate'] = len(timeout_episodes) / num_episodes
        
        # 🏆 成功案例的详细KPIs
        if success_episodes:
            completion_times = [ep['completion_time'] for ep in success_episodes]
            kpis['champion/avg_completion_time_success'] = np.mean(completion_times)
            kpis['champion/std_completion_time_success'] = np.std(completion_times)
            kpis['champion/min_completion_time'] = np.min(completion_times)
            kpis['champion/max_completion_time'] = np.max(completion_times)
            
            # 抖动统计 (仅成功案例)
            jerks = [ep.get('avg_jerk', 0) for ep in success_episodes if 'avg_jerk' in ep]
            if jerks:
                kpis['champion/avg_jerk_success'] = np.mean(jerks)
                kpis['champion/std_jerk_success'] = np.std(jerks)
            
            # 安全距离统计 (仅成功案例)
            safety_dists = [ep.get('min_safety_distance', float('inf')) for ep in success_episodes if 'min_safety_distance' in ep]
            if safety_dists:
                kpis['champion/avg_min_safety_distance_success'] = np.mean(safety_dists)
                kpis['champion/std_min_safety_distance_success'] = np.std(safety_dists)
                
            # 安全违规统计
            violations = [ep.get('safety_violations', 0) for ep in success_episodes if 'safety_violations' in ep]
            if violations:
                kpis['champion/avg_safety_violations_success'] = np.mean(violations)
        
        # 🥇 寻找最佳episode (成功案例中最短时间)
        if success_episodes:
            best_episode = min(success_episodes, key=lambda x: x['completion_time'])
            kpis['champion/best_episode_file'] = best_episode.get('episode_file', 'unknown')
            kpis['champion/best_completion_time'] = best_episode['completion_time']
            kpis['champion/best_episode_jerk'] = best_episode.get('avg_jerk', 0)
            kpis['champion/best_episode_safety'] = best_episode.get('min_safety_distance', float('inf'))
        
        # 🔥 整体鲁棒性指标
        all_completion_times = [ep['completion_time'] for ep in all_episodes]
        kpis['champion/avg_episode_length'] = np.mean(all_completion_times)
        kpis['champion/robustness_score'] = kpis['champion/success_rate'] * (1 - kpis['champion/collision_rate'])
        
        return kpis
    
    def _print_champion_summary(self, kpi_metrics: Dict[str, float], episode_files: list) -> None:
        """
        打印冠军级别的KPI总结。
        
        参数:
            kpi_metrics: 计算好的KPI指标
            episode_files: episode文件列表
        """
        print(f"\n🏆 ═══════════════════════════════════════════════════")
        print(f"🏆           冠军评估体系 - KPI总结报告           ")  
        print(f"🏆 ═══════════════════════════════════════════════════")
        
        # 基础性能指标
        print(f"📊 基础性能:")
        print(f"   ✅ 成功率: {kpi_metrics.get('champion/success_rate', 0):.1%}")
        print(f"   ❌ 碰撞率: {kpi_metrics.get('champion/collision_rate', 0):.1%}")
        print(f"   ⏰ 超时率: {kpi_metrics.get('champion/timeout_rate', 0):.1%}")
        print(f"   🛡️ 鲁棒性得分: {kpi_metrics.get('champion/robustness_score', 0):.3f}")
        
        # 成功案例分析
        if 'champion/avg_completion_time_success' in kpi_metrics:
            print(f"\n🎯 成功案例分析:")
            print(f"   ⏱️ 平均完成时间: {kpi_metrics['champion/avg_completion_time_success']:.1f} ± {kpi_metrics.get('champion/std_completion_time_success', 0):.1f} 步")
            print(f"   🚀 最佳完成时间: {kpi_metrics.get('champion/min_completion_time', 0):.0f} 步")
            print(f"   🎭 最差完成时间: {kpi_metrics.get('champion/max_completion_time', 0):.0f} 步")
            
            if 'champion/avg_jerk_success' in kpi_metrics:
                print(f"   📈 平均抖动: {kpi_metrics['champion/avg_jerk_success']:.4f} ± {kpi_metrics.get('champion/std_jerk_success', 0):.4f}")
                
            if 'champion/avg_min_safety_distance_success' in kpi_metrics:
                print(f"   🛡️ 平均安全距离: {kpi_metrics['champion/avg_min_safety_distance_success']:.3f} ± {kpi_metrics.get('champion/std_min_safety_distance_success', 0):.3f}")
        
        # 最佳episode信息
        if 'champion/best_episode_file' in kpi_metrics:
            print(f"\n🥇 冠军Episode:")
            print(f"   📁 文件: {os.path.basename(kpi_metrics['champion/best_episode_file'])}")
            print(f"   ⏱️ 完成时间: {kpi_metrics.get('champion/best_completion_time', 0):.0f} 步")
            print(f"   📈 抖动值: {kpi_metrics.get('champion/best_episode_jerk', 0):.4f}")
            print(f"   🛡️ 安全距离: {kpi_metrics.get('champion/best_episode_safety', float('inf')):.3f}")
        
        print(f"\n💾 数据文件: {len(episode_files)} 个episode已保存")
        print(f"🏆 ═══════════════════════════════════════════════════")
        