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
        self.log_dir = self.config.get('log_dir', 'logs/bptt')
        self.run_name = self.config.get('run_name', 'BPTT_Run')
        self.num_agents = self.config.get('num_agents', 8)
        self.area_size = self.config.get('area_size', 1.0)
        
        # 训练参数
        self.training_steps = self.config.get('training_steps', 10000)
        self.eval_interval = self.config.get('eval_interval', 100)
        self.save_interval = self.config.get('save_interval', 1000)
        self.horizon_length = self.config.get('horizon_length', 50)
        self.eval_horizon = self.config.get('eval_horizon', 100)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)
        
        # 损失权重
        self.goal_weight = self.config.get('goal_weight', 1.0)
        self.safety_weight = self.config.get('safety_weight', 10.0)
        self.control_weight = self.config.get('control_weight', 0.1)
        self.jerk_weight = self.config.get('jerk_weight', 0.05)  # 加速度变化率权重
        self.alpha_reg_weight = self.config.get('alpha_reg_weight', 0.01)  # Alpha正则化权重
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
                actions, alpha = self.policy_network(observations)
                
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
                
                # 如果提供了CBF网络，则应用安全过滤
                if self.cbf_network is not None:
                    cbf_values = self.cbf_network(observations)
                    
                    # 根据CBF值计算安全损失
                    # 负值表示不安全状态
                    safety_loss = torch.mean(torch.relu(-cbf_values))
                    safety_losses.append(safety_loss)
                
                # 在环境中使用动态alpha进行一步
                step_result = self.env.step(current_state, actions, alpha)
                next_state = step_result.next_state
                rewards = step_result.reward
                costs = step_result.cost
                
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
            
            # 加速度变化率损失（加速度变化率）
            # 计算连续动作之间的差异
            jerk_loss = 0.0
            if len(trajectory_actions) > 1:
                action_diffs = []
                for i in range(1, len(trajectory_actions)):
                    action_diff = trajectory_actions[i] - trajectory_actions[i-1]
                    action_diffs.append(action_diff)
                if action_diffs:
                    stacked_diffs = torch.stack(action_diffs)
                    jerk_loss = torch.mean(stacked_diffs ** 2)
            
            # 安全损失
            if safety_losses:
                stacked_safety = torch.stack(safety_losses)
                total_safety_loss = torch.mean(stacked_safety)
            else:
                # 如果没有CBF网络，则使用环境成本
                stacked_costs = torch.stack(trajectory_costs)
                total_safety_loss = torch.mean(stacked_costs)
            
            # Alpha正则化损失（鼓励更小的alpha值以提高效率）
            stacked_alphas = torch.stack(trajectory_alphas)
            alpha_regularization_loss = torch.mean(stacked_alphas)
            
            # 计算总损失作为加权和
            total_loss = (
                self.goal_weight * goal_loss +
                self.safety_weight * total_safety_loss +
                self.control_weight * control_effort +
                self.jerk_weight * jerk_loss +  # 添加加速度变化率惩罚
                self.alpha_reg_weight * alpha_regularization_loss  # 添加Alpha正则化
            )
            
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
            
            # 计算日志指标
            metrics = {
                "train/total_loss": total_loss.item(),
                "train/goal_loss": goal_loss.item(),
                "train/safety_loss": total_safety_loss.item(),
                "train/control_loss": control_effort.item(),
                "train/jerk_loss": jerk_loss if isinstance(jerk_loss, float) else jerk_loss.item(),
                "train/alpha_reg_loss": alpha_regularization_loss.item(),
                "train/avg_alpha": torch.mean(stacked_alphas).item(),
                "train/lr": self.optimizer.param_groups[0]['lr'],
                "step": step,
            }
            
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
                print(f"  Jerk Loss: {jerk_loss if isinstance(jerk_loss, float) else jerk_loss.item():.4f}")
                print(f"  Alpha Reg Loss: {alpha_regularization_loss.item():.4f}")
                print(f"  Avg Alpha: {torch.mean(stacked_alphas).item():.4f}")
                print(f"  Evaluation Success Rate: {eval_metrics['eval/success_rate']:.2f}")
                print(f"  Evaluation Collision Rate: {eval_metrics['eval/collision_rate']:.2f}")
            
            # 保存模型
            if (step + 1) % self.save_interval == 0:
                self.save_models(step + 1)
            
            pbar.update(1)
        
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
                        cbf_values = self.cbf_network(observations)
                        min_cbf_val = cbf_values.min().item()
                        avg_min_cbf = min(avg_min_cbf, min_cbf_val)
                    
                    # 从策略网络获取动作和alpha
                    actions, alpha = self.policy_network(observations)
                    
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
        
        print(f"Models loaded from step {step}") 