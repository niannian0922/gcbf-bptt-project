import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union, Any


class PerceptionModule(nn.Module):
    """
    感知模块，用于处理传感器输入。
    
    该模块可配置为处理不同类型的输入：
    - 基于视觉的观测（使用CNN处理深度图像）
    - 密集向量观测（使用MLP处理状态向量）
    """
    
    def __init__(self, config: Dict):
        """
        初始化感知模块。
        
        参数:
            config: 配置字典，包含以下键值：
                视觉模式:
                - 'vision_enabled': 是否启用基于视觉的处理
                - 'input_channels': 输入通道数（深度图默认为1）
                - 'conv_channels': CNN通道大小列表
                - 'kernel_sizes': 每个卷积层的核大小列表
                - 'image_size': 输入图像大小（假设为正方形）
                
                状态模式:
                - 'input_dim': 输入维度大小
                - 'hidden_dim': 隐藏维度大小
                - 'num_layers': 隐藏层数量
                - 'activation': 激活函数名称
                - 'use_batch_norm': 是否使用批归一化
        """
        super(PerceptionModule, self).__init__()
        
        # 检查是否启用视觉模式
        self.vision_enabled = config.get('vision_enabled', False)
        hidden_dim = config.get('hidden_dim', 64)
        activation = config.get('activation', 'relu')
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.05)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        if self.vision_enabled:
            # 基于视觉的CNN处理
            input_channels = config.get('input_channels', 1)  # 深度图像
            conv_channels = config.get('conv_channels', [32, 64, 128])
            kernel_sizes = config.get('kernel_sizes', [5, 3, 3])
            image_size = config.get('image_size', 64)
            
            # 构建CNN层
            cnn_layers = []
            in_channels = input_channels
            
            for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
                cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, 
                                           stride=2, padding=kernel_size//2))
                cnn_layers.append(nn.BatchNorm2d(out_channels))
                cnn_layers.append(self.activation)
                in_channels = out_channels
            
            self.cnn = nn.Sequential(*cnn_layers)
            
            # 计算卷积后的尺寸
            # 每个步长为2的卷积层将空间维度减半
            final_size = image_size // (2 ** len(conv_channels))
            cnn_output_size = conv_channels[-1] * final_size * final_size
            
            # 最终MLP获得期望的输出维度
            self.cnn_projection = nn.Sequential(
                nn.Linear(cnn_output_size, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            self.output_dim = hidden_dim
            
        else:
            # 基于状态的MLP处理（原始实现）
            input_dim = config.get('input_dim', 9)
            num_layers = config.get('num_layers', 2)
            use_batch_norm = config.get('use_batch_norm', False)
            
            # 构建MLP层
            layers = []
            
            # 输入层
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            
            # 隐藏层
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(self.activation)
            
            self.mlp = nn.Sequential(*layers)
            self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        通过感知模块处理输入。
        
        参数:
            x: 输入张量 
               - 视觉模式: [batch_size, n_agents, channels, height, width]
               - 状态模式: [batch_size, n_agents, input_dim] 或 [batch_size, input_dim]
            
        Returns:
            处理后的特征 [batch_size, output_dim] 或 [batch_size, n_agents, output_dim]
        """
        original_shape = x.shape
        
        if self.vision_enabled:
            # 处理视觉输入: [batch_size, n_agents, channels, height, width]
            if len(original_shape) == 5:
                batch_size, n_agents, channels, height, width = original_shape
                
                # 重塑为 [batch_size * n_agents, channels, height, width]
                x_flat = x.reshape(batch_size * n_agents, channels, height, width)
                
                # 通过CNN处理
                cnn_features = self.cnn(x_flat)  # [batch_size * n_agents, final_channels, final_h, final_w]
                
                # 展平空间维度
                cnn_flat = cnn_features.view(cnn_features.size(0), -1)  # [batch_size * n_agents, flat_size]
                
                # 投影到期望的输出维度
                features = self.cnn_projection(cnn_flat)  # [batch_size * n_agents, output_dim]
                
                # 重塑回 [batch_size, n_agents, output_dim]
                return features.view(batch_size, n_agents, -1)
            else:
                raise ValueError(f"视觉模式期望5D输入 [batch, agents, channels, height, width]，得到 {original_shape}")
        
        else:
            # 处理基于状态的输入（原始实现）
            if len(original_shape) == 3:
                batch_size, n_agents, input_dim = original_shape
                
                # 重塑为 [batch_size * n_agents, input_dim]
                x_flat = x.reshape(batch_size * n_agents, input_dim)
                
                # 通过MLP处理
                features = self.mlp(x_flat)
                
                # 重塑回 [batch_size, n_agents, output_dim]
                return features.view(batch_size, n_agents, -1)
            else:
                # 简单批处理 [batch_size, input_dim]
                return self.mlp(x)


class MemoryModule(nn.Module):
    """
    记忆模块，用于维护时序状态信息。
    
    使用GRU网络维护智能体的内部状态，支持多智能体场景。
    可选择是否在不同时间步之间保持记忆状态。
    """
    
    def __init__(self, config: Dict):
        """
        初始化记忆模块。
        
        参数:
            config: 包含记忆模块参数的字典
                必需键值:
                - 'input_dim': 输入维度
                - 'hidden_dim': 隐藏状态维度
                可选键值:
                - 'num_layers': GRU层数（默认1）
                - 'dropout': dropout率（默认0.0）
        """
        super(MemoryModule, self).__init__()
        
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config.get('num_layers', 1)
        self.dropout = config.get('dropout', 0.0)
        
        # 创建GRU单元
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True
        )
        
        self.hidden_state = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：处理输入并更新内部状态。
        
        参数:
            x: 输入张量，形状为[batch_size, n_agents, input_dim]或[batch_size, input_dim]
               
        返回:
            输出张量，形状为[batch_size, n_agents, hidden_dim]或[batch_size, hidden_dim]
        """
        if x.dim() == 3:  # 多智能体情况
            batch_size, n_agents, input_dim = x.shape
            
            # 处理多智能体观测
            # 重塑为 [batch_size * n_agents, 1, input_dim]，因为GRU期望序列长度维度
            x = x.view(batch_size * n_agents, 1, input_dim)
            
            # 初始化或重置隐藏状态（如果需要）
            if self.hidden_state is None or self.hidden_state.size(1) != batch_size * n_agents:
                self.hidden_state = torch.zeros(self.num_layers, batch_size * n_agents, self.hidden_dim, 
                                               device=x.device, dtype=x.dtype)
            
            # 创建新的张量而不是原地修改
            if self.hidden_state.device != x.device:
                self.hidden_state = self.hidden_state.to(x.device)
            
            # 更新隐藏状态
            output, new_hidden = self.gru(x, self.hidden_state)
            
            # 存储新的隐藏状态（不破坏计算图）
            self.hidden_state = new_hidden.detach()
            
            # 重塑回 [batch_size, n_agents, hidden_dim]
            return output.view(batch_size, n_agents, self.hidden_dim)
        else:
            # 简单批处理
            batch_size, input_dim = x.shape
            x = x.view(batch_size, 1, input_dim)
            
            # 初始化或重置隐藏状态（如果需要）
            if self.hidden_state is None or self.hidden_state.size(1) != batch_size:
                self.hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim, 
                                               device=x.device, dtype=x.dtype)
            
            # 创建新的张量而不是原地修改
            if self.hidden_state.device != x.device:
                self.hidden_state = self.hidden_state.to(x.device)
            
            # 更新隐藏状态
            output, new_hidden = self.gru(x, self.hidden_state)
            
            # 存储新的隐藏状态（不破坏计算图）
            self.hidden_state = new_hidden.detach()
            
            return output.squeeze(1)  # 移除序列长度维度
    
    def reset(self) -> None:
        """重置记忆状态。"""
        self.hidden_state = None


class PolicyHeadModule(nn.Module):
    """
    策略头模块，用于生成动作。
    
    将特征转换为动作输出，可选择应用动作边界和其他变换。
    支持自适应安全边距（动态Alpha）预测。
    """
    
    def __init__(self, config: Dict):
        """
        初始化策略头模块。
        
        参数:
            config: 包含策略头参数的字典
                必需键值:
                - 'input_dim': 输入特征维度
                - 'output_dim': 动作输出维度
                可选键值:
                - 'hidden_dims': 隐藏层维度列表
                - 'activation': 激活函数名称
                - 'output_activation': 输出层激活函数
                - 'action_scale': 动作缩放因子
                - 'predict_alpha': 是否预测动态alpha（默认True）
                - 'alpha_hidden_dim': alpha网络隐藏层维度
        """
        super(PolicyHeadModule, self).__init__()
        
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.hidden_dims = config.get('hidden_dims', [256, 256])
        self.action_scale = config.get('action_scale', 1.0)
        
        # 激活函数
        activation_name = config.get('activation', 'relu')
        self.activation = getattr(nn, activation_name.capitalize())() if hasattr(nn, activation_name.capitalize()) else nn.ReLU()
        
        output_activation = config.get('output_activation', None)
        self.output_activation = getattr(nn, output_activation.capitalize())() if output_activation and hasattr(nn, output_activation.capitalize()) else None
        
        # 自适应安全边距配置
        self.predict_alpha = config.get('predict_alpha', True)
        
        # 构建动作预测MLP层
        self.action_layers = nn.ModuleList()
        
        # 动作的隐藏层
        layer_dims = [self.input_dim] + self.hidden_dims
        for i in range(len(layer_dims) - 1):
            self.action_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            self.action_layers.append(self.activation)
        
        # 动作输出层
        self.action_layers.append(nn.Linear(self.hidden_dims[-1] if self.hidden_dims else self.input_dim, self.output_dim))
        
        self.action_network = nn.Sequential(*self.action_layers)
        
        # 仅当predict_alpha为True时构建alpha预测MLP
        if self.predict_alpha:
            alpha_hidden_dim = config.get('alpha_hidden_dim', self.hidden_dims[0] // 2 if self.hidden_dims else 32)
            self.alpha_network = nn.Sequential(
                nn.Linear(self.input_dim, alpha_hidden_dim),
                self.activation,
                nn.Linear(alpha_hidden_dim, 1),  # 为每个智能体预测单个alpha
                nn.Softplus()  # 确保alpha > 0
            )
        else:
            self.alpha_network = None
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播：生成动作和可选的alpha值。
        
        参数:
            features: 输入特征，形状为[batch_size, n_agents, input_dim]或[batch_size, input_dim]
               
        返回:
            元组(actions, alpha):
            - actions: 动作张量
            - alpha: 动态alpha值（如果启用）或None
        """
        if features.dim() == 3:  # 多智能体情况
            batch_size, n_agents, input_dim = features.shape
            
            # 处理多智能体特征
            # 重塑为 [batch_size * n_agents, input_dim]
            features_flat = features.view(-1, input_dim)
            
            # 通过动作网络处理
            actions_flat = self.action_network(features_flat)
            
            # 应用输出激活函数
            if self.output_activation is not None:
                actions_flat = self.output_activation(actions_flat)
            
            # 缩放动作（如果需要）
            if self.action_scale != 1.0:
                actions_flat = actions_flat * self.action_scale
            
            # 重塑动作回 [batch_size, n_agents, -1]
            actions = actions_flat.view(batch_size, n_agents, -1)
            
            # 如果启用，通过alpha网络处理
            if self.alpha_network is not None:
                alpha_flat = self.alpha_network(features_flat)
                alpha = alpha_flat.view(batch_size, n_agents, 1)
            else:
                alpha = None
                
            return actions, alpha
        else:
            # 简单批处理
            actions = self.action_network(features)
            
            # 缩放动作（如果需要）
            if self.action_scale != 1.0:
                actions = actions * self.action_scale
            
            # 如果启用，通过alpha网络处理
            if self.alpha_network is not None:
                alpha = self.alpha_network(features)
            else:
                alpha = None
                
            return actions, alpha


class BPTTPolicy(nn.Module):
    """
    时序反向传播（BPTT）策略网络。
    
    结合感知、记忆和策略头模块，实现端到端的策略学习。
    支持多智能体场景和自适应安全边距机制。
    """
    
    def __init__(self, config: Dict):
        """
        初始化BPTT策略网络。
        
        参数:
            config: 包含策略网络完整配置的字典
        """
        super(BPTTPolicy, self).__init__()
        
        # 提取子配置
        perception_config = config.get('perception', {})
        memory_config = config.get('memory', {})
        policy_head_config = config.get('policy_head', {})
        
        # 创建感知模块
        self.perception = PerceptionModule(perception_config)
        
        # 基于感知输出更新记忆输入维度
        memory_config['input_dim'] = self.perception.output_dim
        
        # 创建记忆模块
        self.memory = MemoryModule(memory_config)
        
        # 基于记忆输出更新策略头输入维度
        policy_head_config['input_dim'] = self.memory.hidden_dim
        
        # 创建策略头模块
        self.policy_head = PolicyHeadModule(policy_head_config)
        
        # 存储配置
        self.config = config
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播：将观测转换为动作和可选的alpha值。
        
        参数:
            observations: 观测张量
               
        返回:
            元组(actions, alpha):
            - actions: 动作张量  
            - alpha: 动态alpha值（如果启用）或None
        """
        # 通过感知模块处理
        features = self.perception(observations)
        
        # 通过记忆模块处理
        memory_output = self.memory(features)
        
        # 通过策略头生成动作和alpha
        actions, alpha = self.policy_head(memory_output)
        
        return actions, alpha
    
    def reset(self) -> None:
        """重置策略的内部状态（例如记忆）。"""
        if hasattr(self, 'memory'):
            self.memory.reset()


class EnsemblePolicy(nn.Module):
    """
    集成策略网络。
    
    结合多个策略网络的输出，提供更稳定的动作预测。
    支持简单平均和加权平均两种集成方法。
    """
    
    def __init__(self, config: Dict):
        """
        初始化集成策略网络。
        
        参数:
            config: 包含集成策略配置的字典
                必需键值:
                - 'policies': 策略配置列表
                可选键值:
                - 'ensemble_method': 集成方法（'mean'或'weighted'）
                - 'num_policies': 策略数量
        """
        super(EnsemblePolicy, self).__init__()
        
        # 提取配置参数
        policies_config = config.get('policies', [])
        self.ensemble_method = config.get('ensemble_method', 'mean')
        self.num_policies = config.get('num_policies', len(policies_config))
        
        # 创建策略集成
        self.policies = nn.ModuleList()
        for policy_config in policies_config:
            policy = BPTTPolicy(policy_config)
            self.policies.append(policy)
        
        # 如果使用加权集成，创建权重参数
        if self.ensemble_method == 'weighted':
            self.ensemble_weights = nn.Parameter(torch.ones(self.num_policies))
            # 设备将在模型移动到设备时设置
        
        # 存储配置
        self.config = config
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播：通过集成策略生成动作和alpha值。
        
        参数:
            observations: 观测张量
               
        返回:
            元组(actions, alpha):
            - actions: 集成后的动作张量
            - alpha: 集成后的alpha值（如果启用）或None
        """
        # 从每个策略获取动作和alpha
        policy_outputs = []
        for policy in self.policies:
            actions, alpha = policy(observations)
            policy_outputs.append((actions, alpha))
        
        # 分离动作和alpha
        actions_list = [output[0] for output in policy_outputs]
        alphas_list = [output[1] for output in policy_outputs if output[1] is not None]
        
        # 堆叠以便组合 [num_policies, batch_size, action_dim/1]
        stacked_actions = torch.stack(actions_list, dim=0)
        stacked_alphas = torch.stack(alphas_list, dim=0) if alphas_list else None
        
        # 基于集成方法组合动作和alpha
        if self.ensemble_method == 'mean':
            # 简单平均
            final_actions = torch.mean(stacked_actions, dim=0)
            final_alpha = torch.mean(stacked_alphas, dim=0) if stacked_alphas is not None else None
        elif self.ensemble_method == 'weighted':
            # 加权平均
            weights = torch.softmax(self.ensemble_weights, dim=0)
            weights = weights.view(-1, 1, 1, 1)  # 广播形状
            final_actions = torch.sum(stacked_actions * weights, dim=0)
            final_alpha = torch.sum(stacked_alphas * weights, dim=0) if stacked_alphas is not None else None
        else:
            # 默认使用均值
            final_actions = torch.mean(stacked_actions, dim=0)
            final_alpha = torch.mean(stacked_alphas, dim=0) if stacked_alphas is not None else None
        
        return final_actions, final_alpha
    
    def reset(self) -> None:
        """重置集成中的所有策略。"""
        for policy in self.policies:
            policy.reset()


def create_policy_from_config(config: Dict) -> nn.Module:
    """
    Factory function to create a policy from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Policy network instance
    """
    policy_type = config.get('type', 'bptt')
    
    if policy_type == 'bptt':
        return BPTTPolicy(config)
    elif policy_type == 'ensemble':
        return EnsemblePolicy(config)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}") 