#!/usr/bin/env python3
"""
Episode Evaluation with Data Logging

This script evaluates trained models and generates comprehensive episode data logs
for offline analysis and visualization.

Usage:
    python evaluate_with_logging.py --model-dir logs/bptt/models/9500 --config config/alpha_medium_obs.yaml --episodes 5
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gcbfplus.trainer.bptt_trainer import BPTTTrainer
from gcbfplus.env.double_integrator import DoubleIntegratorEnv
from gcbfplus.policy.bptt_policy import BPTTPolicy
from gcbfplus.env.gcbf_safety_layer import GCBFSafetyLayer


def load_model_and_config(model_dir: str, config_path: str) -> tuple:
    """
    Load trained model and configuration.
    
    Args:
        model_dir: Directory containing saved models (policy.pt, cbf.pt, etc.)
        config_path: Path to configuration YAML file
        
    Returns:
        Tuple of (policy_network, cbf_network, config)
    """
    print(f"Loading model from: {model_dir}")
    print(f"Loading config from: {config_path}")
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Try to load model-specific config if available
    model_config_path = os.path.join(model_dir, "config.pt")
    if os.path.exists(model_config_path):
        try:
            model_config = torch.load(model_config_path, map_location='cpu', weights_only=False)
            # Merge model config with base config (model config takes precedence)
            if isinstance(model_config, dict):
                config.update(model_config)
                print("Model-specific config loaded and merged")
        except Exception as e:
            print(f"Warning: Could not load model config: {e}")
    
    # Check for model files
    policy_path = os.path.join(model_dir, "policy.pt")
    cbf_path = os.path.join(model_dir, "cbf.pt")
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy model not found: {policy_path}")
    
    # Load policy network
    print("Creating policy network...")
    policy_config = config.get('networks', {}).get('policy', {})
    
    # Set default values if not specified
    if 'perception' not in policy_config:
        policy_config['perception'] = {
            'input_dim': 9,  # Default for obstacle environments
            'hidden_dim': 128,
            'activation': 'relu'
        }
    
    if 'memory' not in policy_config:
        policy_config['memory'] = {
            'hidden_dim': 128,
            'num_layers': 1
        }
    
    if 'policy_head' not in policy_config:
        policy_config['policy_head'] = {
            'output_dim': 2,  # 2D actions
            'hidden_dim': 128,
            'activation': 'relu',
            'predict_alpha': True
        }
    
    policy_network = BPTTPolicy(policy_config)
    
    # Load policy weights
    policy_state_dict = torch.load(policy_path, map_location='cpu', weights_only=True)
    missing_keys, unexpected_keys = policy_network.load_state_dict(policy_state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing policy keys: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected policy keys: {unexpected_keys}")
    
    print("Policy network loaded successfully")
    
    # Load CBF network if available
    cbf_network = None
    if os.path.exists(cbf_path):
        print("Loading CBF network...")
        
        # Create CBF network
        cbf_config = config.get('networks', {}).get('cbf', {})
        
        # GCBFSafetyLayer expects a config dict, not individual parameters
        cbf_layer_config = {
            'alpha': config.get('env', {}).get('cbf_alpha', 1.0),
            'eps': 0.02,
            'safety_margin': config.get('env', {}).get('agent_radius', 0.2),
            'use_qp': True
        }
        
        cbf_network = GCBFSafetyLayer(cbf_layer_config)
        
        # Load CBF weights
        cbf_state_dict = torch.load(cbf_path, map_location='cpu', weights_only=True)
        cbf_network.load_state_dict(cbf_state_dict, strict=False)
        print("CBF network loaded successfully")
    else:
        print("Warning: CBF network not found, proceeding without safety layer")
    
    return policy_network, cbf_network, config


def create_environment(config: Dict[str, Any]) -> DoubleIntegratorEnv:
    """
    Create environment from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Environment instance
    """
    print("🌍 Creating environment...")
    
    env_config = config.get('env', {})
    
    # Set default values if not specified
    defaults = {
        'num_agents': 2,
        'area_size': 2.0,
        'dt': 0.05,
        'mass': 0.1,
        'max_force': 1.0,
        'agent_radius': 0.2,
        'comm_radius': 1.0,
        'max_steps': 200
    }
    
    for key, default_value in defaults.items():
        if key not in env_config:
            env_config[key] = default_value
    
    # Create environment
    env = DoubleIntegratorEnv(env_config)
    print(f"✅ Environment created: {env_config['num_agents']} agents, area {env_config['area_size']}x{env_config['area_size']}")
    
    return env


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained model with comprehensive logging")
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Directory containing saved models (e.g., logs/bptt/models/9500)")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file (e.g., config/alpha_medium_obs.yaml)")
    parser.add_argument("--episodes", type=int, default=3,
                       help="Number of episodes to evaluate (default: 3)")
    parser.add_argument("--eval-horizon", type=int, default=300,
                       help="Maximum steps per episode (default: 300)")
    parser.add_argument("--output-dir", type=str, default="episode_analysis",
                       help="Directory to save analysis results (default: episode_analysis)")
    parser.add_argument("--auto-plot", action="store_true",
                       help="Automatically generate plots for all episodes")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use for evaluation (default: auto)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model_dir):
        print(f"❌ Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    # Set device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Starting evaluation with data logging")
    print(f"   Model: {args.model_dir}")
    print(f"   Config: {args.config}")
    print(f"   Episodes: {args.episodes}")
    print(f"   Max steps: {args.eval_horizon}")
    print(f"   Device: {device}")
    print("=" * 70)
    
    try:
        # Load model and configuration
        policy_network, cbf_network, config = load_model_and_config(args.model_dir, args.config)
        
        # Create environment
        env = create_environment(config)
        
        # Move models to device
        policy_network = policy_network.to(device)
        if cbf_network is not None:
            cbf_network = cbf_network.to(device)
        env = env.to(device)
        
        # Create trainer with logging enabled
        trainer_config = config.copy()
        trainer_config.update({
            'enable_episode_logging': True,
            'eval_horizon': args.eval_horizon,
            'log_dir': args.output_dir,
            'device': str(device)
        })
        
        # Initialize trainer
        trainer = BPTTTrainer(
            env=env,
            policy_network=policy_network,
            cbf_network=cbf_network,
            config=trainer_config
        )
        
        print(f"\n🎯 Running evaluation with {args.episodes} episodes...")
        
        # Run evaluation with logging
        metrics = trainer.evaluate_with_logging(
            num_episodes=args.episodes,
            log_episodes=True
        )
        
        # Print legacy results (backward compatibility)
        print(f"\n📊 传统评估结果:")
        print("=" * 50)
        for key, value in metrics.items():
            if key != "eval/episode_files" and not key.startswith("champion/"):
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
        
        # 🏆 **NEW: 冠军评估体系的KPI结果已经在trainer中打印了**
        
        # Get episode files for plotting
        episode_files = metrics.get("eval/episode_files", [])
        
        # 🏆 **NEW: 生成冠军性能仪表盘**
        try:
            print(f"\n🏆 生成冠军性能仪表盘...")
            
            # 提取冠军KPIs
            champion_kpis = {k: v for k, v in metrics.items() if k.startswith("champion/")}
            
            if champion_kpis and episode_files:
                # Import plotting functionality
                from plot_results import EpisodePlotter
                from gcbfplus.utils.episode_logger import load_episode_data
                
                # 寻找最佳episode文件
                best_episode_file = champion_kpis.get('champion/best_episode_file')
                best_episode_data = None
                
                if best_episode_file and os.path.exists(best_episode_file):
                    try:
                        best_episode_data = load_episode_data(best_episode_file)
                        print(f"   🥇 已加载冠军Episode数据: {os.path.basename(best_episode_file)}")
                    except Exception as e:
                        print(f"   ⚠️ 无法加载冠军Episode数据: {e}")
                
                # 创建仪表盘
                dashboard_path = os.path.join(args.output_dir, "CHAMPION_PERFORMANCE_DASHBOARD.png")
                
                # 使用任一episode文件创建plotter（仅用于仪表盘生成）
                plotter = EpisodePlotter(episode_files[0])
                dashboard_fig = plotter.generate_performance_dashboard(
                    kpi_results=champion_kpis,
                    best_episode_data=best_episode_data,
                    output_path=dashboard_path
                )
                
                print(f"   🏆 冠军性能仪表盘已保存: {dashboard_path}")
                
        except Exception as e:
            print(f"   ❌ 生成冠军仪表盘失败: {e}")
        
        # 标准绘图流程 (如果启用)
        if episode_files and args.auto_plot:
            print(f"\n🎨 生成标准plots for {len(episode_files)} episodes...")
            
            # Import plotting functionality
            from plot_results import EpisodePlotter
            
            plot_dir = os.path.join(args.output_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            for i, episode_file in enumerate(episode_files):
                try:
                    print(f"   📈 Plotting episode {i+1}: {os.path.basename(episode_file)}")
                    plotter = EpisodePlotter(episode_file)
                    
                    # Generate plots in the plot directory
                    plotter.save_all_plots(plot_dir)
                    
                except Exception as e:
                    print(f"   ❌ Failed to plot episode {i+1}: {e}")
            
            print(f"📁 所有标准plots已保存到: {plot_dir}")
        
        # Provide usage instructions
        print(f"\n💡 冠军评估体系使用说明:")
        print(f"   🏆 冠军性能仪表盘: {os.path.join(args.output_dir, 'CHAMPION_PERFORMANCE_DASHBOARD.png')}")
        print(f"   📊 Episode原始数据: {trainer.episode_logger.log_dir}")
        print(f"   📈 标准详细plots: {os.path.join(args.output_dir, 'plots')}")
        print(f"   📈 手动绘制individual episodes:")
        
        for i, episode_file in enumerate(episode_files[:2]):  # Show first 2 as examples
            filename = os.path.basename(episode_file)
            print(f"      python plot_results.py {episode_file}")
        
        if len(episode_files) > 2:
            print(f"      ... 以及更多 {len(episode_files) - 2} 个文件")
        
        # 🏆 **NEW: 冠军评估体系总结**
        print(f"\n🏆 冠军评估体系总结:")
        if champion_kpis:
            success_rate = champion_kpis.get('champion/success_rate', 0)
            robustness = champion_kpis.get('champion/robustness_score', 0)
            best_time = champion_kpis.get('champion/best_completion_time', 0)
            print(f"   ✅ 成功率: {success_rate:.1%}")
            print(f"   🛡️ 鲁棒性得分: {robustness:.3f}")
            print(f"   🚀 最佳完成时间: {best_time:.0f} 步")
            
            # 确定模型等级
            if success_rate >= 0.8 and robustness >= 0.7:
                grade = "🥇 冠军级别"
            elif success_rate >= 0.6 and robustness >= 0.5:
                grade = "🥈 优秀级别"
            elif success_rate >= 0.4:
                grade = "🥉 良好级别"
            else:
                grade = "💩 需要改进"
            
            print(f"   🏆 模型等级: {grade}")
        
        print(f"\n✅ 冠军评估体系评估完成!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
