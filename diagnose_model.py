#!/usr/bin/env python3
"""
Diagnose a trained Probabilistic Safety Shield model.

Usage:
  python diagnose_model.py --model_dir logs/probabilistic_safety_shield_5000/models/5000 \
                           --episodes 1 --device auto --out diagnostics

Outputs:
  - diagnostics/safety_confidence_vs_distance.png
  - diagnostics/control_authority_handover.png
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gcbfplus.env.double_integrator import DoubleIntegratorEnv
from gcbfplus.policy.bptt_policy import BPTTPolicy
from gcbfplus.env.gcbf_safety_layer import GCBFSafetyLayer
from gcbfplus.utils.episode_logger import compute_min_distances_to_obstacles


def load_model_and_env(model_dir: str, device: torch.device) -> Tuple[BPTTPolicy, GCBFSafetyLayer, Dict[str, Any], DoubleIntegratorEnv]:
    """Load policy, CBF layer and environment config, construct env."""
    config_pt = os.path.join(model_dir, "config.pt")
    if not os.path.exists(config_pt):
        raise FileNotFoundError(f"Config not found in model dir: {config_pt}")

    config = torch.load(config_pt, map_location='cpu', weights_only=False)
    env = DoubleIntegratorEnv(config.get('env', {})).to(device)

    # Build networks from config
    policy_cfg = config.get('networks', {}).get('policy', config.get('networks', {}))
    policy = BPTTPolicy(policy_cfg).to(device)

    cbf_cfg = config.get('env', {}).get('safety_layer', config.get('networks', {}).get('cbf', {}))
    cbf = GCBFSafetyLayer(cbf_cfg).to(device)

    # Load weights
    policy_path = os.path.join(model_dir, 'policy.pt')
    cbf_path = os.path.join(model_dir, 'cbf.pt')
    policy.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True), strict=False)
    if os.path.exists(cbf_path):
        cbf.load_state_dict(torch.load(cbf_path, map_location=device, weights_only=True), strict=False)

    return policy, cbf, config, env


def run_episode(policy: BPTTPolicy, cbf: GCBFSafetyLayer, env: DoubleIntegratorEnv, device: torch.device, max_steps: int = 300) -> Dict[str, np.ndarray]:
    """Run a single diagnostic episode and collect alpha_safety and distances."""
    policy.eval()
    state = env.reset(batch_size=1)  # returns dict or tensor depending on env implementation
    if isinstance(state, dict) and 'observations' in state:
        obs = state['observations'].to(device)
    else:
        # assume tensor observations
        obs = state.to(device)

    alpha_list: list = []
    dist_list: list = []

    with torch.no_grad():
        for t in range(max_steps):
            actions, _, dynamic_margins = policy(obs)
            # Safety layer blending inside env API
            if hasattr(env, 'apply_safety_layer'):
                blended_action, alpha_safety = env.apply_safety_layer(obs, actions, cbf, dynamic_margins)
            else:
                # Fallback: compute alpha and use raw action
                alpha_safety = cbf.compute_safety_confidence(obs, dynamic_margins)
                blended_action = actions

            # Step environment
            next_state, rewards, done, info = env.step(blended_action)

            # Extract positions and obstacles for distance computation
            if isinstance(next_state, dict) and 'positions' in next_state:
                positions = next_state['positions']  # [1, n_agents, pos_dim]
            else:
                # If env packs state, try info
                positions = info.get('positions') if isinstance(info, dict) else None
            obstacles = None
            if isinstance(info, dict):
                obstacles = info.get('obstacles')

            # Compute min distances (numpy)
            if positions is not None:
                pos_np = positions.detach().cpu().numpy()
                if obstacles is not None:
                    obs_np = obstacles.detach().cpu().numpy() if isinstance(obstacles, torch.Tensor) else np.asarray(obstacles)
                else:
                    # No obstacles in this setup → distance is inf; keep for scatter completeness
                    obs_np = None
                dists = compute_min_distances_to_obstacles(pos_np, obs_np)
                # For 1 batch, average over agents to obtain a scalar per timestep
                dist_scalar = float(np.mean(dists[0])) if dists.size else float('inf')
            else:
                dist_scalar = float('inf')

            # Alpha to scalar per timestep (avg over agents)
            alpha_np = alpha_safety.detach().cpu().numpy()
            alpha_scalar = float(np.mean(alpha_np))

            alpha_list.append(alpha_scalar)
            dist_list.append(dist_scalar)

            obs = next_state['observations'].to(device) if isinstance(next_state, dict) and 'observations' in next_state else next_state.to(device)

            if bool(done):
                break

    return {
        'alpha': np.array(alpha_list, dtype=float),
        'dist': np.array(dist_list, dtype=float)
    }


def plot_safety_confidence_vs_distance(alpha: np.ndarray, dist: np.ndarray, out_dir: str) -> str:
    plt.figure(figsize=(7, 5))
    plt.scatter(dist, alpha, s=12, alpha=0.7)
    plt.xlabel('True Distance to Nearest Obstacle')
    plt.ylabel('Alpha Safety Confidence')
    plt.title('Risk Assessment Profile of the Trained Model')
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(out_dir, 'safety_confidence_vs_distance.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_control_authority_handover(alpha: np.ndarray, out_dir: str) -> str:
    plt.figure(figsize=(8, 4))
    plt.plot(alpha, lw=2)
    plt.xlabel('Timestep')
    plt.ylabel('Alpha Safety Confidence')
    plt.title('Control Authority Handover During Episode')
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(out_dir, 'control_authority_handover.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True, help='Path to model directory containing policy.pt/cbf.pt/config.pt')
    ap.add_argument('--episodes', type=int, default=1)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--out', default='diagnostics')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)

    os.makedirs(args.out, exist_ok=True)

    policy, cbf, config, env = load_model_and_env(args.model_dir, device)

    # Determine max steps from config/eval settings
    max_steps = config.get('training', {}).get('eval_horizon', config.get('env', {}).get('max_steps', 300))

    all_alpha = []
    all_dist = []

    for ep in range(args.episodes):
        print(f"Running diagnostic episode {ep+1}/{args.episodes}...")
        data = run_episode(policy, cbf, env, device, max_steps=max_steps)
        all_alpha.append(data['alpha'])
        all_dist.append(data['dist'])

    alpha_concat = np.concatenate(all_alpha)
    dist_concat = np.concatenate(all_dist)

    p1 = plot_safety_confidence_vs_distance(alpha_concat, dist_concat, args.out)
    p2 = plot_control_authority_handover(all_alpha[0], args.out)

    print(f"Saved: {p1}")
    print(f"Saved: {p2}")


if __name__ == '__main__':
    main()


