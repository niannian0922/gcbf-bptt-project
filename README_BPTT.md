# GCBF+BTN: A Differentiable Multi-Agent Safe Control Framework

This extension to the GCBF+ framework integrates the training philosophy from the paper "Back to Newton's Laws: Learning Vision-based Agile Flight via Differentiable Physics" (BTN) to create an end-to-end, gradient-based optimization approach for multi-agent safe control.

## Overview

The GCBF+BTN framework combines:

1. **Graph Control Barrier Functions (GCBF)** from the original GCBF+ paper for safety certification of multi-agent systems.
2. **Backpropagation Through Time (BPTT)** training paradigm from BTN for end-to-end optimization through differentiable physics.

This integration eliminates the need for the complex data labeling process, expert policies, and Q-learning in the original GCBF+ framework by directly optimizing both policy and CBF networks through backpropagation.

## Key Components

### 1. Differentiable Physics Simulator

A PyTorch-based differentiable simulator (`DifferentiableDoubleIntegrator`) that allows gradients to flow through the state transitions:

```python
x_{t+1} = A * x_t + B * u_t
```

Where:
- `x_t` is the current state [x, y, vx, vy]
- `u_t` is the control input [fx, fy]
- `A` and `B` are state transition matrices

### 2. PyTorch GNN Architecture

Both the control policy (π_φ) and Control Barrier Function (h_θ) are implemented as Graph Neural Networks using PyTorch, allowing for:

- Variable number of neighbors
- Distributed decision-making
- Transferability to different numbers of agents

### 3. End-to-End Training Loop

The core innovation is the BPTT training loop that:

1. Builds graphs from agent states
2. Runs policy network to get actions
3. Computes CBF values and ensures safety constraints
4. Simulates forward dynamics
5. Optimizes networks end-to-end based on goal, safety, and control costs

## Safety Guarantee

Safety is enforced through the CBF condition:
```
h_dot + α * h ≥ 0
```

Where:
- `h` is the CBF value (higher means safer)
- `h_dot` is the time derivative of h
- `α` is a positive constant

This is incorporated in the loss function as a soft constraint, guiding the joint optimization of both policy and CBF networks.

## Usage

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train using the BPTT approach
python train_bptt.py --config config/bptt_config.yaml --cuda --log_dir logs/bptt_run1
```

### Visualization

```bash
# Visualize trained models
python visualize_bptt.py --model_dir logs/bptt_run1 --cuda
```

## Configuration Parameters

Key parameters in `config/bptt_config.yaml`:

- `horizon_length`: Number of simulation steps for BPTT
- `goal_weight`: Weight for the goal-reaching objective
- `safety_weight`: Weight for the safety constraint violation
- `control_weight`: Weight for the control effort minimization
- `cbf_alpha`: Alpha parameter for the CBF condition

## Benefits Over Original GCBF+

1. **Simplified Training**: No need for data labeling or expert policies
2. **End-to-End Optimization**: Joint optimization of policy and CBF
3. **More Gradient Information**: Backpropagation through time provides richer gradient signals
4. **Faster Training**: Direct optimization instead of separate training stages

## References

1. Original GCBF+ paper: [GCBF+: Graph Neural Networks for Distributed Control of Multi-Agent Systems with Safety Guarantees](https://arxiv.org/abs/2307.07417)
2. BTN paper: [Back to Newton's Laws: Learning Vision-based Agile Flight via Differentiable Physics](https://arxiv.org/abs/2303.11555) 