#!/bin/bash

# run_experiments.sh
# Automated script to run all alpha configurations for BPTT experiments

echo "============================================================"
echo "Running GCBF+ BPTT Experiments with Different Alpha Values"
echo "============================================================"

# Environment type
ENV_TYPE="double_integrator"

# Function to run an experiment with a specific configuration
run_experiment() {
    config_file=$1
    config_name=$(basename $config_file .yaml)
    
    echo ""
    echo "------------------------------------------------------------"
    echo "Starting experiment with configuration: $config_name"
    echo "------------------------------------------------------------"
    
    # Run training
    python train_bptt.py --config $config_file --env_type $ENV_TYPE
    
    echo "Experiment $config_name completed"
}

# Run experiments with different alpha values
echo "Running experiment with Low Alpha (0.5)"
run_experiment "config/alpha_low.yaml"

echo "Running experiment with Medium Alpha (1.0)"
run_experiment "config/alpha_medium.yaml"

echo "Running experiment with High Alpha (2.0)"
run_experiment "config/alpha_high.yaml"

echo ""
echo "============================================================"
echo "All experiments completed successfully!"
echo "============================================================" 