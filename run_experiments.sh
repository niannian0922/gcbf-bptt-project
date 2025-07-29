#!/bin/bash

# run_experiments.sh
# Automated script to run the Final Bottleneck Showdown Experiments

echo "============================================================"
echo "    FINAL BOTTLENECK SHOWDOWN: Dynamic vs Fixed Alpha    "
echo "============================================================"

# Environment type
ENV_TYPE="double_integrator"

# Create results directory
mkdir -p results/bottleneck_showdown
RESULTS_DIR="results/bottleneck_showdown"

# Function to run a complete experiment with training and evaluation
run_complete_experiment() {
    config_file=$1
    config_name=$(basename $config_file .yaml)
    
    echo ""
    echo "============================================================"
    echo "🚀 Starting BOTTLENECK experiment: $config_name"
    echo "============================================================"
    
    # Run training for 20,000 steps
    echo "📈 Phase 1: Training for 20,000 steps..."
    python train_bptt.py --config $config_file --env_type $ENV_TYPE
    
    if [ $? -eq 0 ]; then
        echo "✅ Training completed successfully"
        
        # Run visualization and analysis
        echo "📊 Phase 2: Generating visualization and quantitative analysis..."
        
        # Extract the run name from config for finding the model directory
        run_name=$(python -c "
import yaml
with open('$config_file', 'r') as f:
    config = yaml.safe_load(f)
print(config['training']['name'])
")
        
        # Find the latest model directory
        model_dir="logs/${run_name}"
        
        if [ -d "$model_dir" ]; then
            # Generate visualization with bottleneck metrics
            visualization_file="${RESULTS_DIR}/${config_name}_bottleneck_analysis.gif"
            metrics_file="${RESULTS_DIR}/${config_name}_metrics.json"
            
            python visualize_bptt.py \
                --model_dir "$model_dir" \
                --env_type $ENV_TYPE \
                --output "$visualization_file" \
                --metrics_output "$metrics_file"
                
            if [ $? -eq 0 ]; then
                echo "✅ Visualization and analysis completed"
                echo "📁 Results saved:"
                echo "   - Visualization: $visualization_file"
                echo "   - Metrics: $metrics_file"
            else
                echo "❌ Visualization failed"
            fi
        else
            echo "❌ Model directory not found: $model_dir"
        fi
    else
        echo "❌ Training failed for $config_name"
    fi
    
    echo "🏁 Experiment $config_name completed"
}

# Function to generate comparative report
generate_comparative_report() {
    echo ""
    echo "============================================================"
    echo "📊 Generating Comparative Analysis Report"
    echo "============================================================"
    
    report_file="${RESULTS_DIR}/bottleneck_showdown_report.txt"
    
    echo "==============================================================" > $report_file
    echo "    FINAL BOTTLENECK SHOWDOWN - QUANTITATIVE RESULTS" >> $report_file
    echo "==============================================================" >> $report_file
    echo "" >> $report_file
    echo "Experiment Date: $(date)" >> $report_file
    echo "Environment: Bottleneck (narrow gap: 0.25 units)" >> $report_file
    echo "Agents: 8" >> $report_file
    echo "Training Steps: 20,000" >> $report_file
    echo "" >> $report_file
    
    # Process each metrics file
    for config in "bottleneck_dynamic_alpha" "bottleneck_fixed_alpha_low" "bottleneck_fixed_alpha_medium" "bottleneck_fixed_alpha_high"; do
        metrics_file="${RESULTS_DIR}/${config}_metrics.json"
        
        if [ -f "$metrics_file" ]; then
            echo "--------------------------------------------------------------" >> $report_file
            echo "Configuration: $config" >> $report_file
            echo "--------------------------------------------------------------" >> $report_file
            
            # Extract key metrics using Python
            python -c "
import json
try:
    with open('$metrics_file', 'r') as f:
        metrics = json.load(f)
    
    print(f'Time to Goal: {metrics.get(\"time_to_goal\", \"N/A\")} steps')
    print(f'Total Control Effort: {metrics.get(\"total_control_effort\", \"N/A\"):.3f}')
    print(f'Min Separation Distance: {metrics.get(\"min_separation_distance\", \"N/A\"):.3f}')
    
    # Bottleneck-specific metrics
    if 'bottleneck_throughput' in metrics:
        print(f'🎯 Throughput: {metrics[\"bottleneck_throughput\"]:.3f} agents/sec')
        print(f'🌊 Velocity Fluctuation: {metrics[\"bottleneck_velocity_fluctuation\"]:.3f}')
        print(f'⏱️  Total Waiting Time: {metrics[\"bottleneck_total_waiting_time\"]:.2f} sec')
        print(f'🤝 Coordination Efficiency: {metrics[\"bottleneck_coordination_efficiency\"]:.3f}')
        print(f'💥 Collision Rate: {metrics[\"bottleneck_collision_rate\"]:.4f} collisions/agent/sec')
        print(f'🎉 Completion Rate: {metrics[\"bottleneck_completion_rate\"]:.3f}')
    else:
        print('⚠️  Bottleneck metrics not available')
        
except Exception as e:
    print(f'❌ Error reading metrics: {e}')
" >> $report_file
            echo "" >> $report_file
        else
            echo "❌ Metrics file not found: $metrics_file" >> $report_file
            echo "" >> $report_file
        fi
    done
    
    echo "==============================================================" >> $report_file
    echo "📈 SUMMARY: Key Performance Indicators" >> $report_file
    echo "==============================================================" >> $report_file
    echo "Higher is better: Throughput, Coordination Efficiency, Completion Rate" >> $report_file
    echo "Lower is better: Velocity Fluctuation, Waiting Time, Collision Rate" >> $report_file
    echo "==============================================================" >> $report_file
    
    echo "✅ Comparative report generated: $report_file"
    
    # Display the report
    echo ""
    echo "📋 FINAL RESULTS PREVIEW:"
    cat $report_file
}

# Run the four key experiments
echo "🔬 Running 4 comprehensive bottleneck experiments..."

echo "🟢 Experiment 1/4: Dynamic Alpha (Our Innovation)"
run_complete_experiment "config/bottleneck_dynamic_alpha.yaml"

echo "🔴 Experiment 2/4: Fixed Alpha Low (Baseline 1)"
run_complete_experiment "config/bottleneck_fixed_alpha_low.yaml"

echo "🟡 Experiment 3/4: Fixed Alpha Medium (Baseline 2)" 
run_complete_experiment "config/bottleneck_fixed_alpha_medium.yaml"

echo "🟠 Experiment 4/4: Fixed Alpha High (Baseline 3)"
run_complete_experiment "config/bottleneck_fixed_alpha_high.yaml"

# Generate final comparative report
generate_comparative_report

echo ""
echo "============================================================"
echo "🎉 FINAL BOTTLENECK SHOWDOWN COMPLETED SUCCESSFULLY! 🎉"
echo "============================================================"
echo "📂 All results saved in: $RESULTS_DIR"
echo "📊 Comparative report: ${RESULTS_DIR}/bottleneck_showdown_report.txt"
echo ""
echo "🏆 READY FOR PAPER PUBLICATION! 🏆"
echo "============================================================" 