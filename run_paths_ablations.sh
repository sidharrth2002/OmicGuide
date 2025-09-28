#!/bin/bash
set -m

# filepath: /home/sn666/dissertation/benchmarking/PATHS/run_paths_ablations.sh

# Generate a unique log file for each Python script invocation
generate_log_file() {
    local script_name=$(basename "$1" .py)
    local timestamp=$(date "+%Y%m%d-%H%M%S")
    echo "/home/sn666/dissertation/benchmarking/PATHS/logs/${script_name}_${timestamp}.log"
}

conda activate paths2

export HF_TOKEN=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
export WANDB_API_KEY=eedb37fd2f84d7a84f7df28b901b02d3e377718a

cd ~/dissertation/benchmarking/PATHS

# frac 0.2

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.2/kirp_paths_0 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.2/kirp_paths_1 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.2/kirp_paths_2 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.2/kirp_paths_3 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.2/kirp_paths_4 | tee -a $LOG_FILE

# frac 0.4
LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.4/kirp_paths_0 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.4/kirp_paths_1 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.4/kirp_paths_2 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.4/kirp_paths_3 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.4/kirp_paths_4 | tee -a $LOG_FILE

# frac 0.6
LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.6/kirp_paths_0 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.6/kirp_paths_1 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.6/kirp_paths_2 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.6/kirp_paths_3 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.6/kirp_paths_4 | tee -a $LOG_FILE

# frac 0.8
LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.8/kirp_paths_0 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.8/kirp_paths_1 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.8/kirp_paths_2 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.8/kirp_paths_3 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_0.8/kirp_paths_4 | tee -a $LOG_FILE

# frac 1.0
LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_1.0/kirp_paths_0 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_1.0/kirp_paths_1 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_1.0/kirp_paths_2 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_1.0/kirp_paths_3 | tee -a $LOG_FILE

LOG_FILE=$(generate_log_file ablations)
echo "Logging to $LOG_FILE"
python train.py -m ablations/leaf_frac/leaf_frac_1.0/kirp_paths_4 | tee -a $LOG_FILE