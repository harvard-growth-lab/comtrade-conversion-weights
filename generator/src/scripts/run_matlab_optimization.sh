#!/bin/bash
# Load MATLAB module

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOGS_DIR="${SCRIPT_DIR}/../../logs/matlab_optimization"
mkdir -p "$LOGS_DIR"

LOG_FILE="${LOGS_DIR}/optimization_run_$(date +%Y%m%d_%H%M%S).log"
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Change to working directory
cd "$SCRIPT_DIR"

MATLAB_PARAMS_FILE="$SCRIPT_DIR/../../data/temp/matlab_script_params.txt"
source "$MATLAB_PARAMS_FILE"


echo "=== DIRECTORY DEBUG ==="
echo "Current working directory: $(pwd)"
echo "Script location: $(dirname "${BASH_SOURCE[0]}")"
echo "Script absolute path: $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


MATLAB_OPTIMIZATION_FILE="$SCRIPT_DIR/../matlab/"
cd "$MATLAB_OPTIMIZATION_FILE"
echo "matlab file location ${MATLAB_OPTIMIZATION_FILE}"


# Convert the space-separated strings into arrays
START_YEARS=($START_YEARS)
END_YEARS=($END_YEARS)
MAX_GROUPS=($MAX_GROUPS)

log_message "Starting optimization runs"
log_message "Configuration:"
log_message "Start years: ${START_YEARS[*]}"
log_message "End years: ${END_YEARS[*]}"
log_message "Max groups: ${MAX_GROUPS[*]}"

# Set tolerance
TOL=1e-20

# Loop through all year pairs
for i in "${!START_YEARS[@]}"; do
    log_message "Processing: Start year ${START_YEARS[i]}, End year ${END_YEARS[i]}, Max group ${MAX_GROUPS[i]}"
    
    matlab -nosplash -nodesktop -r "start_year=${START_YEARS[i]}; end_year=${END_YEARS[i]}; groups=1:${MAX_GROUPS[i]}; tol=1e-20; diary('${LOGS_DIR}/matlab_output_${START_YEARS[i]}_${END_YEARS[i]}.log'); run('MAIN_Matlab_optimization_GL.m'); diary off; exit;" 2>&1 | tee -a "$LOG_FILE"
    
    log_message "Completed run $((i+1))/${#START_YEARS[@]}"
    log_message "----------------------------------------"
done

log_message "All runs completed"