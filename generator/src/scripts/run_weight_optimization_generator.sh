#!/bin/bash
#SBATCH -p shared
#SBATCH -t 0-12:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -o matlab_job_%j.out
#SBATCH -e matlab_job_%j.err

# Load MATLAB module
module load matlab/R2021a-fasrc01

# Record start time
start_time=$(date +%s)

# Change to working directory
cd '/n/holystore01/LABS/hausmann_lab/lab/atlas/bustos_yildirim/weights_generator/generator/src/matlab'

source /n/hausmann_lab/lab/atlas/bustos_yildirim/weights_generator/generator/data/temp/matlab_script_params.txt

START_YEARS=($START_YEARS)
END_YEARS=($END_YEARS)
MAX_GROUPS=($MAX_GROUPS)


# Set tolerance
TOL=1e-20

# Loop through all year pairs
for i in ${!START_YEARS[@]}; do
    echo "Processing: Start year ${START_YEARS[$i]}, End year ${END_YEARS[$i]}, Max group ${MAX_GROUPS[$i]}"
    
    
    matlab -nosplash -nodesktop -r "start_year=${START_YEARS[$i]}; end_year=${END_YEARS[$i]}; groups=1:${MAX_GROUPS[$i]}; tol=1e-20; run('MAIN_Matlab_optimization_GL.m'); exit;"
    
    echo "Completed run $((i+1))/${#START_YEARS[@]}"
    echo "----------------------------------------"
done

echo "All runs completed"