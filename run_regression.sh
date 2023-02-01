#!/bin/bash 
#SBATCH -c 10
#SBATCH -t 0-11:59
#SBATCH -p short 
#SBATCH --mem=10G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu


#################### NEED TO PASS IN ARGUMENTS: DRUG, DRUG_ABBR, AND ANALYSIS_DIR, IN THIS ORDER ####################

# list of config files to use
config_array=(
 'config_files/binary_01.yaml'
 'config_files/binary_02.yaml'
 'config_files/binary_03.yaml'
 'config_files/binary_04.yaml'
 'config_files/binary_05.yaml'
 'config_files/binary_06.yaml'
 'config_files/binary_07.yaml'
 'config_files/binary_08.yaml'
 'config_files/binary_09.yaml'
 'config_files/binary_10.yaml'
 'config_files/binary_11.yaml'
 'config_files/binary_12.yaml'
 'config_files/binary_13.yaml'
 'config_files/binary_14.yaml'
 'config_files/binary_15.yaml'
 'config_files/binary_16.yaml'
)

for i in ${!config_array[@]}; do
    python3 -u 01_make_model_inputs.py "${config_array[$i]}" "$1" "$2"
    python3 -u 02_run_regression.py "${config_array[$i]}" "$1" "$2"
done

# get the folder name (basename, then split on "_" and get the first word, and make it uppercase)
folder=$(basename "${config_array[0]}" | cut -d "_" -f 1 | tr '[:lower:]' '[:upper:]')
python3 -u 03_compute_univariate_stats.py "$drug" "${folder}" "$3"