#!/bin/bash 
#SBATCH -c 2
#SBATCH -t 0-11:59
#SBATCH -p short
#SBATCH --mem=20G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu


#################### NEED TO PASS IN ARGUMENTS: DRUG, DRUG_ABBR, IN THIS ORDER ####################


source activate who-analysis

# list of config files to use
config_array=(
 'config_files/mic_01.yaml'
 'config_files/mic_02.yaml'
 'config_files/mic_03.yaml'
 'config_files/mic_04.yaml'
 'config_files/mic_05.yaml'
 'config_files/mic_06.yaml'
)


# get the folder name (basename, then split on "_" and get the first word, and make it uppercase)
folder=$(basename "${config_array[0]}" | cut -d "_" -f 1 | tr '[:lower:]' '[:upper:]')
echo $folder

for i in ${!config_array[@]}; do
    python3 -u 01_make_model_inputs.py "${config_array[$i]}" "$1" "$2"
    python3 -u 02_run_regression.py "${config_array[$i]}" "$1" "$2"
    # python3 -u 03_likelihood_ratio_test.py "${config_array[$i]}" "$1" "$2"
    # python3 -u 05_AUC_permutation_test.py "${config_array[$i]}" "$1"
done