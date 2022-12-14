#!/bin/bash 
#SBATCH -c 10
#SBATCH -t 0-11:59
#SBATCH -p short 
#SBATCH --mem=50G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

drug="Moxifloxacin"
drug_abbr="MXF"

# list of config files to use
config_array=(
 'config_files/mic_01.yaml'
 'config_files/mic_02.yaml'
 'config_files/mic_03.yaml'
 'config_files/mic_04.yaml'
 'config_files/mic_05.yaml'
 'config_files/mic_06.yaml'
 'config_files/mic_07.yaml'
 'config_files/mic_08.yaml'
)

# for i in ${!config_array[@]}; do
#     python3 -u 01_make_model_inputs.py "${config_array[$i]}" "$drug" "$drug_abbr"
#     python3 -u 02_regression_with_bootstrap.py "${config_array[$i]}" "$drug" "$drug_abbr"
#     python3 -u 03_model_analysis.py "${config_array[$i]}" "$drug" "$drug_abbr"
# done

python3 -u 04_compute_univariate_stats.py "$drug" "MIC" "/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue"