#!/bin/bash 
#SBATCH -c 10
#SBATCH -t 0-11:59
#SBATCH -p short 
#SBATCH --mem=100G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

drug="Pyrazinamide"
drug_abbr="PZA"

# list of config files to use
config_array=(
 'config_files/config_01.yaml'
 'config_files/config_02.yaml'
 'config_files/config_03.yaml'
 'config_files/config_04.yaml'
 'config_files/config_05.yaml'
 'config_files/config_06.yaml'
 'config_files/config_07.yaml'
 'config_files/config_08.yaml'
 'config_files/config_09.yaml'
 'config_files/config_10.yaml'
 'config_files/config_11.yaml'
 'config_files/config_12.yaml'
 'config_files/config_13.yaml'
 'config_files/config_14.yaml'
 'config_files/config_15.yaml'
 'config_files/config_16.yaml'
)

for i in ${!config_array[@]}; do 
    python3 -u 01_make_model_inputs.py "${config_array[$i]}" "$drug" "$drug_abbr"
    python3 -u 02_regression_with_bootstrap.py "${config_array[$i]}" "$drug" "$drug_abbr"
    python3 -u 03_model_analysis.py "${config_array[$i]}" "$drug" "$drug_abbr"
done

python3 -u analysis/01_combine_model_analyses.py "$drug"