#!/bin/bash 
#SBATCH -c 10
#SBATCH -t 0-04:00
#SBATCH -p short 
#SBATCH --mem=150G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

drug="Streptomycin"
drug_abbr="STM"

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
 'config_files/binary_17.yaml'
 'config_files/binary_18.yaml'
 'config_files/binary_19.yaml'
 'config_files/binary_20.yaml'
)

# for i in ${!config_array[@]}; do
#     python3 -u 01_make_model_inputs.py "${config_array[$i]}" "$drug" "$drug_abbr"
#     python3 -u 02_regression_with_bootstrap.py "${config_array[$i]}" "$drug" "$drug_abbr"
#     python3 -u 03_model_analysis.py "${config_array[$i]}" "$drug" "$drug_abbr"
# done

python3 -u 04_compute_univariate_stats.py "$drug" "BINARY" "/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue"