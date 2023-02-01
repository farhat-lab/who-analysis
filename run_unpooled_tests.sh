#!/bin/bash 
#SBATCH -c 10
#SBATCH -t 2-00:00
#SBATCH -p medium 
#SBATCH --mem=10G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

#################### NEED TO PASS IN ARGUMENTS: DRUG AND DRUG_ABBR, IN THIS ORDER ####################

config_array=(
 'config_files/binary_03.yaml'
 'config_files/binary_07.yaml'
 'config_files/binary_11.yaml'
 'config_files/binary_15.yaml'
)

# only run these 2 analyses on the 
for i in ${!config_array[@]}; do
    python3 -u 04_likelihood_ratio_test.py "${config_array[$i]}" "$1" "$2"
    python3 -u 05_AUC_permutation_test.py "${config_array[$i]}" "$1" "$2"
done