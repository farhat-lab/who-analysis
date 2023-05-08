#!/bin/bash 
#SBATCH -c 2
#SBATCH -t 0-11:59
#SBATCH -p short
#SBATCH --mem=100G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu


#################### NEED TO PASS IN ARGUMENTS: ANALYSIS_DIR, DRUG, DRUG_ABBR, IN THIS ORDER ####################


source activate who-analysis

# list of config files to use
config_array=(
 'config_files/binary_01.yaml'
 'config_files/binary_04.yaml'
 'config_files/binary_07.yaml'
 'config_files/binary_11.yaml'
)

het_config_array=(
 'config_files/het_01.yaml'
 'config_files/het_04.yaml'
 'config_files/het_07.yaml'
 'config_files/het_11.yaml'
 )

drug_array=(
    # "Bedaquiline"
    # "Linezolid"
    # "Levofloxacin"
    # "Moxifloxacin"
    # "Clofazimine"
    "Delamanid"
    "Pretomanid"
)

for k in ${!drug_array[@]}; do
    for i in ${!config_array[@]}; do
        python3 -u 06_binary_prediction_models_full.py "${config_array[$i]}" "${drug_array[$k]}"
    done
done

for k in ${!drug_array[@]}; do
    for i in ${!het_config_array[@]}; do
        python3 -u 06_binary_prediction_models_full.py "${het_config_array[$i]}" "${drug_array[$k]}"
    done
done