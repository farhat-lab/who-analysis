#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 1-00:00
#SBATCH -p medium
#SBATCH --mem=150G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu


#################### NEED TO PASS IN ARGUMENTS: DRUG, DRUG_ABBR, IN THIS ORDER ####################


source activate who-analysis

drug_array=(
 'Delamanid'
)

# # get the folder name (basename, then split on "_" and get the first word, and make it uppercase)
# folder=$(basename "${config_array[0]}" | cut -d "_" -f 1 | tr '[:lower:]' '[:upper:]')
# echo $folder

for k in ${!drug_array[@]}; do
    
    python3 model/05_binary_prediction_models.py "${drug_array[$k]}" 0.75
    # python3 model/05_binary_prediction_models.py "${drug_array[$k]}" 0.25
    # python3 model/07_binary_prediction_models_SOLO.py "${drug_array[$k]}" 0.75 INITIAL
    # python3 model/07_binary_prediction_models_SOLO.py "${drug_array[$k]}" 0.25 INITIAL
    # python3 model/07_binary_prediction_models_SOLO.py "${drug_array[$k]}" 0.75 FINAL
    # python3 model/07_binary_prediction_models_SOLO.py "${drug_array[$k]}" 0.25 FINAL

done