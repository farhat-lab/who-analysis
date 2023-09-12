#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-11:59
#SBATCH -p short
#SBATCH --mem=150G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu


#################### NEED TO PASS IN ARGUMENTS: DRUG, DRUG_ABBR, IN THIS ORDER ####################


source activate who-analysis

# list of config files to use
config_array=(
 'config_files/binary_01.yaml'
 'config_files/binary_02.yaml'
 'config_files/binary_03.yaml'
 # 'config_files/binary_04.yaml'
 # 'config_files/binary_05.yaml'
 # 'config_files/binary_06.yaml'
 # 'config_files/binary_07.yaml'
 # 'config_files/binary_08.yaml'
 'config_files/binary_09.yaml'
 'config_files/binary_10.yaml'
 'config_files/binary_11.yaml'
 # 'config_files/binary_12.yaml'
 # 'config_files/binary_13.yaml'
 # 'config_files/binary_14.yaml'
 # 'config_files/binary_15.yaml'
 # 'config_files/binary_16.yaml'
)

drug_array=(
 # 'Pretomanid'
 # 'Delamanid'
 # 'Bedaquiline'
 # 'Clofazimine'
 # 'Linezolid'
 # 'Moxifloxacin'
 # 'Levofloxacin'
 # 'Rifampicin'
 # 'Isoniazid'
 'Ethionamide'
 # 'Kanamycin'
 # 'Amikacin'
 # 'Streptomycin'
 # 'Pyrazinamide'
 # 'Capreomycin'
 # 'Ethambutol'
)

drug_abbr_array=(
 # 'PTM'
 # 'DLM'
 # 'BDQ'
 # 'CFZ'
 # 'LZD'
 # 'MXF'
 # 'LEV'
 # 'RIF'
 # 'INH'
 'ETH'
 # 'KAN'
 # 'AMI'
 # 'STM'
 # 'PZA'
 # 'CAP'
 # 'EMB'
)


# # only one config_file because using all phenotypes and tier 1 only
# for k in ${!drug_array[@]}; do
#     for i in ${!config_array[@]}; do
#         python3 -u model/05_binary_prediction_models.py "${config_array[$i]}" "${drug_array[$k]}" 0.75
#         python3 -u model/05_binary_prediction_models.py "${config_array[$i]}" "${drug_array[$k]}" 0.25
#     done    
# done


# get the folder name (basename, then split on "_" and get the first word, and make it uppercase)
folder=$(basename "${config_array[0]}" | cut -d "_" -f 1 | tr '[:lower:]' '[:upper:]')
echo $folder

for k in ${!drug_array[@]}; do
    for i in ${!config_array[@]}; do
        python3 -u model/01_make_model_inputs.py "${config_array[$i]}" "${drug_array[$k]}" "${drug_abbr_array[$k]}"
        python3 -u model/02_run_regression.py "${config_array[$i]}" "${drug_array[$k]}" "${drug_abbr_array[$k]}"
        python3 -u model/03_likelihood_ratio_test.py "${config_array[$i]}" "${drug_array[$k]}"
    done

    python3 -u model/04_compute_univariate_stats.py "${folder}" "${drug_array[$k]}"
    
done