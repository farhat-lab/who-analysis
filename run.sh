#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-11:59
#SBATCH -p short
#SBATCH --mem=50G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu


#################### NEED TO PASS IN ARGUMENTS: DRUG, DRUG_ABBR, IN THIS ORDER ####################


# cd lineages
# python3 -u get_lineages_isolates_withMut.py

source activate who-analysis

# list of config files to use
config_array=(
 # 'config_files/binary_01.yaml'
 'config_files/binary_02.yaml'
 'config_files/binary_03.yaml'
 # 'config_files/binary_04.yaml'
 'config_files/binary_05.yaml'
 'config_files/binary_06.yaml'
 # 'config_files/mic_01.yaml'
 # 'config_files/mic_02.yaml'
 # 'config_files/mic_03.yaml'
)

drug_array=(
 # 'Pretomanid'
 'Delamanid'
 'Bedaquiline'
 'Clofazimine'
 'Linezolid'
 'Moxifloxacin'
 'Levofloxacin'
 'Rifampicin'
 'Isoniazid'
 'Ethionamide'
 'Kanamycin'
 'Amikacin'
 'Streptomycin'
 'Pyrazinamide'
 'Capreomycin'
 'Ethambutol'
)

# # get the folder name (basename, then split on "_" and get the first word, and make it uppercase)
# folder=$(basename "${config_array[0]}" | cut -d "_" -f 1 | tr '[:lower:]' '[:upper:]')
# echo $folder

# for k in ${!drug_array[@]}; do

#     for i in ${!config_array[@]}; do
#         # python3 -u model/01_make_model_inputs.py -c "${config_array[$i]}" -drug "${drug_array[$k]}"
#         # python3 -u model/02_run_regression.py -c "${config_array[$i]}" -drug "${drug_array[$k]}"
#         python3 -u model/03_likelihood_ratio_test.py -c "${config_array[$i]}" -drug "${drug_array[$k]}"
#     done

#     python3 -u model/04_compute_univariate_stats.py -drug "${drug_array[$k]}" -model 'BINARY'

# done

for k in ${!drug_array[@]}; do

    python3 -u model/05_catalog_model.py --drug "${drug_array[$k]}" --S-assoc
    python3 -u model/05_catalog_model.py --drug "${drug_array[$k]}" --AF 0.25 --S-assoc

    # use regression + GR column
    python3 -u model/05_catalog_model.py --drug "${drug_array[$k]}" --grading-rules --S-assoc
    python3 -u model/05_catalog_model.py --drug "${drug_array[$k]}" --AF 0.25 --grading-rules --S-assoc

    python3 -u model/06_catalog_model_SOLO.py --drug "${drug_array[$k]}" --S-assoc
    python3 -u model/06_catalog_model_SOLO.py --drug "${drug_array[$k]}" --AF 0.25 --S-assoc

done

python3 -u model/05_catalog_model.py --drug "Isoniazid" --remove-mut --S-assoc
python3 -u model/05_catalog_model.py --drug "Isoniazid" --remove-mut --grading-rules --S-assoc

python3 -u model/05_catalog_model.py --drug "Capreomycin" --remove-mut --S-assoc
python3 -u model/05_catalog_model.py --drug "Capreomycin" --remove-mut --grading-rules --S-assoc