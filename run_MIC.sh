#!/bin/bash 
#SBATCH -c 2
#SBATCH -t 0-06:00
#SBATCH -p short
#SBATCH --mem=100G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu


#################### NEED TO PASS IN ARGUMENTS: DRUG, DRUG_ABBR, IN THIS ORDER ####################


source activate who-analysis

drug_array=(
 # 'Pretomanid'
 # 'Delamanid'
 # 'Bedaquiline'
 # 'Clofazimine'
 # 'Linezolid'
 # 'Moxifloxacin'
 # 'Levofloxacin'
 # 'Rifampicin'
 'Isoniazid'
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
  'INH'
)

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

for k in ${!drug_array[@]}; do
    for i in ${!config_array[@]}; do
        python3 -u model/01_make_model_inputs.py "${config_array[$i]}" "${drug_array[$k]}" "${drug_abbr_array[$k]}"
        python3 -u model/02_run_regression.py "${config_array[$i]}" "${drug_array[$k]}" "${drug_abbr_array[$k]}"
    done
done
