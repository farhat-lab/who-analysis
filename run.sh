#!/bin/bash 
#SBATCH -c 10
#SBATCH -t 0-11:59
#SBATCH -p short 
#SBATCH --mem=100G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu


# list of drugs and associated abbreviations
drug_array=(
 # 'Ethambutol'
 'Amikacin'
 'Clofazimine'
 # 'Delamanid'
 'Capreomycin'
 # 'Rifampicin'
 'Ethionamide'
 'Kanamycin'
 # 'Isoniazid'
 'Streptomycin'
 'Pyrazinamide'
 'Linezolid'
 'Bedaquiline'
 'Moxifloxacin'
 'Levofloxacin'
)

abbr_array=(
 # 'EMB'
 'AMI'
 'CFZ'
 # 'DLM'
 'CAP'
 # 'RIF'
 'ETH'
 'KAN'
 # 'INH'
 'STM'
 'PZA'
 'LZD'
 'BDQ'
 'MXF'
 'LEV'
)


for k in ${!drug_array[@]}; do 
    python3 -u 01_make_model_inputs.py config_files/config_01.yaml "${drug_array[$k]}" "${abbr_array[$k]}"
done