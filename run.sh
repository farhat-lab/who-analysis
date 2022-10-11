#!/bin/bash 
#SBATCH -c 10
#SBATCH -t 0-11:59
#SBATCH -p short 
#SBATCH --mem=100G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

# drug_array=(
#  'Ethambutol'
#  'Amikacin'
#  'Clofazimine'
#  'Delamanid'
#  'Capreomycin'
#  'Rifampicin'
#  'Ethionamide'
#  'Kanamycin'
#  'Isoniazid'
#  'Streptomycin'
#  'Pyrazinamide'
#  'Linezolid'
#  'Bedaquiline'
#  'Moxifloxacin'
#  'Levofloxacin'
# )
# abbr_array=(
#  'EMB'
#  'AMI'
#  'CFZ'
#  'DLM'
#  'CAP'
#  'RIF'
#  'ETH'
#  'KAN'
#  'INH'
#  'STM'
#  'PZA'
#  'LZD'
#  'BDQ'
#  'MXF'
#  'LEV'
# )

# drug_array = ('Ethambutol' 'Rifampicin' 'Isoniazid' 'Linezolid')
# abbr_array = ('EMB' 'RIF' 'INH' 'LZD')

# drop variants with any 0.25 < AF < 0.75
printf "\nCapreomycin\n"
# python3 -u 01_make_model_inputs.py config.yaml Capreomycin CAP
# python3 -u 02_regression_with_bootstrap.py config.yaml Capreomycin
python3 -u 03_model_analysis.py config.yaml Capreomycin CAP
python3 -u 03_model_analysis.py config_2.yaml Capreomycin CAP

# printf "\nRunning Tiers 1/2, WHO phenotypic models, dropping HETs, including synonymous, pooling LOF \n"

# # encode variants with AF > 0.25 using their AF
# for i in ${!drug_array[@]}; do 
#   printf "\n${drug_array[$i]}\n"
#   python3 -u 01_make_model_inputs.py config_2.yaml "${drug_array[$i]}" "${abbr_array[$i]}"
#   python3 -u 02_regression_with_bootstrap.py config_2.yaml "${drug_array[$i]}"
#   python3 -u 03_model_analysis.py config_2.yaml "${drug_array[$i]}" "${abbr_array[$i]}"
# done