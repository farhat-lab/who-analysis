#!/bin/bash 
#SBATCH -c 10
#SBATCH -t 4-23:59
#SBATCH -p medium 
#SBATCH --mem=100G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

drug_array=(
 'Ethambutol'
 'Amikacin'
 'Clofazimine'
 'Delamanid'
 'Capreomycin'
 'Rifampicin'
 'Ethionamide'
 'Kanamycin'
 'Isoniazid'
 'Streptomycin'
 'Pyrazinamide'
 'Linezolid'
 'Bedaquiline'
 'Moxifloxacin'
 'Levofloxacin'
)
abbr_array=(
 'EMB'
 'AMI'
 'CFZ'
 'DLM'
 'CAP'
 'RIF'
 'ETH'
 'KAN'
 'INH'
 'STM'
 'PZA'
 'LZD'
 'BDQ'
 'MXF'
 'LEV'
)

printf "\nRunning Tiers 1/2, WHO phenotypic models, dropping HETs \n"

for i in ${!drug_array[@]}; do 
  printf "\n${drug_array[$i]}\n"
  python3 -u 01_make_model_inputs.py config_5.yaml "${drug_array[$i]}" "${abbr_array[$i]}"
  python3 -u 02_regression_with_bootstrap.py config_5.yaml "${drug_array[$i]}"
  python3 -u 03_model_analysis.py config_5.yaml "${drug_array[$i]}" "${abbr_array[$i]}"
done

printf "\nRunning Tiers 1/2, WHO phenotypic models, dropping HETs, including synonymous\n"

for i in ${!drug_array[@]}; do 
  printf "\n${drug_array[$i]}\n"
  python3 -u 01_make_model_inputs.py config_6.yaml "${drug_array[$i]}" "${abbr_array[$i]}"
  python3 -u 02_regression_with_bootstrap.py config_6.yaml "${drug_array[$i]}"
  python3 -u 03_model_analysis.py config_6.yaml "${drug_array[$i]}" "${abbr_array[$i]}"
done