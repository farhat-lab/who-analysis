#!/bin/bash 
#SBATCH -c 10
#SBATCH -t 4-23:59
#SBATCH -p medium 
#SBATCH --mem=100G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

# list of config files to use
config_array=(
 'config.yaml'
)

# list of drugs and associated abbreviations
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

for i in ${!config_array[@]}; do 
    for k in ${!drug_array[@]}; do 
      python3 -u 01_make_model_inputs.py "${config_array[$i]}" "${drug_array[$k]}" "${abbr_array[$k]}"
      python3 -u 02_regression_with_bootstrap.py "${config_array[$i]}" "${drug_array[$k]}"
      python3 -u 03_model_analysis.py "${config_array[$i]}" "${drug_array[$k]}" "${abbr_array[$k]}"
      python3 -u analysis/logReg_metrics.py "${config_array[$i]}" "${drug_array[$k]}"
    done
done