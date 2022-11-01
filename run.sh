#!/bin/bash 
#SBATCH -c 10
#SBATCH -t 3-00:00
#SBATCH -p medium 
#SBATCH --mem=150G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yasha_ektefaie@g.harvard.edu

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

for k in ${!drug_array[@]}; do 
    for i in ${!config_array[@]}; do 
        python3 -u 01_make_model_inputs.py "${config_array[$i]}" "${drug_array[$k]}" "${abbr_array[$k]}"
        python3 -u 02_regression_with_bootstrap.py "${config_array[$i]}" "${drug_array[$k]}" "${abbr_array[$k]}"
        python3 -u 03_model_analysis.py "${config_array[$i]}" "${drug_array[$k]}" "${abbr_array[$k]}"
    
    # combine model results and compute univariate statistics for the significant variants
    python3 -u analysis/01_combine_model_analyses.py "${drug_array[$k]}"
    python3 -u analysis/02_compute_univariate_stats.py "${drug_array[$k]}"
    python3 -u analysis/03_model_metrics.py "${drug_array[$k]}"
    
    done
done
