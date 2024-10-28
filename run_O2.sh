#!/bin/bash
#SBATCH -c 1
#SBATCH -t 4-23:59
#SBATCH --mem=100G 
#SBATCH -p medium
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

# conda install --file requirements.txt -c conda-forge -c bioconda
# pip install fast-lineage-caller==0.3.1

source activate who-analysis

drug_array=(
 "Amikacin"
 "Bedaquiline"
 "Capreomycin"
 "Clofazimine"
 "Delamanid"
 # "Ethambutol"
 "Ethionamide"
 # "Isoniazid"
 "Kanamycin"
 "Levofloxacin"
 "Linezolid"
 "Moxifloxacin"
 "Pyrazinamide"
 # "Rifampicin"
 "Streptomycin"
)

# for drug in "${drug_array[@]}"; do

#     # grading rules
#     python3 -u prediction/catalog_model.py -d $drug --grading-rules

#     # # low AF, no grading rules
#     # python3 -u prediction/catalog_model.py -d $drug --AF 0.25
#     # python3 -u prediction/catalog_model_SOLO.py -d $drug --AF 0.25
# done

# Define the array
config_array=(
 "binary_01.yaml" 
 "binary_02.yaml" 
 "binary_03.yaml"
 "binary_04.yaml"
 "binary_05.yaml"
 "binary_06.yaml"
 "mic_01.yaml"
 "mic_02.yaml"
 "mic_03.yaml"
)

# Iterate through each drug, then each config file
for drug in "${drug_array[@]}"; do
    
    for config in "${config_array[@]}"; do
        # model fitting scripts
        python3 -u model/01_make_model_inputs.py -c "config_files/$config" -d $drug
        python3 -u model/02_run_regression.py -c "config_files/$config" -d $drug
        python3 -u model/03_likelihood_ratio_test.py -c "config_files/$config" -d $drug
    done

    # this script only needs to be run once for each drug because it looks for all available model results
    python3 -u model/04_compute_univariate_stats.py -c "config_files/$config" -d $drug
    
done

# # grading scripts -- don't need to be run on every config file or every drug. Just need a single config file to get the output directory
# python3 -u grading/01_get_single_model_results.py -c "config.yaml"
# python3 -u grading/02_combine_WHO_ALL_results.py -c "config.yaml" -o results/regression_mutation_classifications.csv