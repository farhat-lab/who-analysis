#!/usr/bin/env bash
set -ex

# conda install --file requirements.txt -c conda-forge -c bioconda
# pip install fast-lineage-caller==0.3.1

drug_array=(
 # "Amikacin"
 # "Bedaquiline"
 # "Capreomycin"
 # "Clofazimine"
 "Delamanid"
 # "Ethambutol"
 # "Ethionamide"
 # "Isoniazid"
 # "Kanamycin"
 # "Levofloxacin"
 # "Linezolid"
 # "Moxifloxacin"
 # "Pyrazinamide"
 # "Rifampicin"
 # "Streptomycin"
)

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

# grading scripts -- don't need to be run on every config file or every drug. Just need a single config file to get the output directory
python3 -u grading/01_get_single_model_results.py -c "config.yaml"
python3 -u grading/02_combine_WHO_ALL_results.py -c "config.yaml" -o results/regression_mutation_classifications.csv