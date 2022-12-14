# list of config files to use
drug_array=(
 'Amikacin'
 'Bedaquiline'
 'Capreomycin'
 'Clofazimine'
 'Delamanid'
 'Ethambutol'
 'Ethionamide'
 'Isoniazid'
 'Kanamycin'
 'Levofloxacin'
 'Linezolid'
 'Moxifloxacin'
 'Pyrazinamide'
 'Rifampicin'
 'Streptomycin'
)

drug_abbr_array=(
 'AMI'
 'BDQ'
 'CAP'
 'CFZ'
 'DLM'
 'EMB'
 'ETH'
 'INH'
 'KAN'
 'LEV'
 'LZD'
 'MXF'
 'PZA'
 'RIF'
 'STM'
)

# list of config files to use
config_array=(
 'config_files/binary_01.yaml'
 'config_files/binary_02.yaml'
 'config_files/binary_03.yaml'
 'config_files/binary_04.yaml'
 'config_files/binary_05.yaml'
 'config_files/binary_06.yaml'
 'config_files/binary_07.yaml'
 'config_files/binary_08.yaml'
 'config_files/binary_09.yaml'
 'config_files/binary_10.yaml'
 'config_files/binary_11.yaml'
 'config_files/binary_12.yaml'
 'config_files/binary_13.yaml'
 'config_files/binary_14.yaml'
 'config_files/binary_15.yaml'
 'config_files/binary_16.yaml'
 'config_files/binary_17.yaml'
 'config_files/binary_18.yaml'
 'config_files/binary_19.yaml'
 'config_files/binary_20.yaml'
)

for k in ${!drug_array[@]}; do
    for i in ${!config_array[@]}; do 
        python3 -u 01_make_model_inputs.py "${config_array[$i]}" "${drug_array[$k]}" "${drug_abbr_array[$k]}"
        python3 -u 02_regression_with_bootstrap.py "${config_array[$i]}" "${drug_array[$k]}" "${drug_abbr_array[$k]}"
        python3 -u 03_model_analysis.py "${config_array[$i]}" "${drug_array[$k]}" "${drug_abbr_array[$k]}"
    done
    python3 -u 04_compute_univariate_stats.py "${drug_array[$k]}" "BINARY" "/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue"
done