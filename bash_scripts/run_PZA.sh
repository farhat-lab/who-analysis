drug="Pyrazinamide"
drug_abbr="PZA"

# list of config files to use
config_array=(
 'config_files/atu_01.yaml'
 'config_files/atu_02.yaml'
 'config_files/atu_03.yaml'
 'config_files/atu_04.yaml'
 'config_files/atu_05.yaml'
 'config_files/atu_06.yaml'
 'config_files/atu_07.yaml'
 'config_files/atu_08.yaml'
)

for i in ${!config_array[@]}; do
    python3 -u 01_make_model_inputs.py "${config_array[$i]}" "$drug" "$drug_abbr"
    python3 -u 02_regression_with_bootstrap.py "${config_array[$i]}" "$drug" "$drug_abbr"
    python3 -u 03_model_analysis.py "${config_array[$i]}" "$drug" "$drug_abbr"
done

python3 -u 04_compute_univariate_stats.py "$drug" "ATU" "/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue"