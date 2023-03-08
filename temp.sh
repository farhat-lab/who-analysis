# list of config files to use
config_array=(
 'config_files/binary_01.yaml'
 # 'config_files/binary_02.yaml'
 'config_files/binary_03.yaml'
 'config_files/binary_04.yaml'
 'config_files/binary_05.yaml'
 # 'config_files/binary_06.yaml'
 'config_files/binary_07.yaml'
 'config_files/binary_08.yaml'
 'config_files/binary_09.yaml'
 # 'config_files/binary_10.yaml'
 'config_files/binary_11.yaml'
 'config_files/binary_12.yaml'
 'config_files/binary_13.yaml'
 # 'config_files/binary_14.yaml'
 'config_files/binary_15.yaml'
 'config_files/binary_16.yaml'
)

for i in ${!config_array[@]}; do
    # python3 -u 01_make_model_inputs.py "${config_array[$i]}" "$1" "$2"
    # python3 -u 02_run_regression.py "${config_array[$i]}" "$1" "$2"
    python3 -u 06_binary_prediction_models.py "${config_array[$i]}" "$1" "$2"
done