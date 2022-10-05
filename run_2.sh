#!/bin/bash 
#SBATCH -c 4
#SBATCH -t 0-11:59
#SBATCH -p short 
#SBATCH --mem=100G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

python3 -u 01_make_model_inputs.py config.yaml Moxifloxacin MXF
python3 -u 02_regression_with_bootstrap.py config.yaml Moxifloxacin
python3 -u 03_model_analysis.py config.yaml Moxifloxacin MXF

python3 -u 01_make_model_inputs.py config_2.yaml Moxifloxacin MXF
python3 -u 02_regression_with_bootstrap.py config_2.yaml Moxifloxacin
python3 -u 03_model_analysis.py config_2.yaml Moxifloxacin MXF