#!/bin/bash 
#SBATCH -c 10
#SBATCH -t 0-06:00
#SBATCH -p short 
#SBATCH --mem=300G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

# python3 -u analysis/03_model_metrics.py
# python3 -u temp.py
# python3 -u make_gene_drug_mapping.py data/drug_gene_mapping_new.csv

python3 -u analysis/02_compute_univariate_stats.py Rifampicin
python3 -u analysis/02_compute_univariate_stats.py Isoniazid

# python3 -u analysis/01_combine_model_analyses.py Ethambutol
# python3 -u analysis/02_compute_univariate_stats.py Ethambutol