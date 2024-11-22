import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['figure.dpi'] = 150
import seaborn as sns
import scipy.stats as st
import statsmodels
import statsmodels.api as sm
from functools import reduce

import glob, os, yaml, subprocess, itertools, sparse, sys, statsmodels, shutil, argparse
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'

lineages = pd.read_csv("./combined_lineages_samples.csv")
import collections, warnings
warnings.filterwarnings("ignore")

coll2014 = pd.read_csv("../data/coll2014_SNP_scheme.tsv", sep="\t")
coll2014["#lineage"] = coll2014["#lineage"].str.replace("lineage", "")
coll2014.rename(columns={"#lineage": "Lineage"}, inplace=True)
coll2014['nucleotide'] = [val.split('/')[1] for val in coll2014['allele_change'].values]

drug_abbr_dict = {"Delamanid": "DLM",
                  "Bedaquiline": "BDQ",
                  "Clofazimine": "CFZ",
                  "Ethionamide": "ETO",
                  "Linezolid": "LZD",
                  "Moxifloxacin": "MXF",
                  "Capreomycin": "CAP",
                  "Amikacin": "AMK",
                  "Pretomanid": "PMD",
                  "Pyrazinamide": "PZA",
                  "Kanamycin": "KAN",
                  "Levofloxacin": "LFX",
                  "Streptomycin": "STM",
                  "Ethambutol": "EMB",
                  "Isoniazid": "INH",
                  "Rifampicin": "RIF"
                 }

cc_df = pd.read_csv("../data/drug_CC.csv")

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input_file', type=str, required=True, help='Catalog file to get mutations from')

cmd_line_args = parser.parse_args()
input_file = cmd_line_args.input_file 
results_final = pd.read_csv(input_file)

drug_gene_mapping = pd.read_csv("../data/drug_gene_mapping.csv")

silent_lst = ['synonymous_variant', 'initiator_codon_variant', 'stop_retained_variant']
lof_lst = ["frameshift", "start_lost", "stop_gained", "feature_ablation"]

regression_upgrades = results_final.loc[(~results_final['REGRESSION FINAL CONFIDENCE GRADING'].isin(['3) Uncertain significance'])) & (results_final['SOLO FINAL CONFIDENCE GRADING']=='3) Uncertain significance')]

print(f"Getting lineage distributions of {len(regression_upgrades)} (drug, mutation) pairs")

lineage_dist_table = []

for drug in np.sort(regression_upgrades.Drug.unique()):

    df_genos = pd.read_csv(os.path.join(analysis_dir, drug, "genos_1.csv.gz"), compression='gzip', usecols=['sample_id', 'resolved_symbol', 'variant_category', 'variant_allele_frequency', 'predicted_effect'])

    df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv"))

    df_genos['mutation'] = df_genos['resolved_symbol'] + '_' + df_genos['variant_category']
    del df_genos['resolved_symbol']
    del df_genos['variant_category']

    single_drug_lineage_dist = []
    
    for mutation in regression_upgrades.query("Drug==@drug").mutation.values:

        if "LoF" in mutation:
            gene = mutation.replace('_LoF', '')
            samples_with_variant = df_genos.query("mutation.str.contains(@gene) & predicted_effect in @lof_lst & variant_allele_frequency > 0.75").sample_id.values
        else:
            samples_with_variant = df_genos.query("mutation==@mutation & variant_allele_frequency > 0.75").sample_id.values
        
        print(f"{len(samples_with_variant)} isolates with {mutation} for {drug}")
        
        single_table = lineages.query("Sample_ID in @samples_with_variant")[['Sample_ID', 'Coll2014', 'Freschi2020', 'Lipworth2019', 'Shitikov2017', 'Stucki2016', 'Lineage']].rename(columns={'Sample_ID': 'sample_id'})

        # add phenotypes
        single_table = single_table.merge(df_phenos, on='sample_id')

        # add mutation
        single_table['mutation'] = mutation

        single_drug_lineage_dist.append(single_table)

    # combine results for all mutations for a single drug
    single_drug_lineage_dist = pd.concat(single_drug_lineage_dist, axis=0)
    single_drug_lineage_dist['Drug'] = drug
    lineage_dist_table.append(single_drug_lineage_dist)
    
    print(f"Finished {drug}")

pd.concat(lineage_dist_table, axis=0).to_csv("isolates_with_new_muts.csv", index=False)