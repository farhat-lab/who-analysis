import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 125
import seaborn as sns
from Bio import SeqIO, Seq
import scipy.stats as st
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import glob, os, yaml, subprocess, itertools, sparse, pickle

who_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_resistance_variants_all.csv")
analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'
genos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/full_genotypes'


def read_in_all_genos(drug):
            
    # first get all the genotype files associated with the drug
    geno_files = []

    for subdir in os.listdir(os.path.join(genos_dir, f"drug_name={drug}")):

        # subdirectory (tiers)
        full_subdir = os.path.join(genos_dir, f"drug_name={drug}", subdir)

        # the last character is the tier number. Get variants from both tiers
        if full_subdir[-1] in ["1", "2"]:
            for fName in os.listdir(full_subdir):
                if "run" in fName:
                    geno_files.append(os.path.join(full_subdir, fName))

    print(f"    {len(geno_files)} files with genotypes")

    dfs_lst = []
    for i, fName in enumerate(geno_files):

        print(f"Reading in genotypes dataframe {fName}")
        df = pd.read_csv(fName, low_memory=False)
        if "tier=1" in fName:
            df["Tier"] = 1
        elif "tier=2" in fName:
            df["Tier"] = 2
        else:
            raise ValueError("Invalid path!")
            
        dfs_lst.append(df)

    # fail-safe if there are duplicate rows
    return pd.concat(dfs_lst).drop_duplicates().reset_index(drop=True)


finished_drugs = ['Amikacin', 'Bedaquiline', 'Capreomycin', 'Clofazimine',
       'Delamanid', 'Ethambutol', 'Ethionamide', 'Kanamycin',
       'Levofloxacin', 'Linezolid', 'Moxifloxacin', 'Pyrazinamide',
       'Rifampicin', 'Streptomycin', 'Isoniazid']

for drug in finished_drugs:
    
    df_model = read_in_all_genos(drug)

    if len(df_model.loc[~pd.isnull(df_model["neutral"])]) == 0:
        del df_model["neutral"]

    df_model.to_csv(os.path.join(analysis_dir, drug, "genos.csv.gz"), compression="gzip", index=False)
    print(f"Finished {drug}")


# def final_processing(drug):
#     '''
#     Functions for processing outputs before sending to everyone else.
    
#     1. Remove principal components (will describe them separately)
#     2. Add LOF to the predicted_effect column for pooled LOF mutations
#     3. Remove genome_index column (should actually do that earlier, but will fix later)
#     4. Remove the logistic regression coefficient columns (they will prefer to work with odds ratios)
#     5. Any other column renaming or dropping for clarity
#     '''
    
#     analysis_df = pd.read_csv(os.path.join(analysis_dir, drug, "final_analysis.csv"))
#     analysis_df.rename(columns={"orig_variant": "mutation"}, inplace=True)
    
#     # remove logReg coefficients. Keep only odds ratios. Remove the other two columns, which were present mainly for me to see
#     # if we were picking up many mutations that were in the 2021 mutation catalog
#     del analysis_df["confidence_WHO_2021"]
#     analysis_df = analysis_df[analysis_df.columns[~analysis_df.columns.str.contains("coef")]]
    
#     # remove significant principal components and replace the NaNs in the predicted effect column for the gene loss of functions
#     analysis_df = analysis_df.loc[~analysis_df["mutation"].str.contains("PC")]
    
#     # predicted effect should not be NaN for anything. position is NaN only for the pooled LOF mutations
#     assert len(analysis_df.loc[pd.isnull(analysis_df["predicted_effect"])]) == 0
    
#     # analysis_df.loc[(analysis_df["Phenos"]=='WHO') & (analysis_df["Tier"]==1) & (analysis_df["unpooled"]==0) & (analysis_df["synonymous"]==0), "core"] = 1
#     # analysis_df["core"] = analysis_df["core"].fillna(0)
    
#     assert len(analysis_df[['Num_Isolates', 'Total_Isolates', 'TP', 'FP', 'TN', 'FN']].dropna()) == len(analysis_df)
#     analysis_df[['Num_Isolates', 'Total_Isolates', 'TP', 'FP', 'TN', 'FN']] = analysis_df[['Num_Isolates', 'Total_Isolates', 'TP', 'FP', 'TN', 'FN']].astype(int)
#     analysis_df["HET"] = "DROP"
    
#     # reorder columns
#     analysis_df = analysis_df[['mutation', 'predicted_effect', 'Odds_Ratio', 'pval', 'BH_pval', 'Bonferroni_pval',
#        'Num_Isolates', 'Total_Isolates', 'TP', 'FP', 'TN', 'FN', 'PPV', 'Sens', 'Spec', 'LR+', 'LR-',
#        'OR_LB', 'OR_UB', 'PPV_LB', 'PPV_UB', 'Sens_LB', 'Sens_UB', 'Spec_LB', 'Spec_UB', 'LR+_LB', 'LR+_UB', 'LR-_LB', 'LR-_UB', 
#                                'Tier', 'Phenos', 'unpooled', 'synonymous', 'HET'
#                               #'core'
#                               ]]
            
#     return analysis_df



# def add_AF_models(drug):
    
#     analysis = final_processing(drug)
#     analysis_order = pd.read_csv(os.path.join(analysis_dir, drug, "analysis_order.csv"))
    
#     test = analysis_order[["orig_variant", "Odds_Ratio", 'pval', 'BH_pval',
#        'Bonferroni_pval', "OR_LB", "OR_UB", 'Tier', 'Phenos', 'unpooled', 'synonymous', 'HET']].merge(analysis[['mutation', 'predicted_effect', 'Num_Isolates', 'Total_Isolates', 'TP', 'FP', 'TN',
#        'FN', 'PPV', 'Sens', 'Spec', 'LR+', 'LR-', 'PPV_LB',
#        'PPV_UB', 'Sens_LB', 'Sens_UB', 'Spec_LB', 'Spec_UB', 'LR+_LB',
#        'LR+_UB', 'LR-_LB', 'LR-_UB']], left_on="orig_variant", right_on="mutation", how="outer")

#     del test["mutation"]

#     test.loc[test["orig_variant"].str.contains("lof"), "predicted_effect"] = "lof"
#     test.loc[test["orig_variant"].str.contains("inframe"), "predicted_effect"] = "inframe"

#     df_genos = pd.read_csv(os.path.join(analysis_dir, drug, "genos.csv.gz"), compression="gzip", usecols=["resolved_symbol", "variant_category", "predicted_effect", "position"]).drop_duplicates(subset=["resolved_symbol", "variant_category"], keep="first")
#     df_genos["mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
    
#     test["position"] = test["orig_variant"].map(dict(zip(df_genos["mutation"], df_genos["position"])))
#     test.loc[pd.isnull(test["predicted_effect"]), "predicted_effect"] = test.loc[pd.isnull(test["predicted_effect"])]["orig_variant"].map(dict(zip(df_genos["mutation"], df_genos["predicted_effect"])))
        
#     assert len(test.loc[(pd.isnull(test["predicted_effect"])) & (~test["orig_variant"].str.contains("PC"))]) == 0
#     test.to_csv(os.path.join(analysis_dir, drug, "combined_analyses.csv"), index=False)
    


# # those that are actually done
# finished_drugs = ['Amikacin', 'Bedaquiline', 'Capreomycin', 'Clofazimine',
#        'Delamanid', 'Ethambutol', 'Ethionamide', 'Kanamycin',
#        'Levofloxacin', 'Linezolid', 'Moxifloxacin', 'Pyrazinamide',
#        'Rifampicin', 'Streptomycin']

# for drug in finished_drugs:
    
#     # subprocess.run(f"python3 analysis/01_combine_model_analyses.py {drug}", shell=True, encoding='utf8', stdout=subprocess.PIPE)
#     # assert os.path.isfile(os.path.join(analysis_dir, drug, "analysis_order.csv"))
#     # add_AF_models(drug)
#     df = pd.read_csv(os.path.join(analysis_dir, drug, "combined_analyses.csv"))
    
#     df_genos = pd.read_csv(os.path.join(analysis_dir, drug, "genos.csv.gz"), compression="gzip", usecols=["resolved_symbol", "variant_category", "predicted_effect", "position"]).drop_duplicates(subset=["resolved_symbol", "variant_category"], keep="first")
#     df_genos["mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
    
#     df["position"] = df["orig_variant"].map(dict(zip(df_genos["mutation"], df_genos["position"])))
#     if len(df.loc[(pd.isnull(df["position"])) & (~df["orig_variant"].str.contains("|".join(["inframe", "lof", "PC"])))]) > 0:
#         print(f"Problem with {drug}")
#     else:
#         df.to_csv(os.path.join(analysis_dir, drug, "combined_analyses.csv"), index=False)
#     print(f"Finished {drug}")