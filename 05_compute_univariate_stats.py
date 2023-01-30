import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import binomtest
import sys, glob, os, yaml, tracemalloc

# analysis utils is in the analysis folder
sys.path.append(os.path.join(os.getcwd(), "analysis"))
from stats_utils import *



def get_annotated_genos(analysis_dir, drug):
    '''
    This function gets the phenotype and genotype dataframes for a given drug. These are the same across models of the same type (i.e. all BINARY with WHO phenotypes, or all ATU models). 
    '''
 
    genos_files = glob.glob(os.path.join(analysis_dir, drug, "genos*.csv.gz"))
    print(f"{len(genos_files)} genotypes files")
    df_genos = pd.concat([pd.read_csv(fName, compression="gzip", low_memory=False, 
                                      usecols=["resolved_symbol", "variant_category", "predicted_effect", "position"]
                                     ) for fName in genos_files]).drop_duplicates()
    
    # get annotations for mutations to combine later. Exclude lof and inframe, these will be manually replaced later
    df_genos["mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
    del df_genos["resolved_symbol"]
    del df_genos["variant_category"]
    
    annotated_genos = df_genos.drop_duplicates(["mutation", "predicted_effect", "position"])[["mutation", "predicted_effect", "position"]]
    del df_genos
    
    return annotated_genos



def compute_statistics_single_model(model_path, model_suffix, df_phenos, annotated_genos, alpha=0.05):
    
    # read in the matrix of inputs and the coefficient outputs
    matrix = pd.read_pickle(os.path.join(model_path, "model_matrix.pkl"))
    res_df = pd.read_csv(os.path.join(model_path, f"model_analysis{model_suffix}.csv"))
    pool_type = model_path.split("_")[-1]
    
    # pivot to matrix and add sample IDs and phenotypes to the matrix
    matrix = matrix.merge(df_phenos[["sample_id", "phenotype"]], left_index=True, right_on="sample_id", how="left").reset_index(drop=True)

    # coefficient dictionary to keep track of which variants have positive and negative coefficients
    variant_coef_dict = dict(zip(res_df["mutation"], res_df["coef"]))

    # get dataframe of the univariate stats add them to the results dataframe
    full_predict_values = compute_univariate_stats(matrix[matrix.columns[~matrix.columns.str.contains("PC")]], variant_coef_dict)
    res_df = res_df.merge(full_predict_values, on="mutation", how="outer").drop_duplicates("mutation", keep="first")

    # add confidence intervals for all stats except the likelihood ratios
    res_df = compute_exact_confidence_intervals(res_df, alpha)

    # add confidence intervals for the likelihood ratios
    res_df = compute_likelihood_ratio_confidence_intervals(res_df, alpha)

    # get effect annotations and merge them with the results dataframe
    res_df = res_df.merge(annotated_genos, on="mutation", how="outer")
    res_df = res_df.loc[~pd.isnull(res_df["coef"])]

    if pool_type == "poolSeparate":
        # one pooled LOF feature
        res_df.loc[res_df["predicted_effect"].isin(["frameshift", "start_lost", "stop_gained", "feature_ablation"]),
                    "predicted_effect"
                   ] = "lof"
        # one pooled inframe feature
        res_df.loc[res_df["predicted_effect"].str.contains("inframe"),
                    "predicted_effect"
                   ] = "inframe"
    elif pool_type == "poolALL":
        # combine these predicted effects into a single lof_all feature
        res_df.loc[res_df["predicted_effect"].isin(["inframe_insertion", "inframe_deletion", "frameshift", "start_lost", "stop_gained", "feature_ablation"]),
                    "predicted_effect"
                   ] = "lof_all"

    res_df.loc[(res_df["mutation"].str.contains("lof")) & (~res_df["mutation"].str.contains("all")), "predicted_effect"] = "lof"
    res_df.loc[(res_df["mutation"].str.contains("inframe")) & (~res_df["mutation"].str.contains("all")), "predicted_effect"] = "inframe"
    res_df.loc[res_df["mutation"].str.contains("all"), "predicted_effect"] = "lof_all"

    # check that every mutation is present in at least 1 isolate
    if res_df.Num_Isolates.min() < 1:
        print("Some total numbers of isolates are less than 1")

    # predicted effect should only be NaN for PCs. position is NaN only for the pooled mutations and PCs
    assert len(res_df.loc[(pd.isnull(res_df["predicted_effect"])) & (~res_df["mutation"].str.contains("|".join(["PC"])))]) == 0
    assert len(res_df.loc[(pd.isnull(res_df["position"])) & (~res_df["mutation"].str.contains("|".join(["lof", "inframe", "PC"])))]) == 0

    # check that every mutation is present in at least 1 isolate
    assert res_df.Num_Isolates.min() > 0
    
    # check that there are no NaNs in the univariate statistics. Don't include LR+ upper and lower bounds because they can be NaN if LR+ = inf
    assert len(res_df.query("~mutation.str.contains('PC')")[pd.isnull(res_df[['Num_Isolates', 'Total_Isolates', 'TP', 'FP', 'TN', 'FN', 'PPV', 'NPV', 'Sens', 'Spec', 'LR+', 'LR-',
                                   'PPV_LB', 'PPV_UB', 'NPV_LB', 'NPV_UB', 'Sens_LB', 'Sens_UB', 'Spec_LB', 'Spec_UB', 'LR-_LB', 'LR-_UB']]).any(axis=1)]) == 0
        
    # check confidence intervals. LB ≤ var ≤ UB, and no confidence intervals have width 0. When the probability is 0 or 1, there are numerical precision issues
    # i.e. python says 1 != 1. So ignore those cases when checking
    for var in ["PPV", "NPV", "Sens", "Spec", "LR+", "LR-"]:
        assert len(res_df.loc[(~res_df[var].isin([0, 1])) & (res_df[var] < res_df[f"{var}_LB"])]) == 0
        assert len(res_df.loc[(~res_df[var].isin([0, 1])) & (res_df[var] > res_df[f"{var}_UB"])]) == 0
        
        width = res_df[f"{var}_UB"] - res_df[f"{var}_LB"]
        assert np.min(width) > 0
    
    # add significance. encodeAF not included in this one because only the HET == DROP models should be passed in here
    secondary_analysis_criteria = ["+2", "ALL", "unpooled", "poolALL", "withSyn"]
    significance_thresh = 0.05
    
    # if any of the secondary analysis criteria are met, change the threshold
    for string in secondary_analysis_criteria:
        if string in model_path:
            significance_thresh = 0.01
    
    res_df.loc[res_df["BH_pval"] < significance_thresh, "Significant"] = 1
    res_df["Significant"] = res_df["Significant"].fillna(0).astype(int)
    res_df = res_df.sort_values("Odds_Ratio", ascending=False).drop_duplicates("mutation", keep="first")
        
    del matrix
    
    # remove principal components, then convert the columns to int
    res_df = res_df.query("~mutation.str.contains('PC')")
    res_df[['Num_Isolates', 'Total_Isolates', 'TP', 'FP', 'TN', 'FN']] = res_df[['Num_Isolates', 'Total_Isolates', 'TP', 'FP', 'TN', 'FN']].astype(int)
    
    res_df[['mutation', 'predicted_effect', 'position', 'confidence', 'Odds_Ratio', 'pval', 'BH_pval',
       'Bonferroni_pval', 'Significant', 'Num_Isolates', 'Total_Isolates', 'TP', 'FP', 'TN', 'FN', 'PPV', 'NPV', 'Sens', 'Spec',
       'LR+', 'LR-', 'PPV_LB', 'PPV_UB', 'NPV_LB', 'NPV_UB', 'Sens_LB',
       'Sens_UB', 'Spec_LB', 'Spec_UB', 'LR+_LB', 'LR+_UB', 'LR-_LB', 'LR-_UB',
       ]].to_csv(os.path.join(model_path, f"model_analysis{model_suffix}.csv"), index=False)  

    
    


def add_significance_predicted_effect(model_path, annotated_genos, model_suffix):
    '''
    Use this function for models without binary inputs and outputs. This function just adds significance and predicted effect and saves the dataframe. 
    '''
    
    # get coefficient dataframe and add mutation predicted effects
    res_df = pd.read_csv(os.path.join(model_path, f"model_analysis{model_suffix}.csv"))
    res_df = res_df.merge(annotated_genos, on="mutation", how="outer")
    res_df = res_df.loc[~pd.isnull(res_df["coef"])]
    
    res_df.loc[(res_df["mutation"].str.contains("lof")) & (~res_df["mutation"].str.contains("all")), "predicted_effect"] = "lof"
    res_df.loc[(res_df["mutation"].str.contains("inframe")) & (~res_df["mutation"].str.contains("all")), "predicted_effect"] = "inframe"
    res_df.loc[res_df["mutation"].str.contains("all"), "predicted_effect"] = "lof_all"

    # use p-value threshold of 0.01 because this function will be used for secondary analyses (non-binary inputs (encodeAF) /outputs (MICs))
    res_df.loc[res_df["BH_pval"] < 0.01, "Significant"] = 1
    res_df["Significant"] = res_df["Significant"].fillna(0).astype(int)

    # no odds ratios, save coefficients. For both, drop principal components when saving the new dataframe
    if "MIC" in model_path:
        res_df.query("~mutation.str.contains('PC')")[['mutation', 'predicted_effect', 'position', 'confidence', 'coef', 'pval', 'BH_pval', 'Bonferroni_pval', 'Significant']].sort_values("coef", ascending=False).drop_duplicates("mutation", keep="first").to_csv(os.path.join(model_path, f"model_analysis{model_suffix}.csv"), index=False)
    # save odds ratios
    else:
        res_df.query("~mutation.str.contains('PC')")[['mutation', 'predicted_effect', 'position', 'confidence', 'Odds_Ratio', 'pval', 'BH_pval','Bonferroni_pval', 'Significant']].sort_values("Odds_Ratio", ascending=False).drop_duplicates("mutation", keep="first").to_csv(os.path.join(model_path, f"model_analysis{model_suffix}.csv"), index=False)

    

# starting the memory monitoring
tracemalloc.start()

_, drug, folder, analysis_dir = sys.argv

# this is the intermediate folder
folder = folder.upper()
assert folder in ["BINARY", "ATU", "MIC"]

# get all models to compute univariate statistics for
analysis_paths = []

for tier in os.listdir(os.path.join(analysis_dir, drug, folder)):
    
    # there are other folders in this folder. also, there is a file called significant_tiers=1+2_variants.csv
    if "tiers" in tier and os.path.isdir(os.path.join(analysis_dir, drug, folder, tier)):
        tiers_path = os.path.join(analysis_dir, drug, folder, tier)

        for level_1 in os.listdir(tiers_path):
            level1_path = os.path.join(analysis_dir, drug, folder, tier, level_1)

            if os.path.isfile(os.path.join(level1_path, "model_matrix.pkl")):
                analysis_paths.append(level1_path)

            for level_2 in os.listdir(level1_path):
                level2_path = os.path.join(analysis_dir, drug, folder, tier, level_1, level_2)

                if os.path.isfile(os.path.join(level2_path, "model_matrix.pkl")):
                    analysis_paths.append(level2_path)
                            
                        
phenos_file = os.path.join(analysis_dir, drug, f"phenos_{folder.lower()}.csv")    
df_phenos = pd.read_csv(phenos_file)
annotated_genos = get_annotated_genos(analysis_dir, drug)

    
if folder == "BINARY":
    print(f"\nComputing univariate statistics for {len(analysis_paths)} BINARY models")
    model_suffix = ""

    for model_path in analysis_paths:  
        if "dropAF" in model_path:
            #compute_statistics_single_model(model_path, df_phenos, df_genos, annotated_genos, model_suffix, alpha=0.05)
            compute_statistics_single_model(model_path, model_suffix, df_phenos, annotated_genos, alpha=0.05)
        else:
            add_significance_predicted_effect(model_path, annotated_genos, model_suffix)


elif folder == "ATU":
    print(f"\nComputing univariate statistics for {len(analysis_paths)*2} CC and CC-ATU models")

    for model_path in analysis_paths:  
        for model_suffix in ["_CC", "_CC_ATU"]:
            if "dropAF" in model_path:
                compute_statistics_single_model(model_path, df_phenos, df_genos, annotated_genos, model_suffix, alpha=0.05)
            else:
                add_significance_predicted_effect(model_path, annotated_genos, model_suffix)

else:
    print(f"\nComputing univariate statistics for {len(analysis_paths)} MIC models")
    model_suffix = ""

    # just add predicted effect and Significance to these because there are no positive or negative outputs (output is continuous MIC)
    for model_path in analysis_paths:
        add_significance_predicted_effect(model_path, annotated_genos, model_suffix)

        
# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")