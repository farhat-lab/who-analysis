import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import binomtest
import sys, glob, os, yaml, tracemalloc, warnings, argparse
warnings.filterwarnings("ignore")

# utils files are in a separate folder
sys.path.append("utils")
from stats_utils import *



def get_annotated_genos(analysis_dir, drug):
    '''
    This function gets annotations (predicted effect and position) for mutations to merge them into the final analysis dataframes
    '''
 
    genos_files = glob.glob(os.path.join(analysis_dir, drug, "genos*.csv.gz"))
    print(f"{len(genos_files)} genotypes files")
    df_genos = pd.concat([pd.read_csv(fName, compression="gzip", low_memory=False, 
                                      usecols=["resolved_symbol", "variant_category", "predicted_effect", "position"]
                                     ) for fName in genos_files]).drop_duplicates()
    
    # get annotations for mutations to combine later. Exclude LoF and inframe, these will be manually replaced later
    df_genos["mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
    del df_genos["resolved_symbol"]
    del df_genos["variant_category"]
    
    annotated_genos = df_genos.drop_duplicates(["mutation", "predicted_effect", "position"])[["mutation", "predicted_effect", "position"]]
    del df_genos
    
    return annotated_genos




def compute_statistics_single_model(model_analysis_file, df_phenos, annotated_genos, encodeAF=False, alpha=0.05):
    
    # read in the matrix of inputs and the coefficient outputs
    model_path = os.path.dirname(model_analysis_file)
    matrix = pd.read_pickle(os.path.join(model_path, "model_matrix.pkl"))
    
    # if the model is an encodeAF model, consider present as AFs > 0.25 and absent as AFs <= 0.25
    # normally, AFs > 0.75 are considered present (variant_binary_status = 1)
    if encodeAF:
        matrix[matrix > 0.25] = 1
        matrix[matrix <= 0.25] = 0

    # check that the matrix is now binary
    assert len(np.unique(matrix.values)) == 2
    
    res_df = pd.read_csv(model_analysis_file)
    pool_type = model_path.split("_")[-1]
    
    # previous naming convention
    del_cols = ["TP", "TN", "FP", "FN"]
    for col in del_cols:
        if col in res_df.columns:
            del res_df[col]
        
    # add sample IDs and phenotypes to the matrix
    matrix = matrix.merge(df_phenos[["sample_id", "phenotype"]], left_index=True, right_on="sample_id", how="inner").reset_index(drop=True)

    # get dataframe of the univariate stats add them to the results dataframe
    full_predict_values = compute_univariate_stats(matrix[matrix.columns[~matrix.columns.str.contains("PC")]])
    
    # old versions will be dropped because the new versions are more updated (because changed the phenotype flipping back to the original, where 1 = R, 0 = S)
    res_df = res_df.merge(full_predict_values, on="mutation", how="outer", suffixes=('_DROP', '')).drop_duplicates("mutation", keep="first")
    res_df = res_df[res_df.columns[~res_df.columns.str.contains("_DROP")]]

    # add confidence intervals for all stats except the likelihood ratios
    res_df = compute_exact_confidence_intervals(res_df, alpha)

    # add confidence intervals for the likelihood ratios
    res_df = compute_likelihood_ratio_confidence_intervals(res_df, alpha)

    # get effect annotations and merge them with the results dataframe
    res_df = res_df.merge(annotated_genos, on="mutation", how="outer", suffixes=('', '_DROP'))
    res_df = res_df[res_df.columns[~res_df.columns.str.contains("_DROP")]]
    
    res_df = res_df.loc[~pd.isnull(res_df["coef"])]

    if pool_type == "poolSeparate":
        # one pooled LOF feature
        res_df.loc[res_df["predicted_effect"].isin(["frameshift", "start_lost", "stop_gained", "feature_ablation"]),
                    "predicted_effect"
                   ] = "LoF"
        # one pooled inframe feature
        res_df.loc[(~pd.isnull(res_df["predicted_effect"])) & (res_df["predicted_effect"].str.contains("inframe")),
                    "predicted_effect"
                   ] = "inframe"
    elif pool_type == "poolALL":
        # combine these predicted effects into a single LoF_all feature
        res_df.loc[res_df["predicted_effect"].isin(["inframe_insertion", "inframe_deletion", "frameshift", "start_lost", "stop_gained", "feature_ablation"]),
                    "predicted_effect"
                   ] = "LoF_all"

    # fix LoF naming convention
    res_df.loc[(res_df["mutation"].str.contains("LoF")) & (~res_df["mutation"].str.contains("all")), "predicted_effect"] = "LoF"
    res_df.loc[(res_df["mutation"].str.contains("inframe")) & (~res_df["mutation"].str.contains("all")), "predicted_effect"] = "inframe"
    res_df.loc[res_df["mutation"].str.contains("all"), "predicted_effect"] = "LoF_all"

    # predicted effect should only be NaN for PCs. position is NaN only for the pooled mutations and PCs
    assert len(res_df.loc[(pd.isnull(res_df["predicted_effect"])) & (~res_df["mutation"].str.contains("|".join(["PC"])))]) == 0
    assert len(res_df.loc[(pd.isnull(res_df["position"])) & (~res_df["mutation"].str.contains("|".join(["LoF", "inframe", "PC"])))]) == 0

    # check that every mutation is present in at least 1 isolate
    assert res_df.Num_Isolates.min() > 0

    # check that there are no NaNs in the univariate statistics. Don't include LR+ upper and lower bounds because they can be NaN if LR+ = inf
    assert len(res_df.loc[~res_df["mutation"].str.contains("PC")].loc[pd.isnull(res_df[['Num_Isolates', "Mut_R", "Mut_S", "NoMut_S", "NoMut_R", 'R_PPV', 'S_PPV', 'NPV', 'Sens', 'Spec', 'LR+', 'LR-', 'R_PPV_LB', 'R_PPV_UB', 'S_PPV_LB', 'S_PPV_UB', 'Sens_LB', 'Sens_UB', 'Spec_LB', 'Spec_UB', 'LR-_LB', 'LR-_UB']]).any(axis=1)]) == 0

    # check confidence intervals. LB ≤ var ≤ UB, and no confidence intervals have width 0. When the probability is 0 or 1, there are numerical precision issues
    # i.e. python says 1 != 1. So ignore those cases when checking
    for var in ["R_PPV", "S_PPV", "NPV", "Sens", "Spec", "LR+", "LR-"]:
        assert len(res_df.loc[(~res_df[var].isin([0, 1])) & (res_df[var] < res_df[f"{var}_LB"])]) == 0
        assert len(res_df.loc[(~res_df[var].isin([0, 1])) & (res_df[var] > res_df[f"{var}_UB"])]) == 0

        width = res_df[f"{var}_UB"] - res_df[f"{var}_LB"]
        assert np.min(width) > 0

    res_df = res_df.sort_values("Odds_Ratio", ascending=False).drop_duplicates("mutation", keep="first")
    del matrix

    res_df[['mutation', 'predicted_effect', 'position', 'coef', 'Odds_Ratio', 'pval', 'BH_pval',
       'Bonferroni_pval', 'neutral_pval', 'BH_neutral_pval', 'Bonferroni_neutral_pval', 'Num_Isolates', "Mut_R", "Mut_S", "NoMut_S", "NoMut_R", 'R_PPV', 'S_PPV', 'NPV', 'Sens', 'Spec',
       'LR+', 'LR-', 'R_PPV_LB', 'R_PPV_UB', 'S_PPV_LB', 'S_PPV_UB', 'NPV_LB', 'NPV_UB', 'Sens_LB',
       'Sens_UB', 'Spec_LB', 'Spec_UB', 'LR+_LB', 'LR+_UB', 'LR-_LB', 'LR-_UB',
       ]].to_csv(os.path.join(model_analysis_file), index=False)



# starting the memory monitoring
tracemalloc.start()

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", dest='config_file', default='config.ini', type=str, required=True)

cmd_line_args = parser.parse_args()
config_file = cmd_line_args.config_file

kwargs = yaml.safe_load(open(config_file))
analysis_dir = kwargs["output_dir"]
binary = kwargs["binary"]
atu_analysis = kwargs["atu_analysis"]

if not binary:
    model_type = "MIC"
else:
    if atu_analysis:
        folder = "ATU"
    else:
        folder = "BINARY"

if model_type == "MIC":
    print("There are no univariate statistics to add for MIC models. Quitting this script.")
    exit()

# get all models to compute univariate statistics for
analysis_paths = []

for tier in os.listdir(os.path.join(analysis_dir, drug, folder)):
    
    # there are other folders in this folder. also, there is a file called significant_tiers=1+2_variants.csv
    if "tiers" in tier and os.path.isdir(os.path.join(analysis_dir, drug, folder, tier)):
        tiers_path = os.path.join(analysis_dir, drug, folder, tier)

        # level_1 = phenotypes (if it exists) 
        for level_1 in os.listdir(tiers_path):
            level1_path = os.path.join(analysis_dir, drug, folder, tier, level_1)

            if os.path.isfile(os.path.join(level1_path, "model_matrix.pkl")):
                analysis_paths.append(level1_path)

            # level_2 = model names
            for level_2 in os.listdir(level1_path):
                level2_path = os.path.join(analysis_dir, drug, folder, tier, level_1, level_2)

                if os.path.isfile(os.path.join(level2_path, "model_matrix.pkl")):
                    analysis_paths.append(level2_path)
                            
                        
phenos_file = os.path.join(analysis_dir, drug, f"phenos_{folder.lower()}.csv")    
df_phenos = pd.read_csv(phenos_file)
annotated_genos = get_annotated_genos(analysis_dir, drug)
    
for model_path in analysis_paths:
    for fName in glob.glob(os.path.join(model_path, "model_analysis.csv")):

        if model_type == "AF":
            if "encodeAF" in model_path:
                print(model_path)
                compute_statistics_single_model(fName, df_phenos, annotated_genos, encodeAF=True, alpha=0.05)
        else:
            if "dropAF" in model_path:
                print(model_path)
                compute_statistics_single_model(fName, df_phenos, annotated_genos, encodeAF=False, alpha=0.05)        

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")
