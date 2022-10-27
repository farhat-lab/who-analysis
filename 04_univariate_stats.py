import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
import scipy.stats as st
import sys

import glob, os, yaml
import warnings
warnings.filterwarnings("ignore")
from memory_profiler import profile

# open file for writing memory logs to. Append to file, not overwrite
mem_log=open('memory_usage.log','a+')



def compute_predictive_values(combined_df, return_stats=[]):
    '''
    Compute positive predictive value. 
    Compute sensitivity, specificity, and positive and negative likelihood ratios. 
    
    PPV = true_positive / all_positive. NPV = true_negative / all_negative
    Sens = true_positive / (true_positive + false_negative)
    Spec = true_negative / (true_negative + false_positive)
    
    Also return the number of isolates with each variant = all_positive
    
    Positive LR = sens / (1 – spec)
    Negative LR = (1 – sens) / spec

    '''
    # make a copy to keep sample_id in one dataframe
    melted = combined_df.melt(id_vars=["sample_id", "phenotype"])
    melted_2 = melted.copy()
    del melted_2["sample_id"]
    
    # get counts of isolates grouped by phenotype and variant -- so how many isolates have a variant and have a phenotype (all 4 possibilities)
    grouped_df = pd.DataFrame(melted_2.groupby(["phenotype", "variable"]).value_counts()).reset_index()
    grouped_df = grouped_df.rename(columns={"variable": "orig_variant", "value": "variant", 0:"count"})
    
    # dataframes of the counts of the 4 values
    true_pos_df = grouped_df.query("variant == 1 & phenotype == 1").rename(columns={"count": "TP"})
    false_pos_df = grouped_df.query("variant == 1 & phenotype == 0").rename(columns={"count": "FP"})
    true_neg_df = grouped_df.query("variant == 0 & phenotype == 0").rename(columns={"count": "TN"})
    false_neg_df = grouped_df.query("variant == 0 & phenotype == 1").rename(columns={"count": "FN"})

    assert len(true_pos_df) + len(false_pos_df) + len(true_neg_df) + len(false_neg_df) == len(grouped_df)
    
    # combine the 4 dataframes into a single dataframe (concatenating on axis = 1)
    final = true_pos_df[["orig_variant", "TP"]].merge(
            false_pos_df[["orig_variant", "FP"]], on="orig_variant", how="outer").merge(
            true_neg_df[["orig_variant", "TN"]], on="orig_variant", how="outer").merge(
            false_neg_df[["orig_variant", "FN"]], on="orig_variant", how="outer").fillna(0)

    assert len(final) == len(melted["variable"].unique())
    assert len(final) == len(final.drop_duplicates("orig_variant"))
    
    # check that all feature rows have the same number of samples    
    assert len(np.unique(final[["TP", "FP", "TN", "FN"]].sum(axis=1))) == 1
        
    final["Num_Isolates"] = final["TP"] + final["FP"]
    final["Total_Isolates"] = final["TP"] + final["FP"] + final["TN"] + final["FN"]
    final["PPV"] = final["TP"] / (final["TP"] + final["FP"])
    final["Sens"] = final["TP"] / (final["TP"] + final["FN"])
    final["Spec"] = final["TN"] / (final["TN"] + final["FP"])
    final["LR+"] = final["Sens"] / (1 - final["Spec"])
    final["LR-"] = (1 - final["Sens"]) / final["Spec"]
    #final["NPV"] = final["TN"] / (final["TN"] + final["FN"])
    
    if len(return_stats) == 0:
        return final[["orig_variant", "Num_Isolates", "Total_Isolates", "TP", "FP", "TN", "FN", "PPV", "Sens", "Spec", "LR+", "LR-"]]
    else:
        return final[return_stats]
    
    

def compute_univariate_stats(**kwargs):
    
    tiers_lst = kwargs["tiers_lst"]
    pheno_category_lst = kwargs["pheno_category_lst"]
    model_prefix = kwargs["model_prefix"]
    het_mode = kwargs["het_mode"]
    synonymous = kwargs["synonymous"]
    pool_lof = kwargs["pool_lof"]
    AF_thresh = kwargs["AF_thresh"]

    num_PCs = kwargs["num_PCs"]
    num_bootstrap = kwargs["num_bootstrap"]
    alpha = kwargs["alpha"]

    # model_analysis file with all nominally significant variants
    res_df = pd.read_csv(os.path.join(out_dir, "model_analysis.csv"))
    
    # read in all genotypes and phenotypes and combine into a single dataframe
    model_inputs = pd.read_pickle(os.path.join(out_dir, "model_matrix.pkl"))
    df_phenos = pd.read_csv(os.path.join(out_dir, "phenos.csv"))
    combined = model_inputs.merge(df_phenos[["sample_id", "phenotype"]], on="sample_id").reset_index(drop=True)

    # compute univariate stats for only the lof variable
    if pool_lof:
        keep_variants = list(res_df.loc[res_df["orig_variant"].str.contains("lof")]["orig_variant"].values)
    else:
        keep_variants = list(res_df.loc[~res_df["orig_variant"].str.contains("PC")]["orig_variant"].values)
        
    # check that all samples were preserved
    combined_small = combined[["sample_id", "phenotype"] + keep_variants]
    assert len(combined_small) == len(combined)
    
    #### Compute univariate statistics only for cases where genotypes are binary (no AF), synonymous are included, all features ####
    #### For LOF, only compute univariate stats for the LOF variables. Otherwise, the corresponding non-LOF model contains everything #### 
    #### In the LOF case, if no LOF variants (there is 1 LOF per gene) are significant, then keep_variants = [], and we don't run this block of code ####
    #if (het_mode != "AF") & (synonymous == True) and (len(tiers_lst) > 1) and (len(keep_variants) > 0):
    if (het_mode != "AF") & (len(keep_variants) > 0):
        
        # get dataframe of predictive values for the non-zero coefficients and add them to the results dataframe
        full_predict_values = compute_predictive_values(combined_small)
        res_df = res_df.merge(full_predict_values, on="orig_variant", how="outer")

        print(f"Computing and bootstrapping predictive values with {num_bootstrap} replicates")
        bs_results = pd.DataFrame(columns = keep_variants)

        # need confidence intervals for 5 stats: PPV, sens, spec, + likelihood ratio, - likelihood ratio
        for i in range(num_bootstrap):

            # get bootstrap sample
            bs_idx = np.random.choice(np.arange(0, len(combined_small)), size=len(combined_small), replace=True)
            bs_combined = combined_small.iloc[bs_idx, :]

            # check ordering of features because we're just going to append bootstrap dataframes
            assert sum(bs_combined.columns[2:] != bs_results.columns) == 0

            # get predictive values from the dataframe of bootstrapped samples. Only return the 5 we want CI for, and the variant
            bs_values = compute_predictive_values(bs_combined, return_stats=["orig_variant", "PPV", "Sens", "Spec", "LR+", "LR-"])
            bs_results = pd.concat([bs_results, bs_values.set_index("orig_variant").T], axis=0)
            
            if i % int(num_bootstrap / 10) == 0:
                print(i)

        # check this because have had some issues with np.nanpercentile giving an error about incompatible data types
        bs_results = bs_results.astype(float)
                
        # add the confidence intervals to the dataframe
        for variable in ["PPV", "Sens", "Spec", "LR+", "LR-"]:

            lower, upper = np.nanpercentile(bs_results.loc[variable, :], q=[2.5, 97.5], axis=0)

            # LR+ can be infinite if spec is 1, and after percentile, it will be NaN, so replace with infinity
            if variable == "LR+":
                res_df[variable] = res_df[variable].fillna(np.inf)
                lower[np.isnan(lower)] = np.inf
                upper[np.isnan(upper)] = np.inf

            res_df = res_df.merge(pd.DataFrame({"orig_variant": bs_results.columns, 
                                f"{variable}_LB": lower,
                                f"{variable}_UB": upper,
                               }), on="orig_variant", how="outer")

            # sanity checks -- lower bounds should be <= true values, and upper bounds should be >= true values
            assert sum(res_df[variable] < res_df[f"{variable}_LB"]) == 0
            assert sum(res_df[variable] > res_df[f"{variable}_UB"]) == 0
            
    return res_df


_, config_file, drug = sys.argv

kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
pheno_category_lst = kwargs["pheno_category_lst"]
model_prefix = kwargs["model_prefix"]
het_mode = kwargs["het_mode"]
synonymous = kwargs["synonymous"]

out_dir = '/n/data1/hms/dbmi/farhat/ye12/who/analysis'
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
else:
    phenos_name = "WHO"

out_dir = os.path.join(out_dir, drug, f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}", model_prefix)

if not os.path.isdir(out_dir):
    print("No model for this analysis")
    exit()

# run analysis
model_analysis_univariate_stats = compute_univariate_stats(**kwargs)

# save, overwriting the original dataframe
model_analysis_univariate_stats.to_csv(os.path.join(out_dir, "model_analysis.csv"), index=False)