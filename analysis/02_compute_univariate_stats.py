import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
import scipy.stats as st
import sys

import glob, os, yaml
import warnings
warnings.filterwarnings("ignore")
import tracemalloc

# starting the memory monitoring
tracemalloc.start()
analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'


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
    


def compute_univariate_stats(drug, analysis_dir, num_bootstrap=1000):

    # final_analysis file with all significant variants for a drug
    res_df = pd.read_csv(os.path.join(analysis_dir, drug, "final_analysis.csv"))
    df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv"))
    df_genos = pd.read_csv(os.path.join(analysis_dir, drug, "genos.csv.gz"), compression="gzip")
    df_genos["orig_variant"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
    df_copy = df_genos.copy()

    # pool LOF and inframe mutations
    df_copy.loc[df_copy["predicted_effect"].isin(["frameshift", "start_lost", "stop_gained", "feature_ablation"]), ["variant_category", "position"]] = ["lof", np.nan]
    df_copy.loc[df_copy["predicted_effect"].isin(["inframe_insertion", "inframe_deletion"]), ["variant_category", "position"]] = ["inframe", np.nan]

    # update the orig_variant column using the new variant categories (lof and inframe) and combine the pooled and unpooled variants
    df_copy["orig_variant"] = df_copy["resolved_symbol"] + "_" + df_copy["variant_category"]
    df_pooled = df_copy.query(f"variant_category in ['lof', 'inframe']").sort_values(by=["variant_binary_status", "variant_allele_frequency"], ascending=False, na_position="last").drop_duplicates(subset=["sample_id", "orig_variant"], keep="first")
    del df_copy
    df_genos_full = pd.concat([df_genos, df_pooled], axis=0)
    del df_genos
    del df_pooled
    
    # keep only variants that are in the final_analysis dataframe and drop NaNs (NaNs = either isolate didn't pass QC or it's a Het) 
    # We can't process Hets here because they need to be binary to have univariate statistics
    df_genos_full = df_genos_full.loc[(df_genos_full["orig_variant"].isin(res_df["orig_variant"].values))].dropna(subset="variant_binary_status")
    
    # check that the only variants that are in res_df but not in df_genos_full are the principal components
    if sum(~pd.Series(list(set(res_df["orig_variant"]) - set(df_genos_full["orig_variant"]))).str.contains("PC")) > 0:
        raise ValueError("Variants are missing from df_genos_full!")
        
    combined = df_genos_full.pivot(index="sample_id", columns="orig_variant", values="variant_binary_status")
    
    # predicted effect annotations for later
    annotated_genos = df_genos_full.query("variant_category not in ['lof', 'inframe']").drop_duplicates(["orig_variant", "predicted_effect"])[["orig_variant", "predicted_effect"]]
    del df_genos_full
    combined = combined.merge(df_phenos[["sample_id", "phenotype"]], left_index=True, right_on="sample_id").reset_index(drop=True)
        
    # get dataframe of predictive values for the non-zero coefficients and add them to the results dataframe
    full_predict_values = compute_predictive_values(combined)
    res_df = res_df.merge(full_predict_values, on="orig_variant", how="outer")
    
    # save the analysis dataframe with the univariate stats. Do this in case an error occurs during the 
    res_df.to_csv(os.path.join(analysis_dir, drug, "final_analysis.csv"), index=False)

    print(f"Computing and bootstrapping predictive values with {num_bootstrap} replicates")
    # Can't compute univariate stats for PCs. For tractability, only compute bootstrap stats for variants with positive coefficients.
    # The stats for variants with negative coefficients are often edge numbers and not informative, so this saves time. 
    keep_variants = list(res_df.loc[(~res_df["orig_variant"].str.contains("PC")) & (res_df["coef"] > 0)]["orig_variant"].values)
    
    # Remake this dataframe with fewer features because only going to bootstrap stats for variants with positive coefficients.
    # check that all samples were preserved. 
    combined_small = combined[["sample_id", "phenotype"] + keep_variants]
    assert len(combined_small) == len(combined)
    
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

    # ensure everything is float because had some issues with np.nanpercentile giving an error about incompatible data types
    bs_results = bs_results.astype(float)
    if len(bs_results.index.unique()) != 5:
        print(bs_results.index.unique())

    # add the confidence intervals to the dataframe
    for variable in ["PPV", "Sens", "Spec", "LR+", "LR-"]:

        lower, upper = np.nanpercentile(bs_results.loc[variable, :], q=[2.5, 97.5], axis=0)

        # LR+ can be infinite if spec is 1, and after percentile, it will be NaN, so replace with infinity. Ignore principal components because they will be NaN
        if variable == "LR+":
            res_df.loc[~res_df["orig_variant"].str.contains("PC"), variable] = res_df.loc[~res_df["orig_variant"].str.contains("PC"), variable].fillna(np.inf)
            lower[np.isnan(lower)] = np.inf
            upper[np.isnan(upper)] = np.inf

        res_df = res_df.merge(pd.DataFrame({"orig_variant": bs_results.columns, 
                            f"{variable}_LB": lower,
                            f"{variable}_UB": upper,
                           }), on="orig_variant", how="outer")

        # sanity checks -- lower bounds should be <= true values, and upper bounds should be >= true values
        # numerical precision can make this fail though, so commented out for now
        # assert sum(res_df[variable] < res_df[f"{variable}_LB"]) == 0
        # assert sum(res_df[variable] > res_df[f"{variable}_UB"]) == 0
        
    # get effect annotations and merge them with the results dataframe
    final_res = res_df.merge(annotated_genos, on="orig_variant", how="outer")
    final_res = final_res.loc[~pd.isnull(final_res["coef"])]
    final_res.loc[final_res["orig_variant"].str.contains("lof"), "predicted_effect"] = "lof"
    final_res.loc[final_res["orig_variant"].str.contains("inframe"), "predicted_effect"] = "inframe"

    if len(set(final_res.orig_variant).symmetric_difference(res_df.orig_variant)) != 0:
        raise ValueError("Not all mutations have associated effects!")
    else:
        del res_df

    final_res.sort_values("coef", ascending=False).to_csv(os.path.join(analysis_dir, drug, "final_analysis.csv"), index=False)


_, drug = sys.argv

# run analysis
model_analysis_univariate_stats = compute_univariate_stats(drug, analysis_dir, num_bootstrap=1000)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()

# write peak memory usage in GB
with open("memory_usage.log", "a+") as file:
    file.write(f"{os.path.basename(__file__)}: {script_memory} GB\n")