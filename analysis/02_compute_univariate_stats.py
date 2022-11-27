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


def compute_univariate_stats(combined_df, variant_coef_dict, return_stats=[]):
    '''
    Compute positive predictive value. 
    Compute sensitivity, specificity, and positive and negative likelihood ratios. 
    
    PPV = true_positive / all_positive
    NPV = true_negative / all_negative
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
    grouped_df = grouped_df.rename(columns={"variable": "variant", "value": "present", 0:"count"})
    
    # add coefficients, create new column for the switched phenotypes (keep the old ones in actual_pheno)
    grouped_df["coef"] = grouped_df["variant"].map(variant_coef_dict)
    grouped_df["actual_pheno"] = grouped_df["phenotype"].copy()
    assert sum(grouped_df["phenotype"] != grouped_df["actual_pheno"]) == 0

    # switch sign of the phenotypes for the negative coefficients and check
    grouped_df.loc[grouped_df["coef"] < 0, "phenotype"] = 1 - grouped_df.loc[grouped_df["coef"] < 0, "actual_pheno"]
    assert sum(grouped_df["phenotype"] != grouped_df["actual_pheno"]) == len(grouped_df.query("coef < 0"))
    
    # dataframes of the counts of the 4 values
    true_pos_df = grouped_df.query("present == 1 & phenotype == 1").rename(columns={"count": "TP"})
    false_pos_df = grouped_df.query("present == 1 & phenotype == 0").rename(columns={"count": "FP"})
    true_neg_df = grouped_df.query("present == 0 & phenotype == 0").rename(columns={"count": "TN"})
    false_neg_df = grouped_df.query("present == 0 & phenotype == 1").rename(columns={"count": "FN"})

    assert len(true_pos_df) + len(false_pos_df) + len(true_neg_df) + len(false_neg_df) == len(grouped_df)
    
    # combine the 4 dataframes into a single dataframe (concatenating on axis = 1)
    final = true_pos_df[["variant", "TP"]].merge(
            false_pos_df[["variant", "FP"]], on="variant", how="outer").merge(
            true_neg_df[["variant", "TN"]], on="variant", how="outer").merge(
            false_neg_df[["variant", "FN"]], on="variant", how="outer").fillna(0)

    assert len(final) == len(melted["variable"].unique())
    assert len(final) == len(final.drop_duplicates("variant"))
        
    final["Num_Isolates"] = final["TP"] + final["FP"]
    final["Total_Isolates"] = final[["TP", "FP", "TN", "FN"]].sum(axis=1)
    final["PPV"] = final["TP"] / (final["TP"] + final["FP"])
    final["NPV"] = final["TN"] / (final["TN"] + final["FN"])
    final["Sens"] = final["TP"] / (final["TP"] + final["FN"])
    final["Spec"] = final["TN"] / (final["TN"] + final["FP"])
    final["LR+"] = final["Sens"] / (1 - final["Spec"])
    final["LR-"] = (1 - final["Sens"]) / final["Spec"]
    
    if len(return_stats) == 0:
        return final[["variant", "Num_Isolates", "Total_Isolates", "TP", "FP", "TN", "FN", "PPV", "NPV", "Sens", "Spec", "LR+", "LR-"]]
    else:
        return final[return_stats]
    


def compute_wilson_conf_interval(var, alpha, res_df):
    '''
    Compute confidence intervals for values that range between 0 and 1. Use this for sensitivity, specificity, and PPV. 
    Both the normal approximation (i.e. Wald interval) and  Wilson interval are used for such binomial cases. 
    But the Wilson interval is preferred when the number of trials (big N) is small or the probabilities are extreme (i.e. 0 or 1)
    
    There are many cases of extreme sensitivity, specificity, or PPV in this analysis, so we will use Wilson intervals. 
    '''
    
    if var == "PPV":
        N = res_df["TP"] + res_df["FP"]
    elif var == "NPV":
        N = res_df["TN"] + res_df["FN"]
    elif var == "Sens":
        N = res_df["TP"] + res_df["FN"]
    elif var == "Spec":
        N = res_df["TN"] + res_df["FP"]
        
    z = np.abs(st.t.ppf(q=alpha/2, df=N, loc=0, scale=1))
    
    center = 1 / (1 + z**2 / N) * (res_df[var] + z**2 / (2 * N))

    error = z / (1 + z**2 / N) * np.sqrt(res_df[var] * (1 - res_df[var]) / N + z**2 / (4 * N**2))
        
    return (center - error, center + error)


    
def compute_likelihood_ratio_confidence_intervals(alpha, res_df):
    
    z = np.abs(st.t.ppf(q=alpha/2, df=res_df["Total_Isolates"]-1, loc=0, scale=1))
    
    LRpos_error = np.exp(z * np.sqrt(1/res_df["TP"] - 1/(res_df["TP"] + res_df["FN"]) + 1/res_df["FP"] - 1/(res_df["FP"] + res_df["TN"])))
    LRneg_error = np.exp(z * np.sqrt(1/res_df["FN"] - 1/(res_df["TP"] + res_df["FN"]) + 1/res_df["TN"] - 1/(res_df["FP"] + res_df["TN"])))
    
    res_df["LR+_LB"] = res_df["LR+"] / LRpos_error
    res_df["LR+_UB"] = res_df["LR+"] * LRpos_error
    
    res_df["LR-_LB"] = res_df["LR-"] / LRneg_error
    res_df["LR-_UB"] = res_df["LR-"] * LRneg_error

    return res_df



def run_all(drug, analysis_dir, alpha=0.05):

    # final_analysis file with all significant variants for a drug
    res_df = pd.read_csv(os.path.join(analysis_dir, drug, "final_analysis.csv"))
    insig_df = pd.read_csv(os.path.join(analysis_dir, drug, "all_insignificant_features.csv"))
    res_df = pd.concat([res_df, insig_df], axis=0)
    del insig_df
    
    df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv"))
    df_genos = pd.read_csv(os.path.join(analysis_dir, drug, "genos.csv.gz"), compression="gzip")
    df_genos["variant"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
    df_copy = df_genos.copy()

    # pool LOF and inframe mutations
    df_copy.loc[df_copy["predicted_effect"].isin(["frameshift", "start_lost", "stop_gained", "feature_ablation"]), ["variant_category", "position"]] = ["lof", np.nan]
    df_copy.loc[df_copy["predicted_effect"].isin(["inframe_insertion", "inframe_deletion"]), ["variant_category", "position"]] = ["inframe", np.nan]

    # update the variant column using the new variant categories (lof and inframe) and combine the pooled and unpooled variants
    df_copy["variant"] = df_copy["resolved_symbol"] + "_" + df_copy["variant_category"]
    df_pooled = df_copy.query(f"variant_category in ['lof', 'inframe']").sort_values(by=["variant_binary_status", "variant_allele_frequency"], ascending=False, na_position="last").drop_duplicates(subset=["sample_id", "variant"], keep="first")
    del df_copy
    df_genos_full = pd.concat([df_genos, df_pooled], axis=0)
    del df_genos
    del df_pooled
    
    # keep only variants that are in the final_analysis dataframe and drop NaNs (NaNs = either isolate didn't pass QC or it's a Het) 
    # We can't process Hets here because they need to be binary to have univariate statistics
    df_genos_full = df_genos_full.loc[(df_genos_full["variant"].isin(res_df["variant"].values))].dropna(subset="variant_binary_status")
    
    # check that the only variants that are in res_df but not in df_genos_full are the principal components
    if sum(~pd.Series(list(set(res_df["variant"]) - set(df_genos_full["variant"]))).str.contains("PC")) > 0:
        raise ValueError("Variants are missing from df_genos_full!")
        
    combined = df_genos_full.pivot(index="sample_id", columns="variant", values="variant_binary_status")
    
    # predicted effect annotations for later
    annotated_genos = df_genos_full.query("variant_category not in ['lof', 'inframe']").drop_duplicates(["variant", "predicted_effect", "position"])[["variant", "predicted_effect", "position"]]
    del df_genos_full
    combined = combined.merge(df_phenos[["sample_id", "phenotype"]], left_index=True, right_on="sample_id").reset_index(drop=True)
    
    # Can't compute univariate stats for PCs or for variants in the HET models. Because the mutations were encoded with AF, there are no positives and negatives
    keep_variants = list(res_df.loc[(~res_df["variant"].str.contains("PC")) & (res_df["HET"] != 'AF')]["variant"].values)
    combined = combined[["sample_id", "phenotype"] + keep_variants]
        
    # coefficient dictionary to keep track of which variants have positive and negative coefficients
    variant_coef_dict = dict(zip(res_df["variant"], res_df["coef"]))

    # get dataframe of the univariate stats add them to the results dataframe
    full_predict_values = compute_univariate_stats(combined, variant_coef_dict)
    res_df = res_df.merge(full_predict_values, on="variant", how="outer").drop_duplicates("variant", keep="first")
    
    # save the analysis dataframe with the univariate stats. Do this in case an error occurs during the 
    res_df.to_csv(os.path.join(analysis_dir, drug, "final_analysis_with_univariate.csv"), index=False)
    
    # add confidence intervals for all stats except the likelihood ratios
    for var in ["PPV", "NPV", "Sens", "Spec"]:
        
        var_lb, var_ub = compute_wilson_conf_interval(var, alpha, res_df)
        
        # in case there are any numbers outside the range [0, 1] 
        res_df[f"{var}_LB"] = np.max([var_lb, np.zeros(len(res_df))], axis=0)
        res_df[f"{var}_UB"] = np.min([var_ub, np.ones(len(res_df))], axis=0)

    # add confidence intervals for the likelihood ratios
    res_df = compute_likelihood_ratio_confidence_intervals(alpha, res_df)
        
    # get effect annotations and merge them with the results dataframe
    final_res = res_df.merge(annotated_genos, on="variant", how="outer")
    final_res = final_res.loc[~pd.isnull(final_res["coef"])]
    final_res.loc[final_res["variant"].str.contains("lof"), "predicted_effect"] = "lof"
    final_res.loc[final_res["variant"].str.contains("inframe"), "predicted_effect"] = "inframe"

    if len(set(final_res["variant"]).symmetric_difference(res_df["variant"])) != 0:
        raise ValueError("Not all mutations have associated effects!")
    else:
        del res_df

    final_res.sort_values("coef", ascending=False).drop_duplicates("variant", keep="first").to_csv(os.path.join(analysis_dir, drug, "final_analysis_with_univariate.csv"), index=False)


_, drug = sys.argv

# run analysis
run_all(drug, analysis_dir, alpha=0.05)
print(f"Finished {drug}!")

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()

# write peak memory usage in GB
with open("memory_usage.log", "a+") as file:
    file.write(f"{os.path.basename(__file__)}: {script_memory} GB\n")
