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

who_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_resistance_variants_all.csv", usecols=["drug", "confidence", "variant"])
variant_mapping = pd.read_csv("data/v1_to_v2_variants_mapping.csv", usecols=["gene_name", "variant", "raw_variant_mapping_data.variant_category"])
variant_mapping.columns = ["gene", "V1", "V2"]
variant_mapping["mutation"] = variant_mapping["gene"] + "_" + variant_mapping["V2"]

# combine with the new names to get a dataframe with the confidence leve,s and variant mappings between 2021 and 2022
who_variants_combined = who_variants.merge(variant_mapping[["V1", "mutation"]], left_on="variant", right_on="V1", how="inner")
del who_variants_combined["variant"]

assert len(set(who_variants_combined["V1"]).symmetric_difference(set(who_variants["variant"]))) == 0
who_variants_combined = who_variants_combined.drop_duplicates(subset="mutation")
del who_variants_combined["V1"]
del who_variants


def get_genos_phenos(analysis_dir, drug):
    '''
    This function gets the phenotype and genotype dataframes for a given drug. These are the same across models. 
    
        who_only: boolean of whether to include only WHO-approved phenotypes or not. 
    '''
        
    df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv"))
    df_genos = pd.read_csv(os.path.join(analysis_dir, drug, "genos.csv.gz"), compression="gzip")

    # get pooled variants
    df_genos["variant"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
    df_copy = df_genos.copy()

    # pool LOF and inframe mutations
    df_copy.loc[df_copy["predicted_effect"].isin(["frameshift", "start_lost", "stop_gained", "feature_ablation"]), ["variant_category", "position"]] = ["lof", np.nan]
    df_copy.loc[df_copy["predicted_effect"].isin(["inframe_insertion", "inframe_deletion"]), ["variant_category", "position"]] = ["inframe", np.nan]

    # update the variant column using the new variant categories (lof and inframe) and combine the pooled and unpooled variants
    df_copy["variant"] = df_copy["resolved_symbol"] + "_" + df_copy["variant_category"]
    df_pooled = df_copy.query(f"variant_category in ['lof', 'inframe']").sort_values(by=["variant_binary_status", "variant_allele_frequency"], ascending=False, na_position="last").drop_duplicates(subset=["sample_id", "variant"], keep="first")
    del df_copy
    df_genos = pd.concat([df_genos, df_pooled], axis=0)
    del df_pooled
    
    # get annotations for mutations to combine later. Exclude lof and inframe, these will be manually replaced later
    annotated_genos = df_genos.query("variant_category not in ['lof', 'inframe']").drop_duplicates(["variant", "predicted_effect", "position"])[["variant", "predicted_effect", "position"]]

    return df_phenos, df_genos, annotated_genos



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



def compute_statistics_single_model(model_path, df_phenos, df_genos, annotated_genos, who_variants_single_drug, alpha=0.05):

    # final_analysis file with all significant variants for a drug
    old_df = pd.read_csv(os.path.join(model_path, "model_analysis.csv"))
    res_df = old_df.copy()
    
    # check that all variants are there
    if len(set(res_df.loc[~res_df["variant"].str.contains("PC")]["variant"]) - set(df_genos["variant"])) > 0:
        raise ValueError("Variants are missing from df_genos!")
    
    # pivot to matrix and add sample IDs and phenotypes
    combined = df_genos.query("variant in @res_df.variant").pivot(index="sample_id", columns="variant", values="variant_binary_status")
    combined = combined.merge(df_phenos[["sample_id", "phenotype"]], left_index=True, right_on="sample_id").reset_index(drop=True)
    
    # coefficient dictionary to keep track of which variants have positive and negative coefficients
    variant_coef_dict = dict(zip(res_df["variant"], res_df["coef"]))

    # get dataframe of the univariate stats add them to the results dataframe
    full_predict_values = compute_univariate_stats(combined[combined.columns[~combined.columns.str.contains("PC")]], variant_coef_dict)
    res_df = res_df.merge(full_predict_values, on="variant", how="outer").drop_duplicates("variant", keep="first")
    
    # add confidence intervals for all stats except the likelihood ratios
    for var in ["PPV", "NPV", "Sens", "Spec"]:
        
        var_lb, var_ub = compute_wilson_conf_interval(var, alpha, res_df)
        
        # in case there are any numbers outside the range [0, 1] 
        res_df[f"{var}_LB"] = np.max([var_lb, np.zeros(len(res_df))], axis=0)
        res_df[f"{var}_UB"] = np.min([var_ub, np.ones(len(res_df))], axis=0)

    # add confidence intervals for the likelihood ratios
    res_df = compute_likelihood_ratio_confidence_intervals(alpha, res_df)
        
    # get effect annotations and merge them with the results dataframe
    res_df = res_df.merge(annotated_genos, on="variant", how="outer")
    res_df = res_df.loc[~pd.isnull(res_df["coef"])]
    res_df.loc[res_df["variant"].str.contains("lof"), "predicted_effect"] = "lof"
    res_df.loc[res_df["variant"].str.contains("inframe"), "predicted_effect"] = "inframe"
        
    # predicted effect should only be NaN for PCs. position is NaN only for the pooled mutations and PCs
    assert len(res_df.loc[(pd.isnull(res_df["predicted_effect"])) & (~res_df["variant"].str.contains("|".join(["PC"])))]) == 0
    assert len(res_df.loc[(pd.isnull(res_df["position"])) & (~res_df["variant"].str.contains("|".join(["lof", "inframe", "PC"])))]) == 0
    
    # check that every variant is present in at least 1 isolate
    assert res_df.Num_Isolates.min() > 0
    
    # check that there are no NaNs in the univariate statistics. Don't include LR+ upper and lower bounds because they can be NaN if LR+ = inf
    assert len(res_df.loc[~res_df["variant"].str.contains("PC")][pd.isnull(res_df[['Num_Isolates', 'Total_Isolates', 'TP', 'FP', 'TN', 'FN', 'PPV', 'NPV', 'Sens', 'Spec', 'LR+', 'LR-',
                                   'PPV_LB', 'PPV_UB', 'NPV_LB', 'NPV_UB', 'Sens_LB', 'Sens_UB', 'Spec_LB', 'Spec_UB', 'LR-_LB', 'LR-_UB']]).any(axis=1)]) == 0
        
    # check confidence intervals. LB ≤ var ≤ UB, and no confidence intervals have width 0
    for var in ["PPV", "NPV", "Sens", "Spec", "LR+", "LR-"]:
        assert len(res_df.loc[(~res_df[var].isin([0, 1])) & (res_df[var] < res_df[f"{var}_LB"])]) == 0
        assert len(res_df.loc[(~res_df[var].isin([0, 1])) & (res_df[var] > res_df[f"{var}_UB"])]) == 0
        
        width = res_df[f"{var}_UB"] - res_df[f"{var}_LB"]
        assert np.min(width) > 0
        
    # add in the WHO 2021 catalog confidence levels, using the dataframe with 2021 to 2022 mapping
    res_df.rename(columns={"variant": "mutation"}, inplace=True)
    res_df = res_df.merge(who_variants_single_drug, on="mutation", how="left")
    
    # add significance
    secondary_analysis_criteria = ["+2", "ALL", "unpooled", "withSyn"]
    significance_thresh = 0.05
    
    # if any of the secondary analysis criteria are met, change the threshold
    for string in secondary_analysis_criteria:
        if string in model_path:
            significance_thresh = 0.01
    
    res_df.loc[res_df["BH_pval"] < significance_thresh, "Significant"] = 1
    res_df["Significant"] = res_df["Significant"].fillna(0).astype(int)
    
    res_df = res_df.sort_values("Odds_Ratio", ascending=False).drop_duplicates("mutation", keep="first")
    assert len(old_df) == len(res_df)
        
    res_df[['mutation', 'predicted_effect', 'position', 'confidence', 'Odds_Ratio', 'OR_LB', 'OR_UB', 'pval', 'BH_pval',
       'Bonferroni_pval', 'Significant', 'Num_Isolates', 'Total_Isolates', 'TP', 'FP', 'TN', 'FN', 'PPV', 'NPV', 'Sens', 'Spec',
       'LR+', 'LR-', 'PPV_LB', 'PPV_UB', 'NPV_LB', 'NPV_UB', 'Sens_LB',
       'Sens_UB', 'Spec_LB', 'Spec_UB', 'LR+_LB', 'LR+_UB', 'LR-_LB', 'LR-_UB',
       ]].to_csv(os.path.join(model_path, "model_analysis_with_stats.csv"), index=False)
    

_, drug, drug_WHO_abbr = sys.argv

# get dataframe of 2021 WHO confidence gradings
who_variants_single_drug = who_variants_combined.query("drug==@drug_WHO_abbr")
del who_variants_single_drug["drug"]
del who_variants_combined

who_analysis_paths = ["tiers=1/phenos=WHO/dropAF_noSyn",
                  "tiers=1/phenos=WHO/dropAF_noSyn_unpooled",
                  "tiers=1/phenos=WHO/dropAF_withSyn",
                  "tiers=1+2/phenos=WHO/dropAF_noSyn",
                  "tiers=1+2/phenos=WHO/dropAF_noSyn_unpooled",
                  "tiers=1+2/phenos=WHO/dropAF_withSyn",
]

all_analysis_paths = ["tiers=1/phenos=ALL/dropAF_noSyn",
                  "tiers=1/phenos=ALL/dropAF_noSyn_unpooled",
                  "tiers=1/phenos=ALL/dropAF_withSyn",
                  "tiers=1+2/phenos=ALL/dropAF_noSyn",
                  "tiers=1+2/phenos=ALL/dropAF_noSyn_unpooled",
                  "tiers=1+2/phenos=ALL/dropAF_withSyn",
]

continuous_analysis_paths = ["tiers=1/phenos=WHO/encodeAF_noSyn",
                  "tiers=1+2/phenos=WHO/encodeAF_noSyn",
                  "tiers=1/phenos=ALL/encodeAF_noSyn",
                  "tiers=1+2/phenos=ALL/encodeAF_noSyn",
]

# get dataframes of variants for WHO isolates only
df_phenos, df_genos, annotated_genos = get_genos_phenos(analysis_dir, drug)

print("Computing univariate stats for the ALL analyses...")
for path in all_analysis_paths:  
    compute_statistics_single_model(os.path.join(analysis_dir, drug, path), df_phenos, df_genos, annotated_genos, who_variants_single_drug, alpha=0.05)
    
# reduce dataframe size for the WHO only analyses
df_phenos = df_phenos.query("phenotypic_category=='WHO'")
df_genos = df_genos.query("sample_id in @df_phenos.sample_id")

print("Computing univariate stats for the WHO analyses...")
for path in who_analysis_paths:
    compute_statistics_single_model(os.path.join(analysis_dir, drug, path), df_phenos, df_genos, annotated_genos, who_variants_single_drug, alpha=0.05)
    
del df_phenos
del df_genos

# just add WHO 2021 confidence gradings, predicted effect, and Significance to these
for path in continuous_analysis_paths:
    res_df = pd.read_csv(os.path.join(analysis_dir, drug, path, "model_analysis.csv"))

    res_df = res_df.merge(annotated_genos, on="variant", how="outer")
    res_df = res_df.loc[~pd.isnull(res_df["coef"])]
    res_df.loc[res_df["variant"].str.contains("lof"), "predicted_effect"] = "lof"
    res_df.loc[res_df["variant"].str.contains("inframe"), "predicted_effect"] = "inframe"
    
    res_df.rename(columns={"variant": "mutation"}, inplace=True)
    res_df = res_df.merge(who_variants_single_drug, on="mutation", how="left")

    res_df.loc[res_df["BH_pval"] < 0.01, "Significant"] = 1
    res_df["Significant"] = res_df["Significant"].fillna(0).astype(int)
        
    res_df[['mutation', 'predicted_effect', 'position', 'confidence', 'Odds_Ratio', 'OR_LB', 'OR_UB', 'pval', 'BH_pval',
       'Bonferroni_pval', 'Significant']].sort_values("Odds_Ratio", ascending=False).drop_duplicates("mutation", keep="first").to_csv(os.path.join(analysis_dir, drug, path, "model_analysis_with_stats.csv"), index=False)

print(f"Finished {drug}!")

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")