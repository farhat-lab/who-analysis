import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import binomtest
import sys, glob, os, yaml, tracemalloc, warnings
warnings.filterwarnings("ignore")

# starting the memory monitoring
tracemalloc.start()


def get_genos_phenos(analysis_dir, folder, drug):
    '''
    This function gets the phenotype and genotype dataframes for a given drug. These are the same across models of the same type (i.e. all BINARY with WHO phenotypes, or all ATU models). 
    '''
        
    df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, f"phenos_{folder.lower()}.csv"))
    df_genos = pd.read_csv(os.path.join(analysis_dir, drug, "genos.csv.gz"), compression="gzip")
    
    # get only samples that are in the phenotypes file (this is greater than or equal to the number of phenotypes actually used to fit the model because some get dropped)
    df_genos = df_genos.query("sample_id in @df_phenos.sample_id")

    # get pooled mutations
    df_genos["mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
    df_copy = df_genos.copy()

    # pool LOF and inframe mutations
    df_copy.loc[df_copy["predicted_effect"].isin(["frameshift", "start_lost", "stop_gained", "feature_ablation"]), ["variant_category", "position"]] = ["lof", np.nan]
    df_copy.loc[df_copy["predicted_effect"].isin(["inframe_insertion", "inframe_deletion"]), ["variant_category", "position"]] = ["inframe", np.nan]

    # update the mutation column using the new variant categories (lof and inframe) and combine the pooled and unpooled mutations
    df_copy["mutation"] = df_copy["resolved_symbol"] + "_" + df_copy["variant_category"]
    df_pooled = df_copy.query(f"variant_category in ['lof', 'inframe']").sort_values(by=["variant_binary_status", "variant_allele_frequency"], ascending=False, na_position="last").drop_duplicates(subset=["sample_id", "mutation"], keep="first")
    del df_copy
    
    # df_pooled is already sorted by variant_allele_frequency and variant_binary_status
    # drop duplicates by sample and gene to get a pooled LOF all feature for each (sample, gene) pair
    df_pooled_all = df_pooled.drop_duplicates(subset=["sample_id", "resolved_symbol"], keep="first")
    df_pooled_all["variant_category"] = "lof_all"
    df_pooled_all["mutation"] = df_pooled_all["resolved_symbol"] + "_" + df_pooled_all["variant_category"]

    df_genos = pd.concat([df_genos, df_pooled, df_pooled_all], axis=0)
    del df_pooled
    del df_pooled_all
    
    # get annotations for mutations to combine later. Exclude lof and inframe, these will be manually replaced later
    annotated_genos = df_genos.query("variant_category not in ['lof', 'inframe', 'lof_all']").drop_duplicates(["mutation", "predicted_effect", "position"])[["mutation", "predicted_effect", "position"]]

    return df_phenos, df_genos, annotated_genos




def compute_univariate_stats(combined_df, variant_coef_dict):
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
    
    # get counts of isolates grouped by phenotype and mutation -- so how many isolates have a mutation and have a phenotype (all 4 possibilities)
    grouped_df = pd.DataFrame(melted_2.groupby(["phenotype", "variable"]).value_counts()).reset_index()
    grouped_df = grouped_df.rename(columns={"variable": "mutation", "value": "present", 0:"count"})
    
    # add coefficients, create new column for the switched phenotypes (keep the old ones in actual_pheno)
    grouped_df["coef"] = grouped_df["mutation"].map(variant_coef_dict)
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
    final = true_pos_df[["mutation", "TP"]].merge(
            false_pos_df[["mutation", "FP"]], on="mutation", how="outer").merge(
            true_neg_df[["mutation", "TN"]], on="mutation", how="outer").merge(
            false_neg_df[["mutation", "FN"]], on="mutation", how="outer").fillna(0)

    assert len(final) == len(melted["variable"].unique())
    assert len(final) == len(final.drop_duplicates("mutation"))
        
    final["Num_Isolates"] = final["TP"] + final["FP"]
    final["Total_Isolates"] = final[["TP", "FP", "TN", "FN"]].sum(axis=1)
    final["PPV"] = final["TP"] / (final["TP"] + final["FP"])
    final["NPV"] = final["TN"] / (final["TN"] + final["FN"])
    final["Sens"] = final["TP"] / (final["TP"] + final["FN"])
    final["Spec"] = final["TN"] / (final["TN"] + final["FP"])
    final["LR+"] = final["Sens"] / (1 - final["Spec"])
    final["LR-"] = (1 - final["Sens"]) / final["Spec"]
    
    return final[["mutation", "Num_Isolates", "Total_Isolates", "TP", "FP", "TN", "FN", "PPV", "NPV", "Sens", "Spec", "LR+", "LR-"]]
    


def compute_exact_confidence_intervals(res_df, alpha):
    
    res_df = res_df.reset_index(drop=True)
    
    # add exact binomial confidence intervals for the binomial variables. The other two will be done in another function
    for i, row in res_df.iterrows():
        
        # will be null for the principal components, so skip them
        if not pd.isnull(row["TP"]):
            
            # binomtest requires the numbers to be integers
            row[["TP", "FP", "TN", "FN"]] = row[["TP", "FP", "TN", "FN"]].astype(int)
        
            # PPV
            ci = binomtest(k=row["TP"], n=row["TP"] + row["FP"], p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
            res_df.loc[i, ["PPV_LB", "PPV_UB"]] = [ci.low, ci.high]
            
            # NPV
            ci = binomtest(k=row["TN"], n=row["TN"] + row["FN"], p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
            res_df.loc[i, ["NPV_LB", "NPV_UB"]] = [ci.low, ci.high]
            
            # Sensitivity
            ci = binomtest(k=row["TP"], n=row["TP"] + row["FN"], p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
            res_df.loc[i, ["Sens_LB", "Sens_UB"]] = [ci.low, ci.high]
            
            # Specificity
            ci = binomtest(k=row["TN"], n=row["TN"] + row["FP"], p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
            res_df.loc[i, ["Spec_LB", "Spec_UB"]] = [ci.low, ci.high]
    
    return res_df


    
def compute_likelihood_ratio_confidence_intervals(res_df, alpha):
    
    z = np.abs(st.norm.ppf(q=alpha/2))
    
    LRpos_error = np.exp(z * np.sqrt(1/res_df["TP"] - 1/(res_df["TP"] + res_df["FN"]) + 1/res_df["FP"] - 1/(res_df["FP"] + res_df["TN"])))
    LRneg_error = np.exp(z * np.sqrt(1/res_df["FN"] - 1/(res_df["TP"] + res_df["FN"]) + 1/res_df["TN"] - 1/(res_df["FP"] + res_df["TN"])))
    
    res_df["LR+_LB"] = res_df["LR+"] / LRpos_error
    res_df["LR+_UB"] = res_df["LR+"] * LRpos_error
    
    res_df["LR-_LB"] = res_df["LR-"] / LRneg_error
    res_df["LR-_UB"] = res_df["LR-"] * LRneg_error

    return res_df



def compute_statistics_single_model(model_path, df_phenos, df_genos, annotated_genos, model_suffix, alpha=0.05):

    # final_analysis file with all significant variants for a drug
    old_df = pd.read_csv(os.path.join(model_path, f"model_analysis{model_suffix}.csv"))
    res_df = old_df.copy()
    
    # check that all variants are there
    if len(set(res_df.loc[~res_df["mutation"].str.contains("PC")]["mutation"]) - set(df_genos["mutation"])) > 0:
        raise ValueError("Variants are missing from df_genos!")
    
    # pivot to matrix and add sample IDs and phenotypes
    combined = df_genos.query("mutation in @res_df.mutation").pivot(index="sample_id", columns="mutation", values="variant_binary_status")
    combined = combined.merge(df_phenos[["sample_id", "phenotype"]], left_index=True, right_on="sample_id").reset_index(drop=True)
    
    # coefficient dictionary to keep track of which variants have positive and negative coefficients
    variant_coef_dict = dict(zip(res_df["mutation"], res_df["coef"]))

    # get dataframe of the univariate stats add them to the results dataframe
    full_predict_values = compute_univariate_stats(combined[combined.columns[~combined.columns.str.contains("PC")]], variant_coef_dict)
    res_df = res_df.merge(full_predict_values, on="mutation", how="outer").drop_duplicates("mutation", keep="first")
    
    # add confidence intervals for all stats except the likelihood ratios
    res_df = compute_exact_confidence_intervals(res_df, alpha)

    # add confidence intervals for the likelihood ratios
    res_df = compute_likelihood_ratio_confidence_intervals(res_df, alpha)
        
    # get effect annotations and merge them with the results dataframe
    res_df = res_df.merge(annotated_genos, on="mutation", how="outer")
    res_df = res_df.loc[~pd.isnull(res_df["coef"])]
    res_df.loc[(res_df["mutation"].str.contains("lof")) & (~res_df["mutation"].str.contains("all")), "predicted_effect"] = "lof"
    res_df.loc[(res_df["mutation"].str.contains("inframe")) & (~res_df["mutation"].str.contains("all")), "predicted_effect"] = "inframe"
    res_df.loc[res_df["mutation"].str.contains("all"), "predicted_effect"] = "lof_all"
        
    # predicted effect should only be NaN for PCs. position is NaN only for the pooled mutations and PCs
    assert len(res_df.loc[(pd.isnull(res_df["predicted_effect"])) & (~res_df["mutation"].str.contains("|".join(["PC"])))]) == 0
    assert len(res_df.loc[(pd.isnull(res_df["position"])) & (~res_df["mutation"].str.contains("|".join(["lof", "inframe", "PC"])))]) == 0
    
    # check that every mutation is present in at least 1 isolate
    assert res_df.Num_Isolates.min() > 0
    
    # check that there are no NaNs in the univariate statistics. Don't include LR+ upper and lower bounds because they can be NaN if LR+ = inf
    assert len(res_df.loc[~res_df["mutation"].str.contains("PC")][pd.isnull(res_df[['Num_Isolates', 'Total_Isolates', 'TP', 'FP', 'TN', 'FN', 'PPV', 'NPV', 'Sens', 'Spec', 'LR+', 'LR-',
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
    assert len(old_df) == len(res_df)
        
    res_df[['mutation', 'predicted_effect', 'position', 'confidence', 'Odds_Ratio', 'OR_LB', 'OR_UB', 'pval', 'BH_pval',
       'Bonferroni_pval', 'Significant', 'Num_Isolates', 'Total_Isolates', 'TP', 'FP', 'TN', 'FN', 'PPV', 'NPV', 'Sens', 'Spec',
       'LR+', 'LR-', 'PPV_LB', 'PPV_UB', 'NPV_LB', 'NPV_UB', 'Sens_LB',
       'Sens_UB', 'Spec_LB', 'Spec_UB', 'LR+_LB', 'LR+_UB', 'LR-_LB', 'LR-_UB',
       ]].to_csv(os.path.join(model_path, f"model_analysis_with_stats{model_suffix}.csv"), index=False)
    
    print(model_path)
  


def add_significance_predicted_effect(model_path, annotated_genos, model_suffix):
    '''
    Use this function for models without binary inputs and outputs. This function just adds significance and predicted effect and saves the dataframe. 
    '''
    
    res_df = pd.read_csv(os.path.join(model_path, f"model_analysis{model_suffix}.csv"))

    res_df = res_df.merge(annotated_genos, on="mutation", how="outer")
    res_df = res_df.loc[~pd.isnull(res_df["coef"])]
    res_df.loc[(res_df["mutation"].str.contains("lof")) & (~res_df["mutation"].str.contains("all")), "predicted_effect"] = "lof"
    res_df.loc[(res_df["mutation"].str.contains("inframe")) & (~res_df["mutation"].str.contains("all")), "predicted_effect"] = "inframe"
    res_df.loc[res_df["mutation"].str.contains("all"), "predicted_effect"] = "lof_all"

    # use p-value threshold of 0.01 because this function will be used for secondary analyses (non-binary inputs (encodeAF) /outputs (MICs))
    res_df.loc[res_df["BH_pval"] < 0.01, "Significant"] = 1
    res_df["Significant"] = res_df["Significant"].fillna(0).astype(int)

    # no odds ratios, save coefficients
    if "MIC" in model_path:
        res_df[['mutation', 'predicted_effect', 'position', 'confidence', 'coef', 'coef_LB', 'coef_UB', 'pval', 'BH_pval', 'Bonferroni_pval', 'Significant']].sort_values("coef", ascending=False).drop_duplicates("mutation", keep="first").to_csv(os.path.join(model_path, f"model_analysis_with_stats{model_suffix}.csv"), index=False)
    
    # save odds ratios
    else:
        res_df[['mutation', 'predicted_effect', 'position', 'confidence', 'Odds_Ratio', 'OR_LB', 'OR_UB', 'pval', 'BH_pval','Bonferroni_pval', 'Significant']].sort_values("Odds_Ratio", ascending=False).drop_duplicates("mutation", keep="first").to_csv(os.path.join(model_path, f"model_analysis_with_stats{model_suffix}.csv"), index=False)

    

_, drug, folder, analysis_dir = sys.argv

# this is the intermediate folder
folder = folder.upper()
assert folder in ["BINARY", "ATU", "MIC"]

# get dataframes of genotypes and phenotypes. the folder argument ensures that we get only samples with CC and CC-ATU phenotypes, instead of all of them
df_phenos, df_genos, annotated_genos = get_genos_phenos(analysis_dir, folder, drug)

# get all models for which to compute univariate statistics
analysis_paths = []

for tier in os.listdir(os.path.join(analysis_dir, drug, folder)):
    tiers_path = os.path.join(analysis_dir, drug, folder, tier)

    for level_1 in os.listdir(tiers_path):
        level1_path = os.path.join(analysis_dir, drug, folder, tier, level_1)

        if os.path.isfile(os.path.join(level1_path, "model_matrix.pkl")):
            analysis_paths.append(level1_path)

        for level_2 in os.listdir(level1_path):
            level2_path = os.path.join(analysis_dir, drug, folder, tier, level_1, level_2)

            if os.path.isfile(os.path.join(level2_path, "model_matrix.pkl")):
                analysis_paths.append(level2_path)

                
if len(df_phenos) > 0 and len(df_genos) > 0:
    
    if folder == "BINARY":
        print(f"\nComputing univariate statistics for {len(analysis_paths)} BINARY models")
        model_suffix = ""

        # compute the univariate statistics using the full dataframes for the ALL models
        for model_path in analysis_paths:  
            if "dropAF" in model_path:
                if "phenos=ALL" in model_path:
                    compute_statistics_single_model(model_path, df_phenos, df_genos, annotated_genos, model_suffix, alpha=0.05)
            else:
                add_significance_predicted_effect(model_path, annotated_genos, model_suffix)

        # reduce dataframe size for the WHO only analyses
        df_phenos = df_phenos.query("phenotypic_category=='WHO'")
        df_genos = df_genos.query("sample_id in @df_phenos.sample_id")

        # mainly for Pretomanid, which has no WHO phenotypes 
        if len(df_phenos) > 0 and len(df_genos) > 0:
            # compute the univariate statistics using the subsetted dataframes for the WHO analyses
            for model_path in analysis_paths:  
                if "dropAF" in model_path and "phenos=WHO" in model_path:
                    compute_statistics_single_model(model_path, df_phenos, df_genos, annotated_genos, model_suffix, alpha=0.05)


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