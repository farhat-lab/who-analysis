import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
import scipy.stats as st
import sys

import glob, os, yaml
import warnings
warnings.filterwarnings("ignore")
who_variants_combined = pd.read_csv("analysis/who_confidence_2021.csv")


############# CODE TO MAKE THE COMBINE WHO 2021 VARIANTS + CONFIDENCES FILE #############
# who_variants = pd.read_csv("analysis/who_resistance_variants_all.csv")
# variant_mapping = pd.read_csv("data/v1_to_v2_variants_mapping.csv", usecols=["gene_name", "variant", "raw_variant_mapping_data.variant_category"])
# variant_mapping.columns = ["gene", "V1", "V2"]
# variant_mapping["mutation"] = variant_mapping["gene"] + "_" + variant_mapping["V2"]

# # combine with the new names to get a dataframe with the confidence leve,s and variant mappings between 2021 and 2022
# who_variants_combined = who_variants.merge(variant_mapping[["V1", "mutation"]], left_on="variant", right_on="V1", how="inner")
# del who_variants_combined["variant"]

# # check that they have all the same variants
# assert len(set(who_variants_combined["V1"]).symmetric_difference(set(who_variants["variant"]))) == 0

# del who_variants_combined["genome_index"]
# del who_variants_combined["gene"]
# del who_variants_combined["V1"]

# # some V1 mutations were combined into a single V2 mutation, so they may have multiple confidences listed. Keep the highest confidence instance
# who_variants_combined = who_variants_combined.dropna().sort_values("confidence", ascending=True).drop_duplicates(subset=["drug", "mutation"], keep="first")
# who_variants_combined.to_csv("analysis/who_confidence_2021.csv", index=False)


def get_pvalues_add_ci(coef_df, bootstrap_df, col, num_samples, alpha=0.05):
    '''
    Compute p-values using the Student's t-distribution. 
    '''
        
    # first compute confidence intervals for the coefficients for all mutation, regardless of tier
    ci = (1-alpha)*100
    diff = (100-ci)/2
        
    # check ordering, then compute upper and lower bounds for the coefficients
    assert sum(coef_df["mutation"].values != bootstrap_df.columns) == 0
    lower, upper = np.percentile(bootstrap_df, axis=0, q=(diff, 100-diff))
    coef_df["coef_LB"] = lower
    coef_df["coef_UB"] = upper
            
    # degrees of freedom: N - k - 1, where N = number of samples, k = number of features
    dof = num_samples - len(coef_df) - 1
    
    for i, row in coef_df.iterrows():
        
        # sanity check
        assert bootstrap_df[row[col]].std() > 0  

        # t-statistic is the true coefficient divided by the standard deviation of the bootstrapped coefficients
        t = np.abs(row["coef"]) / bootstrap_df[row[col]].std()

        # survival function = 1 - CDF = P(t > t_stat) = measure of extremeness        
        coef_df.loc[i, "pval"] = st.t.sf(t, df=dof)
        
    assert len(coef_df.loc[pd.isnull(coef_df["pval"])]) == 0
    return coef_df
        


def BH_FDR_correction(coef_df):
    '''
    Implement Benjamini-Hochberg FDR correction.
    '''
    
    coef_df = coef_df.sort_values("pval", ascending=True)
    
    # assign ranks -- ties get the same value, and only increment by one
    rank_dict = dict(zip(np.unique(coef_df["pval"]), np.arange(len(np.unique(coef_df["pval"])))+1))
    ranks = coef_df["pval"].map(rank_dict).values
    
    coef_df["BH_pval"] = np.min([coef_df["pval"] * len(coef_df) / ranks, np.ones(len(coef_df))], axis=0) 

    return coef_df


def run_all(drug, drug_abbr, who_variants_combined, **kwargs):
    
    tiers_lst = kwargs["tiers_lst"]
    binary = kwargs["binary"]
    synonymous = kwargs["synonymous"]
    alpha = kwargs["alpha"]
    model_prefix = kwargs["model_prefix"]
    pheno_category_lst = kwargs["pheno_category_lst"]
    atu_analysis = kwargs["atu_analysis"]
    atu_analysis_type = kwargs["atu_analysis_type"]
    analysis_dir = kwargs["output_dir"]
    
    if "ALL" in pheno_category_lst:
        phenos_name = "ALL"
    else:
        phenos_name = "WHO"

    if binary:
        if atu_analysis:
            out_dir = os.path.join(analysis_dir, drug, "ATU", f"tiers={'+'.join(tiers_lst)}", model_prefix)
        else:
            out_dir = os.path.join(analysis_dir, drug, "BINARY", f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}", model_prefix)
    else:
        out_dir = os.path.join(analysis_dir, drug, "MIC", f"tiers={'+'.join(tiers_lst)}", model_prefix)

    # get dataframe of 2021 WHO confidence gradings
    who_variants_single_drug = who_variants_combined.query("drug==@drug_abbr")
    del who_variants_single_drug["drug"]
    del who_variants_combined
    
    if atu_analysis:
        model_suffix = f"_{atu_analysis_type.replace('-', '_')}"
    else:
        model_suffix = ""
    
    # no model (basically just for Pretomanid)
    if not os.path.isfile(os.path.join(out_dir, f"regression_coef{model_suffix}.csv")):
        exit()
        
    # coefficients from L2 regularized regression ("baseline" regression)
    coef_df = pd.read_csv(os.path.join(out_dir, f"regression_coef{model_suffix}.csv"))

    # coefficients from bootstrap replicates
    bs_df = pd.read_csv(os.path.join(out_dir, f"coef_bootstrap{model_suffix}.csv"))
    
    # to get the number of samples, read in the model eigenvectors file
    model_eigenvecs = pd.read_pickle(os.path.join(out_dir, "model_eigenvecs.pkl"))

    # add confidence intervals and p-values (both based on the bootstrapped models) to the results dataframe    
    coef_df = get_pvalues_add_ci(coef_df, bs_df, "mutation", len(model_eigenvecs), alpha=alpha)
    del model_eigenvecs
    
    # Benjamini-Hochberg correction
    coef_df = BH_FDR_correction(coef_df)
    
    # Bonferroni correction
    coef_df["Bonferroni_pval"] = np.min([coef_df["pval"] * len(coef_df["pval"]), np.ones(len(coef_df["pval"]))], axis=0)

    # adjusted p-values are larger so that fewer null hypotheses (coef = 0) are rejected
    assert len(coef_df.query("pval > BH_pval")) == 0
    assert len(coef_df.query("pval > Bonferroni_pval")) == 0
    
    # convert to odds ratios
    if binary:
        coef_df["Odds_Ratio"] = np.exp(coef_df["coef"])
        coef_df["OR_LB"] = np.exp(coef_df["coef_LB"])
        coef_df["OR_UB"] = np.exp(coef_df["coef_UB"])
        
    # add in the WHO 2021 catalog confidence levels, using the dataframe with 2021 to 2022 mapping
    final_df = coef_df.merge(who_variants_single_drug, on="mutation", how="left")
    assert len(final_df) == len(coef_df)
    
    # save
    final_df.sort_values("coef", ascending=False).to_csv(os.path.join(out_dir, f"model_analysis{model_suffix}.csv"), index=False)
    

_, config_file, drug, drug_WHO_abbr = sys.argv

kwargs = yaml.safe_load(open(config_file))

# run analysis
run_all(drug, drug_WHO_abbr, who_variants_combined, **kwargs)