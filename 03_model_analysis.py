import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
import scipy.stats as st
import sys

import glob, os, yaml
import warnings
warnings.filterwarnings("ignore")
analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'
who_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_resistance_variants_all.csv")


def get_pvalues_add_ci(coef_df, bootstrap_df, col, num_samples, tier1_variants=None, alpha=0.05):
    '''
    Compute p-values using the Student's t-distribution. 
    '''
    
    # first compute confidence intervals for the coefficients for all variants, regardless of tier
    ci = (1-alpha)*100
    diff = (100-ci)/2
        
    # check ordering, then compute upper and lower bounds for the coefficients
    assert sum(coef_df["variant"].values != bootstrap_df.columns) == 0
    lower, upper = np.percentile(bootstrap_df, axis=0, q=(diff, 100-diff))
    coef_df["coef_LB"] = lower
    coef_df["coef_UB"] = upper
            
    # degrees of freedom: N - k - 1, where N = number of samples, k = number of features
    dof = num_samples - len(coef_df) - 1
    
    for i, row in coef_df.iterrows():
        # compute the t-statistic. 
        # this if statement is for the case when everything is 0 because it's an insignificant feature
        if row["coef"] == 0 and bootstrap_df[row[col]].std() == 0:
            continue
        else:
            # t-statistic is the true coefficient divided by the standard deviation of the bootstrapped coefficients
            t = np.abs(row["coef"]) / bootstrap_df[row[col]].std()

            # survival function = 1 - CDF = P(t > t_stat) = measure of extremeness        
            coef_df.loc[i, "pval"] = st.t.sf(t, df=dof)
        
    return coef_df
        


# aa_code_dict = {'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', \
# 'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',    \
# 'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',    \
# 'GLY':'G', 'PRO':'P', 'CYS':'C'}


# def find_SNVs_in_current_WHO(coef_df, aa_code_dict, drug_abbr):
#     '''
#     This function currently only looks for SNVs in the 2021 version of the WHO mutation catalog. 
#     TO-DO: Code up indel comparisons between the measured variants and the variants in the 2021 catalog.
#     '''
#     new_variants = []

#     # need to change the naming convention of variants in order to compare strings with the current WHO mutation catalog
#     for variant in coef_df.variant.values:

#         new_variant = variant.split(".")[-1]
        
#         # nucleotide change conversions
#         if ">" in new_variant:
#             og_nuc = new_variant.split(">")[0][-1]
#             mut_nuc = new_variant.split(">")[1]

#             og_nuc_idx = list(new_variant).index(">")-1
#             pos = int(new_variant[:og_nuc_idx])

#             new_variant = f"{og_nuc.lower()}{pos}{mut_nuc.lower()}"
            
#         else:
#             # AA conversions
#             for key, value in aa_code_dict.items():

#                 if key in new_variant.upper():
#                     new_variant = new_variant.upper().replace(key, value)

#         new_variants.append(variant.split("_")[0] + "_" + new_variant)
        
#     coef_df["gene"] = [variant.split("_")[0] for variant in coef_df.variant.values]
#     coef_df.rename(columns={"variant": "orig_variant"}, inplace=True)
#     coef_df["variant"] = new_variants
    
#     # WHO variants are named with the gene first, followed by an underscore, then the variant itself
#     who_single_drug = who_variants.query("drug == @drug_abbr")
#     coef_df = coef_df.merge(who_single_drug[["genome_index", "confidence", "variant"]], on="variant", how="outer")
#     coef_df.rename(columns={"confidence": "confidence_WHO_2021"}, inplace=True)
    
#     # drop duplicates that arise from the same AA variant being associated with multiple nucleotide mutations in the 2021 catalog
#     return coef_df.dropna(subset=["orig_variant"]).drop_duplicates(subset=["orig_variant"], keep='first').reset_index(drop=True)




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



def run_all(out_dir, drug, drug_abbr, **kwargs):
    
    tiers_lst = kwargs["tiers_lst"]
    pheno_category_lst = kwargs["pheno_category_lst"]
    model_prefix = kwargs["model_prefix"]
    synonymous = kwargs["synonymous"]

    alpha = kwargs["alpha"]
    binary = kwargs["binary"]
    
    # coefficients from L2 regularized regression ("baseline" regression)
    coef_df = pd.read_csv(os.path.join(out_dir, "regression_coef.csv"))
    coef_df = coef_df.query("coef != 0")

    # coefficients from bootstrap replicates
    bs_df = pd.read_csv(os.path.join(out_dir, "coef_bootstrap.csv"))
    bs_df = bs_df[coef_df["variant"]]
    
    # to get the number of samples, read in the model eigenvectors file, which is the smallest pickle file
    model_eigenvecs = pd.read_pickle(os.path.join(out_dir, "model_eigenvecs.pkl"))

    # add confidence intervals and p-values (both based on the bootstrapped models) to the results dataframe    
    coef_df = get_pvalues_add_ci(coef_df, bs_df, "variant", len(model_eigenvecs), alpha=alpha)
    del model_eigenvecs
        
    # Benjamini-Hochberg correction
    coef_df = BH_FDR_correction(coef_df)

    # Bonferroni correction
    coef_df["Bonferroni_pval"] = np.min([coef_df["pval"] * len(coef_df), np.ones(len(coef_df))], axis=0)

    # adjusted p-values are larger so that fewer null hypotheses (coef = 0) are rejected
    assert len(coef_df.query("pval > BH_pval")) == 0
    assert len(coef_df.query("pval > Bonferroni_pval")) == 0

#     # return all features with non-zero coefficients.
#     coef_df = find_SNVs_in_current_WHO(coef_df, aa_code_dict, drug_abbr)

    # convert to odds ratios
    if binary:
        coef_df["Odds_Ratio"] = np.exp(coef_df["coef"])
        coef_df["OR_LB"] = np.exp(coef_df["coef_LB"])
        coef_df["OR_UB"] = np.exp(coef_df["coef_UB"])
 
    # clean up the dataframe a little -- variant and gene are from the 2021 catalog (redundant with the orig_variant column)
    del coef_df["variant"]
    del coef_df["gene"]
    
    return coef_df.drop_duplicates("orig_variant", keep='first').sort_values("coef", ascending=False).reset_index(drop=True)


_, config_file, drug, drug_WHO_abbr = sys.argv

kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
pheno_category_lst = kwargs["pheno_category_lst"]
model_prefix = kwargs["model_prefix"]
synonymous = kwargs["synonymous"]

if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
else:
    phenos_name = "WHO"

out_dir = os.path.join(analysis_dir, drug, f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}", model_prefix)

if not os.path.isdir(out_dir):
    print("No model for this analysis")
    exit()

# run analysis
model_analysis = run_all(out_dir, drug, drug_WHO_abbr, **kwargs)

# save
model_analysis.to_csv(os.path.join(out_dir, "model_analysis.csv"), index=False)