import numpy as np
import pandas as pd
import scipy.stats as st
import sys, glob, os, yaml
who_variants_combined = pd.read_csv("analysis/who_confidence_2021.csv")

# analysis utils is in the analysis folder
sys.path.append(os.path.join(os.getcwd(), "analysis"))
from stats_utils import *

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

_, config_file, drug, drug_WHO_abbr = sys.argv

kwargs = yaml.safe_load(open(config_file))    
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
who_variants_single_drug = who_variants_combined.query("drug==@drug_WHO_abbr")
del who_variants_single_drug["drug"]
del who_variants_combined

if atu_analysis:
    model_suffix = f"_{atu_analysis_type.replace('-', '_')}"
else:
    model_suffix = ""

# no model (basically just for Pretomanid)
if not os.path.isfile(os.path.join(out_dir, f"regression_coef{model_suffix}.csv")):
    exit()

# coefficients from L2 regularized regression ("baseline" regression) and the coefficients from the permutation test
coef_df = pd.read_csv(os.path.join(out_dir, f"regression_coef{model_suffix}.csv")).reset_index(drop=True)
permute_df = pd.read_csv(os.path.join(out_dir, "coef_permutation.csv"))

# assess significance using the results of the permutation test
for i, row in coef_df.iterrows():
    # p-value is the proportion of permutation coefficients that are AT LEAST AS EXTREME as the test statistic
    if row["coef"] > 0:
        coef_df.loc[i, "pval"] = np.mean(permute_df[row["mutation"]] >= row["coef"])
    else:
        coef_df.loc[i, "pval"] = np.mean(permute_df[row["mutation"]] <= row["coef"])

# # coefficients from bootstrap replicates
# bs_df = pd.read_csv(os.path.join(out_dir, f"coef_bootstrap{model_suffix}.csv"))

# # to get the number of samples, read in the model matrix file
# model_matrix = pd.read_pickle(os.path.join(out_dir, "model_matrix.pkl"))

# # add confidence intervals and p-values (both based on the bootstrapped models) to the results dataframe    
# coef_df = get_pvalues_add_ci(coef_df, bs_df, "mutation", len(model_matrix), alpha=alpha)
# del model_matrix

# Benjamini-Hochberg and Bonferroni corrections
coef_df = add_pval_corrections(coef_df)

# adjusted p-values are larger so that fewer null hypotheses (coef = 0) are rejected
if len(coef_df.query("pval > BH_pval")) > 0:
    print(coef_df.query("pval > BH_pval"))
if len(coef_df.query("pval > Bonferroni_pval")) > 0:
    print(coef_df.query("pval > Bonferroni_pval"))

# convert to odds ratios
if binary:
    coef_df["Odds_Ratio"] = np.exp(coef_df["coef"])
    # coef_df["OR_LB"] = np.exp(coef_df["coef_LB"])
    # coef_df["OR_UB"] = np.exp(coef_df["coef_UB"])

# add in the WHO 2021 catalog confidence levels, using the dataframe with 2021 to 2022 mapping
final_df = coef_df.merge(who_variants_single_drug, on="mutation", how="left")
assert len(final_df) == len(coef_df)

# save
final_df.sort_values("coef", ascending=False).to_csv(os.path.join(out_dir, f"model_analysis{model_suffix}.csv"), index=False)