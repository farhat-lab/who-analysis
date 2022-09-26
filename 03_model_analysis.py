import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
import seaborn as sns
import scipy.stats as st
import sys

import glob, os, yaml, itertools, sparse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import warnings
warnings.filterwarnings("ignore")

who_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_resistance_variants_all.csv")


def get_pvalues_add_ci(coef_df, bootstrap_df, col, num_samples, alpha=0.05):
    '''
    Compute p-values using the Student's t-distribution. 
    '''
    # degrees of freedom: N - k - 1, where N = number of samples, k = number of features
    dof = num_samples - len(coef_df) - 1
        
    pvals = []
    for i, row in coef_df.iterrows():
        
        # compute the t-statistic. Case when everything is 0 because it's an insignificant feature
        if row["coef"] == 0 and bootstrap_df[row[col]].std() == 0:
            pvals.append(1)
        else:
            # t-statistic is the true coefficient divided by the standard deviation of the bootstrapped coefficients
            t = np.abs(row["coef"]) / bootstrap_df[row[col]].std()
            
            # survival function = 1 - CDF = P(t > t_stat) = measure of extremeness
            pvals.append(st.t.sf(t, df=dof))
        
        # add confidence intervals
        ci = (1-alpha)*100   # default 95
        diff = (100-ci)/2
        lower, upper = np.percentile(bootstrap_df[row[col]].values, q=(diff, 100-diff))
        coef_df.loc[i, "coef_LB"] = lower
        coef_df.loc[i, "coef_UB"] = upper
        
    pvals = np.array(pvals)
    return pvals



aa_code_dict = {'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', \
'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',    \
'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',    \
'GLY':'G', 'PRO':'P', 'CYS':'C'}


def find_SNVs_in_current_WHO(coef_df, aa_code_dict, drug_abbr):
    '''
    This function currently only looks for SNVs in the 2021 version of the WHO mutation catalog. 
    TO-DO: Code up indel comparisons between the measured variants and the variants in the 2021 catalog.
    '''
    new_variants = []

    # need to change the naming convention of variants in order to compare strings with the current WHO mutation catalog
    for variant in coef_df.variant.values:

        new_variant = variant.split(".")[-1]
        
        # nucleotide change conversions
        if ">" in new_variant:
            og_nuc = new_variant.split(">")[0][-1]
            mut_nuc = new_variant.split(">")[1]

            og_nuc_idx = list(new_variant).index(">")-1
            pos = int(new_variant[:og_nuc_idx])

            new_variant = f"{og_nuc.lower()}{pos}{mut_nuc.lower()}"
            
        else:
            # AA conversions
            for key, value in aa_code_dict.items():

                if key in new_variant.upper():
                    new_variant = new_variant.upper().replace(key, value)

        # indels???
        new_variants.append(variant.split("_")[0] + "_" + new_variant)
        
    coef_df["gene"] = [variant.split("_")[0] for variant in coef_df.variant.values]
    coef_df.rename(columns={"variant": "orig_variant"}, inplace=True)
    coef_df["variant"] = new_variants
    
    # WHO variants are named with the gene first, followed by an underscore, then the variant itself
    who_cat12 = who_variants.query("drug == @drug_abbr")
    coef_df = coef_df.merge(who_cat12[["genome_index", "confidence", "variant"]], on="variant", how="outer")
    coef_df.rename(columns={"confidence": "confidence_WHO_2021"}, inplace=True)
    
    # drop duplicates that arise from the same AA variant being associated with multiple nucleotide mutations in the 2021 catalog
    return coef_df.dropna(subset="orig_variant").drop_duplicates(subset="orig_variant", keep='first').reset_index(drop=True)



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
    assert len(np.unique(final[["TP", "FP", "TN", "FN"]].sum(axis=1))) == 1
    
    final["Num_Isolates"] = final["TP"] + final["FP"]
    final["Total_Isolates"] = final["TP"] + final["FP"] + final["TN"] + final["FN"]
    final["PPV"] = final["TP"] / (final["TP"] + final["FP"])
    final["Sens"] = final["TP"] / (final["TP"] + final["FN"])
    final["Spec"] = final["TN"] / (final["TN"] + final["FP"])
    final["LR+"] = final["Sens"] / (1 - final["Spec"])
    final["LR-"] = (1 - final["Sens"]) / final["Spec"]
    #final["NPV"] = final["TN"] / (final["TN"] + final["FN"])
    
    # check that all feature rows have the same numbers of samples
    assert len(np.unique(final[["TP", "FP", "TN", "FN"]].sum(axis=1))) == 1
    
    if len(return_stats) == 0:
        return final[["orig_variant", "Num_Isolates", "TP", "FP", "TN", "FN", "PPV", "Sens", "Spec", "LR+", "LR-"]]
    else:
        return final[return_stats]



def BH_FDR_correction(coef_df):
    '''
    Implement Benjamini-Hochberg FDR correction.
    '''
    # sort the individual p-values in ascending order
    coef_df = coef_df.sort_values("pval", ascending=True)
    
    # assign ranks -- ties get the same value, and only increment by one
    rank_dict = dict(zip(np.unique(coef_df["pval"]), np.arange(len(np.unique(coef_df["pval"])))+1))
    ranks = coef_df["pval"].map(rank_dict).values
    
    coef_df["BH_pval"] = np.min([coef_df["pval"] * len(coef_df) / ranks, np.ones(len(coef_df))], axis=0)  
    return coef_df



def run_all(drug, drug_abbr, model_prefix, out_dir, het_mode, alpha=0.05, num_bootstrap=1000):
    
    # coefficients from L2 regularized regression ("baseline" regression)
    coef_df = pd.read_csv(os.path.join(out_dir, model_prefix, "regression_coef.csv"))

    # coefficients from bootstrap replicates
    bs_df = pd.read_csv(os.path.join(out_dir, model_prefix, "coef_bootstrap.csv"))
    
    # read in all genotypes and phenotypes
    model_inputs = pd.read_pickle(os.path.join(out_dir, model_prefix, "model_matrix.pkl"))
    df_phenos = pd.read_csv(os.path.join(out_dir, model_prefix, "phenos.csv"))

    # add p-values and confidence intervals to the results dataframe
    pvals = get_pvalues_add_ci(coef_df, bs_df, "variant", len(model_inputs), alpha=alpha)
    coef_df["pval"] = pvals
    
    # Benjamini-Hochberg correction
    coef_df = BH_FDR_correction(coef_df)
    
    # Bonferroni correction
    coef_df["Bonferroni_pval"] = np.min([coef_df["pval"] * len(coef_df), np.ones(len(coef_df))], axis=0)
    
    # adjusted p-values are larger so that fewer null hypotheses (coef = 0) are rejected
    assert len(coef_df.query("pval > BH_pval")) == 0
    assert len(coef_df.query("pval > Bonferroni_pval")) == 0
        
    # return all features with non-zero coefficients. Don't exclude by p-value yet because some variants with large effects are rare
    res_df = coef_df.query("coef != 0").sort_values("coef", ascending=False).reset_index(drop=True)
    res_df = find_SNVs_in_current_WHO(res_df, aa_code_dict, drug_abbr)
    
    # convert to odds ratios
    res_df["Odds_Ratio"] = np.exp(res_df["coef"])
    res_df["OR_LB"] = np.exp(res_df["coef_LB"])
    res_df["OR_UB"] = np.exp(res_df["coef_UB"])
    
    assert sum(res_df["OR_LB"] > res_df["Odds_Ratio"]) == 0
    assert sum(res_df["OR_UB"] < res_df["Odds_Ratio"]) == 0
    
    combined = model_inputs.merge(df_phenos[["sample_id", "phenotype"]], on="sample_id").reset_index(drop=True)
    
    # for tractability, this is done after filtering out features with a logistic coefficient of 0. Also creates a lot of NaNs in those cases.
    combined_small = combined[["sample_id", "phenotype"] + list(res_df.loc[~res_df["orig_variant"].str.contains("PC")]["orig_variant"].values)]
    assert len(combined_small) == len(combined)
    
    # can only compute predictive values for binary genotypes. Otherwise there are no true/false positives/negatives
    if het_mode != "AF":
    
        # get dataframe of predictive values for the non-zero coefficients and add them to the results dataframe
        full_predict_values = compute_predictive_values(combined_small)
        res_df = res_df.merge(full_predict_values, on="orig_variant", how="outer")

        print(f"Computing and bootstrapping predictive values with {num_bootstrap} replicates")
        bs_results = pd.DataFrame(columns = res_df.loc[~res_df["orig_variant"].str.contains("PC")]["orig_variant"].values).astype(float)
        
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

        # add the confidence intervals to the dataframe
        for variable in ["PPV", "Sens", "Spec", "LR+", "LR-"]:

            lower, upper = np.nanpercentile(bs_results.loc[variable], q=[2.5, 97.5], axis=0)
            
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

    
    # clean up the dataframe a little -- variant and gene are from the 2021 catalog (redundant with the orig_variant column)
    del res_df["variant"]
    del res_df["gene"]
    #del res_df["genome_index"]
    
    return res_df.drop_duplicates("orig_variant", keep='first').sort_values("coef", ascending=False).reset_index(drop=True)


_, config_file, drug, drug_WHO_abbr = sys.argv

kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
pheno_category_lst = kwargs["pheno_category_lst"]
model_prefix = kwargs["model_prefix"]
het_mode = kwargs["het_mode"]

num_PCs = kwargs["num_PCs"]
num_bootstrap = kwargs["num_bootstrap"]
alpha = kwargs["alpha"]

out_dir = '/n/data1/hms/dbmi/farhat/ye12/who/analysis'
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
else:
    phenos_name = "WHO"

out_dir = os.path.join(out_dir, drug, f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}")

# run analysis
model_analysis = run_all(drug, drug_WHO_abbr, model_prefix, out_dir, het_mode, alpha=alpha, num_bootstrap=num_bootstrap)

# save
if num_PCs > 0:
    print(f"{len(model_analysis.loc[(model_analysis['coef'] > 0) & model_analysis['orig_variant'].str.contains('PC')])} principal components have nominally significant non-zero coefficients")
model_analysis.to_pickle(os.path.join(out_dir, model_prefix, "model_analysis.pkl"))