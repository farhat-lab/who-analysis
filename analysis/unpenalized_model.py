import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
import scipy.stats as st
import sklearn.metrics
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
import tracemalloc, pickle
from stats_utils import *
who_variants_combined = pd.read_csv("who_confidence_2021.csv")


# starting the memory monitoring
tracemalloc.start()

_, config_file, drug = sys.argv

kwargs = yaml.safe_load(open(config_file))

analysis_dir = kwargs["output_dir"]
num_PCs = kwargs["num_PCs"]
pheno_category_lst = kwargs["pheno_category_lst"]
tiers_lst = ["1", "2"]

# make sure that both phenotypes are included
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
    pheno_category_lst = ["ALL", "WHO"]
else:
    phenos_name = "WHO"
    
# fit separate models for both the set of mutations that were significant in both Ridge AND LRT, or mutations that were significant in either
# AND_mutations is a subset of OR_mutations
AND_mutations = pd.read_csv(os.path.join(analysis_dir, drug, "BINARY/prediction_models/ridge_AND_LRT_variants.csv"))
print(AND_mutations.shape)

out_dir = os.path.join(analysis_dir, drug, "BINARY/unpenalized")
print(f"Saving results to {out_dir}")

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

    
############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############
    

# read in only the genotypes files for the tiers for this model
df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv")).query("phenotypic_category in @pheno_category_lst")
df_model = pd.concat([pd.read_csv(os.path.join(analysis_dir, drug, f"genos_{num}.csv.gz"), compression="gzip", low_memory=False) for num in tiers_lst])

# then keep only samples of the desired phenotype
df_model = df_model.loc[df_model["sample_id"].isin(df_phenos["sample_id"])]

# keep only the mutations of interest
df_model["mutation"] = df_model["resolved_symbol"] + "_" + df_model["variant_category"]

df_model = df_model.query("mutation in @AND_mutations")

drop_isolates = df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] <= 0.75)].sample_id.unique()
print(f"Dropped {len(drop_isolates)} isolates with any intermediate AFs. Remainder are binary")
df_model = df_model.query("sample_id not in @drop_isolates")    

df_model = df_model.dropna(subset="variant_binary_status", axis=0, how="any")
df_model = df_model.sort_values("variant_binary_status", ascending=False).drop_duplicates(["sample_id", "mutation"], keep="first")
matrix = df_model.pivot(index="sample_id", columns="mutation", values="variant_binary_status")
del df_model

matrix = matrix.dropna(axis=0, how="any")
matrix = matrix[matrix.columns[~((matrix == 0).all())]]

matrix.to_pickle(os.path.join(analysis_dir, drug, f"tiers={'+'.join(tiers_lst)}_phenos={phenos_name}_significant.pkl"))
print(f"Fitting prediction model on {matrix.shape[0]} samples and {matrix.shape[1]} features")   

if len(set(AND_mutations) - set(matrix.columns)) > 0:
    raise ValueError(f"Mutations {set(AND_mutations) - set(matrix.columns)} are not in the model matrix")

# Read in the PC coordinates dataframe, then keep only the desired number of principal components
eigenvec_df = pd.read_csv("../data/eigenvec_10PC.csv", index_col=[0]).iloc[:, :num_PCs]
matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True, how="inner")

df_phenos = df_phenos.set_index("sample_id").loc[matrix.index]
assert sum(matrix.index != df_phenos.index.values) == 0
    

########################## STEP 2: FIT MODEL ##########################


scaler = StandardScaler()

X = scaler.fit_transform(matrix.values)
y = df_phenos["phenotype"].values

model = LogisticRegression(penalty=None,
                           max_iter=10000, 
                           multi_class='ovr',
                           scoring='neg_log_loss',
                           class_weight='balanced'
                          )

model.fit(X, y)
reg_param = 0
print(f"Regularization parameter: {reg_param}")

# save coefficients
coef_df = pd.DataFrame({"mutation": matrix.columns, "coef": np.squeeze(model.coef_)})
coef_df.to_csv(os.path.join(out_dir, "regression_coef.csv"), index=False)
   
    
########################## STEP 4: PERFORM PERMUTATION TEST TO GET SIGNIFICANCE AND BOOTSTRAPPING TO GET ODDS RATIO CONFIDENCE INTERVALS ##########################


print(f"Peforming permutation test with {num_bootstrap} replicates")
permute_df = perform_permutation_test(reg_param, X, y, num_bootstrap, binary=binary)
permute_df.columns = model_inputs.columns
permute_df.to_csv(os.path.join(out_dir, "coef_permutation.csv"), index=False)

print(f"Peforming bootstrapping with {num_bootstrap} replicates")
bootstrap_df = perform_bootstrapping(reg_param, X, y, num_bootstrap, binary=binary, save_summary_stats=False)
bootstrap_df.columns = model_inputs.columns
bootstrap_df.to_csv(os.path.join(out_dir, "coef_bootstrapping.csv"), index=False)

    
########################## STEP 4: ADD PERMUTATION TEST P-VALUES TO THE MAIN COEF DATAFRAME ##########################
    

# get dataframe of 2021 WHO confidence gradings
who_variants_single_drug = who_variants_combined.query("drug==@drug_WHO_abbr")
del who_variants_single_drug["drug"]
del who_variants_combined

# add confidence intervals for the coefficients for all mutation. first check ordering of mutations
ci = (1-alpha)*100
diff = (100-ci)/2
assert sum(coef_df["mutation"].values != bootstrap_df.columns) == 0
lower, upper = np.percentile(bootstrap_df, axis=0, q=(diff, 100-diff))
coef_df["coef_LB"] = lower
coef_df["coef_UB"] = upper
    
# assess significance using the results of the permutation test
for i, row in coef_df.iterrows():
    # p-value is the proportion of permutation coefficients that are AT LEAST AS EXTREME as the test statistic
    if row["coef"] > 0:
        coef_df.loc[i, "pval"] = np.mean(permute_df[row["mutation"]] >= row["coef"])
    else:
        coef_df.loc[i, "pval"] = np.mean(permute_df[row["mutation"]] <= row["coef"])
        
# Benjamini-Hochberg and Bonferroni corrections
coef_df = add_pval_corrections(coef_df)

# adjusted p-values are larger so that fewer null hypotheses (coef = 0) are rejected
assert len(coef_df.query("pval > BH_pval")) == 0
assert len(coef_df.query("pval > Bonferroni_pval")) == 0

# convert to odds ratios
if binary:
    coef_df["Odds_Ratio"] = np.exp(coef_df["coef"])
    coef_df["OR_LB"] = np.exp(coef_df["coef_LB"])
    coef_df["OR_UB"] = np.exp(coef_df["coef_UB"])

# add in the WHO 2021 catalog confidence levels, using the dataframe with 2021 to 2022 mapping and save
final_df = coef_df.merge(who_variants_single_drug, on="mutation", how="left")
assert len(final_df) == len(coef_df)
final_df.sort_values("coef", ascending=False).to_csv(os.path.join(out_dir, "model_analysis.csv"), index=False)        

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")