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

_, config_file, drug, drug_WHO_abbr = sys.argv

kwargs = yaml.safe_load(open(config_file))

analysis_dir = kwargs["output_dir"]
num_PCs = kwargs["num_PCs"]
pheno_category_lst = kwargs["pheno_category_lst"]
tiers_lst = ["1", "2"]
binary = kwargs["binary"]
alpha = kwargs["alpha"]
num_bootstrap = kwargs["num_bootstrap"]

# make sure that both phenotypes are included
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
    pheno_category_lst = ["ALL", "WHO"]
else:
    phenos_name = "WHO"
    
# fit separate models for both the set of mutations that were significant in both Ridge AND LRT, or mutations that were significant in either
# AND_mutations = pd.read_csv(os.path.join(analysis_dir, drug, "BINARY/prediction_models/ridge_AND_LRT_variants.csv")).mutation.values
# print(AND_mutations.shape)

out_dir = os.path.join(analysis_dir, drug, "BINARY/unpenalized")
print(f"Saving results to {out_dir}")

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

    
############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############
    

# read in only the genotypes files for the tiers for this model
df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv")).query("phenotypic_category in @pheno_category_lst")
matrix = pd.read_pickle(os.path.join(analysis_dir, drug, f"BINARY/tiers={'+'.join(tiers_lst)}/phenos={phenos_name}/dropAF_noSyn_unpooled/model_matrix.pkl"))

# if len(set(AND_mutations) - set(matrix.columns)) > 0:
#     raise ValueError(f"Mutations {set(AND_mutations) - set(matrix.columns)} are not in the model matrix")

# matrix = matrix[AND_mutations]
# print(f"Fitting prediction model on {matrix.shape[0]} samples and {matrix.shape[1]} features")   
    
# Read in the PC coordinates dataframe, then keep only the desired number of principal components
eigenvec_df = pd.read_csv("../data/eigenvec_10PC.csv", index_col=[0]).iloc[:, :num_PCs]
matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True, how="inner")

df_phenos = df_phenos.set_index("sample_id").loc[matrix.index]
assert sum(matrix.index != df_phenos.index.values) == 0
    

########################## STEP 2: FIT MODEL ##########################


scaler = StandardScaler()

X = scaler.fit_transform(matrix.values)
y = df_phenos["phenotype"].values
print(f"{X.shape[0]} samples and {X.shape[1]} variables in the model")

model = LogisticRegression(penalty='none',
                           max_iter=10000, 
                           multi_class='ovr',
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
permute_df.columns = matrix.columns
permute_df.to_csv(os.path.join(out_dir, "coef_permutation.csv"), index=False)

print(f"Peforming bootstrapping with {num_bootstrap} replicates")
bootstrap_df = perform_bootstrapping(reg_param, X, y, num_bootstrap, binary=binary, save_summary_stats=False)
bootstrap_df.columns = matrix.columns
bootstrap_df.to_csv(os.path.join(out_dir, "coef_bootstrapping.csv"), index=False)

    
########################## STEP 4: ADD PERMUTATION TEST P-VALUES TO THE MAIN COEF DATAFRAME ##########################
    

final_df = get_coef_and_confidence_intervals(alpha, binary, who_variants_combined, drug_WHO_abbr, coef_df, permute_df, bootstrap_df)
final_df.sort_values("coef", ascending=False).to_csv(os.path.join(out_dir, "model_analysis.csv"), index=False)        

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")