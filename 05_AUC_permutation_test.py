import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
import scipy.stats as st
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV, SGDClassifier, SGDRegressor
import tracemalloc, pickle

# utils files are in a separate folder
sys.path.append("utils")
from stats_utils import *


############# STEP 0: READ IN PARAMETERS FILE AND GET DIRECTORIES #############

    
# starting the memory monitoring
tracemalloc.start()

_, config_file, drug = sys.argv

kwargs = yaml.safe_load(open(config_file))

binary = True
tiers_lst = kwargs["tiers_lst"]
analysis_dir = kwargs["output_dir"]

model_prefix = kwargs["model_prefix"]
num_PCs = kwargs["num_PCs"]
eigenvec_df = pd.read_csv("data/eigenvec_100PC.csv", usecols=["sample_id"] + [f"PC{num+1}" for num in np.arange(num_PCs)]).set_index("sample_id")
num_bootstrap = kwargs["num_bootstrap"]

pheno_category_lst = kwargs["pheno_category_lst"]
# make sure that both phenotypes are included
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
    pheno_category_lst = ["ALL", "WHO"]
else:
    phenos_name = "WHO"

scaler = StandardScaler()

out_dir = os.path.join(analysis_dir, drug, "BINARY", f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}", model_prefix)        
print(f"Saving results to {out_dir}")
    
# no model (basically just for Pretomanid because there are no WHO phenotypes, so some models don't exist)
if not os.path.isfile(os.path.join(out_dir, "model_matrix.pkl")):
    print("There is no model for this AUC test")
    exit()
else:
    matrix = pd.read_pickle(os.path.join(out_dir, "model_matrix.pkl"))
    
if os.path.isfile(os.path.join(out_dir, "AUC_test_results.csv")):
    print("AUC test was already performed for this model")
    exit()
    
mutations_lst = matrix.columns

# if this is for a tier 1 + 2 model, only compute AUC for tier 2 mutations only to save time
if len(tiers_lst) == 2:
    tier1_mutations = pd.read_pickle(os.path.join(out_dir.replace("tiers=1+2", "tiers=1"), "model_matrix.pkl")).columns
    mutations_lst = list(set(mutations_lst) - set(tier1_mutations))
    
# merge with eigenvectors
matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True, how="inner")
print(f"{matrix.shape[0]} samples and {matrix.shape[1]} variables in the largest model")
    
# keep only samples (rows) that are in matrix and use loc with indices to ensure they are in the same order
phenos_file = os.path.join(analysis_dir, drug, "phenos_binary.csv")
df_phenos = pd.read_csv(phenos_file).set_index("sample_id")
df_phenos = df_phenos.loc[matrix.index]

# check that the sample ordering is the same in the genotype and phenotype matrices
assert sum(matrix.index != df_phenos.index) == 0
    
# scale inputs
scaler = StandardScaler()
X = scaler.fit_transform(matrix.values)
y = df_phenos["phenotype"].values
assert len(np.unique(y)) == 2
    
# # set tier-based threshold. Also, if it's a tier 1 + 2 model, only performing the AUC test for the tier 2 mutations, so remove tier 1
# if len(tiers_lst) == 1:
#     thresh = 0.05
#     tier1_mutations = []
# else:
#     tier1_mutations = pd.read_pickle(os.path.join(out_dir.replace("tiers=1+2", "tiers=1"), "model_matrix.pkl")).columns
#     thresh = 0.01
     
# # keep only significant mutations in Ridge regression
# ridge_results = pd.read_csv(os.path.join(out_dir, "model_analysis.csv"))

# # the LRT dataframe does not contain principal components or tier 1 mutations, so they don't need to be removed
# LRT_results = pd.read_csv(os.path.join(out_dir, "LRT_results.csv")).iloc[1:, :]
# LRT_results.rename(columns={LRT_results.columns[0]: "mutation"}, inplace=True)
# LRT_results = add_pval_corrections(LRT_results)

# # get all mutations that have significant p-values in BOTH Ridge or LRT. Ridge is a criteria for Assoc, LRT is for Assoc - strict, so both must be satisfied
# mutations_lst = list(set(ridge_results.query("~mutation.str.contains('PC') & mutation not in @tier1_mutations & BH_pval < @thresh")["mutation"]).intersection(LRT_results.query("BH_pval < @thresh")["mutation"]))

# del ridge_results
# del LRT_results

# if len(mutations_lst) == 0:
#     print(f"There are no significant tier {tiers_lst[-1]} mutations to run the AUC test on")
#     exit()


############# STEP 2: READ IN THE ORIGINAL DATA: MODEL_MATRIX PICKLE FILE FOR A GIVEN MODEL #############

    
def remove_single_mut(matrix, mutation):
    
    if mutation not in matrix.columns:
        raise ValueError(f"{mutation} is not in the argument matrix!")
    
    small_matrix = matrix.loc[:, matrix.columns != mutation]
    assert small_matrix.shape[1] + 1 == matrix.shape[1]
    return small_matrix
    
    
# load in original model to get the regularization term
model = pickle.load(open(os.path.join(out_dir, "model.sav"), "rb"))
reg_param = model.C_[0]

# model has already been fit, so extract metrics
true_null_auc = get_binary_metrics_from_model(model, X, y)["AUC"]
print(f"AUC for model with all mutations: {true_null_auc}")
    
    
def logReg_permutation_difference(matrix, X, y, true_null_auc, mutation, num_bootstrap, reg_param):
    
    X_small = scaler.fit_transform(remove_single_mut(matrix, mutation).values)
    
    # fit original model without the mutation of interest 
    true_alt_model = LogisticRegression(C=reg_param, penalty='l2', max_iter=10000, multi_class='ovr', class_weight='balanced')
    true_alt_model.fit(X_small, y)
    
    true_alt_auc = get_binary_metrics_from_model(true_alt_model, X_small, y)["AUC"]
    true_diff = true_null_auc - true_alt_auc
    return true_diff
    
#     diff_auc = []
    
#     for _ in range(num_bootstrap):

#         y_permute = y.copy()
#         np.random.shuffle(y_permute)

#         null_model = SGDClassifier(loss='log_loss', 
#                                   penalty='l2', 
#                                   alpha=reg_param, 
#                                   l1_ratio=0,
#                                   fit_intercept=True,
#                                   max_iter=1000000,
#                                   n_jobs=-1,
#                                   tol=1e-6,
#                                   n_iter_no_change=100,
#                                   learning_rate='optimal', 
#                                   early_stopping=True, 
#                                   validation_fraction=0.25, 
#                                   class_weight="balanced"
#                                  )
#         null_model.fit(X, y_permute)
        
#         alt_model = SGDClassifier(loss='log_loss', 
#                                   penalty='l2', 
#                                   alpha=reg_param, 
#                                   l1_ratio=0,
#                                   fit_intercept=True,
#                                   max_iter=1000000,
#                                   n_jobs=-1,
#                                   tol=1e-6,
#                                   n_iter_no_change=100,
#                                   learning_rate='optimal', 
#                                   early_stopping=True, 
#                                   validation_fraction=0.25, 
#                                   class_weight="balanced"
#                                  )
#         alt_model.fit(X_small, y_permute)

#         null_auc = get_binary_metrics_from_model(null_model, X, y_permute)["AUC"]
#         alt_auc = get_binary_metrics_from_model(alt_model, X_small, y_permute)["AUC"]
        
#         # null hypothesis is that the model with all features is better than the alternative model (1 feature dropped)
#         diff_auc.append(null_auc - alt_auc)
        
#     # p-value is the number of permuted samples that have a test statistic AT LEAST AS extreme as the true statistic
#     if true_diff > 0:
#         return true_diff, np.mean(np.array(diff_auc) >= true_diff)
#     else:
#         return true_diff, np.mean(np.array(diff_auc) <= true_diff)
            
       
# print(f"\nPerforming AUC permutation test on {len(mutations_lst)} mutations with {num_bootstrap} replicates")

# diff_df = pd.DataFrame(columns=["mutation", "AUC_diff", "pval"])
diff_df = pd.DataFrame(columns=["mutation", "AUC_diff"])

for i, mutation in enumerate(mutations_lst):
    
    diff = logReg_permutation_difference(matrix, X, y, true_null_auc, mutation, num_bootstrap, reg_param)
    # diff_df = pd.concat([diff_df, 
    #                      pd.DataFrame({"mutation": mutation, "AUC_diff": diff, "pval": pval}, index=[-1])
    #                     ], axis=0)
    diff_df = pd.concat([diff_df, 
                         pd.DataFrame({"mutation": mutation, "AUC_diff": diff}, index=[-1])
                        ], axis=0)

#     if i % 10 == 0:
#         print(i)
      
# diff_df = add_pval_corrections(diff_df)
diff_df.to_csv(os.path.join(out_dir, "AUC_test_results.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")