import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
import scipy.stats as st
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
import tracemalloc, pickle

# analysis utils is in the analysis folder
sys.path.append(os.path.join(os.getcwd(), "analysis"))
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
    
    
# set tier-based threshold. Also, if it's a tier 1 + 2 model, only performing the AUC test for the tier 2 mutations, so remove tier 1
if len(tiers_lst) == 1:
    thresh = 0.05
    tier1_mutations = []
else:
    tier1_mutations = pd.read_pickle(os.path.join(out_dir.replace("tiers=1+2", "tiers=1"), "model_matrix.pkl")).columns
    thresh = 0.01
     
# keep only significant mutations in Ridge regression
ridge_results = pd.read_csv(os.path.join(out_dir, "model_analysis.csv"))

# the LRT dataframe does not contain principal components or tier 1 mutations, so they don't need to be removed
LRT_results = pd.read_csv(os.path.join(out_dir, "LRT_results.csv")).iloc[1:, :]
LRT_results.rename(columns={LRT_results.columns[0]: "mutation"}, inplace=True)
LRT_results = add_pval_corrections(LRT_results)

# get all mutations that have significant p-values in BOTH Ridge or LRT. Ridge is a criteria for Assoc, LRT is for Assoc - strict, so both must be satisfied
mutations_lst = list(set(ridge_results.query("~mutation.str.contains('PC') & mutation not in @tier1_mutations & BH_pval < @thresh")["mutation"]).intersection(LRT_results.query("BH_pval < @thresh")["mutation"]))

del ridge_results
del LRT_results

if len(mutations_lst) == 0:
    print(f"There are no significant tier {tiers_lst[-1]} mutations to run the AUC test on")
    
    ###################################### REMOVE THIS SOON ######################################
    if os.path.isfile(os.path.join(out_dir, "AUC_test_results.csv")):
        os.remove(os.path.join(out_dir, "AUC_test_results.csv"))
    ###################################### REMOVE THIS SOON ######################################
    exit()
    
# mutations_lst = list(set(ridge_results.query("~mutation.str.contains('PC') & BH_pval < @thresh")["mutation"]).union(LRT_results.query("~mutation.str.contains('PC') & BH_pval < @thresh")["mutation"]))

# get all mutations that have significant odds ratios and with confidence intervals significantly above or below 1
# mutations_lst = ridge_results.query("~mutation.str.contains('PC') & BH_pval < @thresh & (OR_LB > 1 | OR_UB < 1)")["mutation"].values

if os.path.isfile(os.path.join(out_dir, "AUC_test_results.csv")):
    df_auc = pd.read_csv(os.path.join(out_dir, "AUC_test_results.csv"))
    
    if len(set(mutations_lst) - set(df_auc["mutation"])) == 0:
        print("AUC test was already performed for this model")
        df_auc = df_auc.query("mutation in @mutations_lst")
        df_auc.to_csv(os.path.join(out_dir, "AUC_test_results.csv"), index=False)
        exit()
    

############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES FOR MODEL FITTING #############


df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv")).set_index("sample_id")
    
# Read in the PC coordinates dataframe, then keep only the desired number of principal components
eigenvec_df = pd.read_csv("data/eigenvec_10PC.csv", index_col=[0]).iloc[:, :num_PCs]

# concatenate the eigenvectors to the matrix and check the index ordering against the phenotypes matrix
matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True, how="inner")
df_phenos = df_phenos.loc[matrix.index]
assert sum(matrix.index != df_phenos.index.values) == 0
y = df_phenos["phenotype"].values


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
del model
    
    
def logReg_permutation_difference(matrix, y, mutation, num_bootstrap, reg_param):
    
    X = scaler.fit_transform(matrix.values)
    X_small = scaler.fit_transform(remove_single_mut(matrix, mutation).values)
    
    # fit original models
    true_null_model = LogisticRegression(C=reg_param, penalty='l2', max_iter=10000, multi_class='ovr', class_weight='balanced')
    true_null_model.fit(X, y)

    true_alt_model = LogisticRegression(C=reg_param, penalty='l2', max_iter=10000, multi_class='ovr', class_weight='balanced')
    true_alt_model.fit(X_small, y)
    
    true_null_auc = get_binary_metrics_from_model(true_null_model, X, y, 0)
    true_alt_auc = get_binary_metrics_from_model(true_alt_model, X_small, y, 0)
    true_diff = true_null_auc - true_alt_auc
    
    diff_auc = []
    
    for _ in range(num_bootstrap):

        y_permute = y.copy()
        np.random.shuffle(y_permute)

        null_model = LogisticRegression(C=reg_param, penalty='l2', max_iter=10000, multi_class='ovr', class_weight='balanced')
        null_model.fit(X, y_permute)
        
        alt_model = LogisticRegression(C=reg_param, penalty='l2', max_iter=10000, multi_class='ovr', class_weight='balanced')
        alt_model.fit(X_small, y_permute)

        null_auc = get_binary_metrics_from_model(null_model, X, y_permute, 0)
        alt_auc = get_binary_metrics_from_model(alt_model, X_small, y_permute, 0)
        
        # null hypothesis is that the model with all features is better than the alternative model (1 feature dropped)
        diff_auc.append(null_auc - alt_auc)
        
    # p-value is the number of permuted samples that have a test statistic AT LEAST AS extreme as the true statistic
    if true_diff > 0:
        return true_diff, np.mean(np.array(diff_auc) >= true_diff)
    else:
        return true_diff, np.mean(np.array(diff_auc) <= true_diff)
            
        
num_bootstrap = 100
print(f"\nPerforming AUC permutation test on {len(mutations_lst)} mutations with {num_bootstrap} replicates")

if os.path.isfile(os.path.join(out_dir, "AUC_test_results.csv")):
    diff_df = pd.read_csv(os.path.join(out_dir, "AUC_test_results.csv"))
    print(f"Already completed {len(diff_df)}/{len(mutations_lst)} mutations")
else:    
    diff_df = pd.DataFrame(columns=["mutation", "AUC_diff", "pval"])

for i, mutation in enumerate(mutations_lst):
    
    if mutation not in diff_df["mutation"].values:
        diff, pval = logReg_permutation_difference(matrix, y, mutation, num_bootstrap, reg_param)
        diff_df = pd.concat([diff_df, 
                             pd.DataFrame({"mutation": mutation, "AUC_diff": diff, "pval": pval}, index=[-1])
                            ], axis=0)

    if i % 10 == 0:
        diff_df.to_csv(os.path.join(out_dir, "AUC_test_results.csv"), index=False)
        print(i)
      
    
diff_df = add_pval_corrections(diff_df)
diff_df.to_csv(os.path.join(out_dir, "AUC_test_results.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")