import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
import scipy.stats as st
import sklearn.metrics
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
import tracemalloc, pickle
who_variants_combined = pd.read_csv("analysis/who_confidence_2021.csv")

# analysis utils is in the analysis folder
sys.path.append(os.path.join(os.getcwd(), "analysis"))
from stats_utils import *


############# STEP 0: READ IN PARAMETERS FILE AND GET DIRECTORIES #############

    
# starting the memory monitoring
tracemalloc.start()

_, config_file, drug, drug_WHO_abbr = sys.argv

kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
analysis_dir = kwargs["output_dir"]

model_prefix = kwargs["model_prefix"]
num_PCs = kwargs["num_PCs"]

pheno_category_lst = kwargs["pheno_category_lst"]
# make sure that both phenotypes are included
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
    pheno_category_lst = ["ALL", "WHO"]
else:
    phenos_name = "WHO"

scaler = StandardScaler()

if not os.path.isdir(os.path.join(analysis_dir, drug, "BINARY/LRT")):
    os.makedirs(os.path.join(analysis_dir, drug, "BINARY/LRT"))
    
out_dir = os.path.join(analysis_dir, drug, f"BINARY/LRT/phenos={phenos_name}")
    

############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############


phenos_file = os.path.join(analysis_dir, drug, "phenos_binary.csv")
df_phenos = pd.read_csv(phenos_file).set_index("sample_id")

# different matrices, depending on the phenotypes. Get all mutations 
# no model (basically just for Pretomanid because there are no WHO phenotypes, so some models don't exist)
if not os.path.isfile(os.path.join(analysis_dir, drug, f"BINARY/tiers={'+'.join(tiers_lst)}/phenos={phenos_name}/{model_prefix}/model_matrix.pkl")):
    exit()
else:
    matrix = pd.read_pickle(os.path.join(analysis_dir, drug, f"BINARY/tiers={'+'.join(tiers_lst)}/phenos={phenos_name}/{model_prefix}/model_matrix.pkl"))

mutations_lst = matrix.columns

# if this is for a tier 1 + 2 model, only compute LRT for tier 2 mutations, so remove the tier 1 mutations
if len(tiers_lst) == 2:
    tier1_mutations = pd.read_pickle(os.path.join(analysis_dir, drug, f"BINARY/tiers=1/phenos={phenos_name}/{model_prefix}/model_matrix.pkl")).columns
    mutations_lst = list(set(mutations_lst) - set(tier1_mutations))

# Read in the PC coordinates dataframe, then keep only the desired number of principal components
eigenvec_df = pd.read_csv("data/eigenvec_10PC.csv", index_col=[0]).iloc[:, :num_PCs]

# concatenate the eigenvectors to the matrix and check the index ordering against the phenotypes matrix
matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True)
df_phenos = df_phenos.loc[matrix.index]
assert sum(matrix.index != df_phenos.index.values) == 0

X = scaler.fit_transform(matrix.values)
y = df_phenos["phenotype"].values
print(f"{matrix.shape[0]} samples and {matrix.shape[1]} variables in the largest {phenos_name} model")


############# STEP 2: READ IN THE ORIGINAL DATA: MODEL_MATRIX PICKLE FILE FOR A GIVEN MODEL #############

    
def remove_single_mut(matrix, mutation):
    
    if mutation not in matrix.columns:
        raise ValueError(f"{mutation} is not in the argument matrix!")
    
    small_matrix = matrix.loc[:, matrix.columns != mutation]
    assert small_matrix.shape[1] + 1 == matrix.shape[1]
    return small_matrix
    
    
# this model was trained in 03_likelihood_ratio_test.py, so load it back in 
model = pickle.load(open(os.path.join(out_dir, f"tiers={'+'.join(tiers_lst)}_model.sav"), "rb"))
reg_param = model.C_[0]

def cross_validate_binary_metrics(X, y, reg_param, mutation=None):
    
    if mutation is None:
        print("Fitting full regression...")
        X = matrix.values
        print(X.shape)
    else:
        X = remove_single_mut(matrix, mutation).values
    
    X = scaler.fit_transform(X)

    cv_model = LogisticRegression(C=reg_param, 
                                  penalty='l2', 
                                  max_iter=10000, 
                                  multi_class='ovr',
                                  class_weight='balanced'
                                 )
    
    # dictionary, where everything has the prefix test_ for the test set metrics
    scores = cross_validate(cv_model, 
                             X, 
                             y, 
                             cv=5,
                             scoring=['roc_auc', 'recall', 'accuracy', 'balanced_accuracy'] # recall = sensitivity. balanced_accuracy = 1/2(sens + spec)
                            )

    scores["test_spec"] = 2 * scores['test_balanced_accuracy'] - scores['test_recall']
    scores["test_sens"] = scores.pop("test_recall")

    res_df = pd.DataFrame.from_dict(scores)
    return res_df

    
print(f"Performing cross-validation for binary metrics on {len(mutations_lst)} mutations")
summary_stats_all = pd.DataFrame(columns = ["mutation", "test_roc_auc", "test_sens", "test_spec", "test_accuracy", "test_balanced_accuracy"])

orig_df = cross_validate_binary_metrics(X, y, reg_param, mutation=None)
orig_df["mutation"] = "FULL"
summary_stats_all = pd.concat([summary_stats_all, orig_df], axis=0)


for i, mutation in enumerate(mutations_lst):  

    res_df = cross_validate_binary_metrics(X, y, reg_param, mutation=mutation)
    res_df["mutation"] = mutation
    summary_stats_all = pd.concat([summary_stats_all, res_df], axis=0)
    
    if i % 100 == 0:
        print(f"Finished {mutation}")
        
#     if i % 500 == 0:
#         summary_stats_all.to_csv(os.path.join(out_dir, f"tier={tiers_lst[-1]}_binaryMetrics_CV.csv"), index=False)
        
        
# save dataframe of all CV results
del summary_stats_all["fit_time"]
del summary_stats_all["score_time"]

# add WHO 2021 catalog variant designations
summary_stats_all = summary_stats_all.merge(who_variants_combined.query("drug==@drug_WHO_abbr")[["mutation", "confidence"]],
                       on="mutation", how="left"
                      )

summary_stats_all.to_csv(os.path.join(out_dir, f"tier={tiers_lst[-1]}_binaryMetrics_CV.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")