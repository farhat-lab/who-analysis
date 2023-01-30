import numpy as np
import pandas as pd
import statsmodels.api as sm
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

_, config_file, drug, _ = sys.argv

kwargs = yaml.safe_load(open(config_file))    
binary = kwargs["binary"]
tiers_lst = kwargs["tiers_lst"]
synonymous = kwargs["synonymous"]
alpha = kwargs["alpha"]
model_prefix = kwargs["model_prefix"]
pheno_category_lst = kwargs["pheno_category_lst"]
atu_analysis = kwargs["atu_analysis"]
atu_analysis_type = kwargs["atu_analysis_type"]
analysis_dir = kwargs["output_dir"]
num_PCs = kwargs["num_PCs"]

if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
else:
    phenos_name = "WHO"

scaler = StandardScaler()

out_dir = os.path.join(analysis_dir, drug, f"BINARY/analysis/phenos={phenos_name}")
print(f"Saving results to {out_dir}")

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
    
# run likelihood ratio test only for BINARY models currently
if not binary or atu_analysis:
    exit()


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
matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True, how="inner")
print(f"{matrix.shape[0]} samples and {matrix.shape[1]} variables in the largest {phenos_name} model")

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
    
    
results_df = pd.DataFrame(columns=["log_like", "chi_stat", "pval", "AUC", "Sens", "Spec", "accuracy"])


############# STEP 3: FIT L2-PENALIZED REGRESSION FOR THE LARGEST MODEL #############


def run_regression_for_LRT(matrix, y, results_df, mutation=None, reg_param=None):
    
    if mutation is None:
        print("Fitting full regression...")
        X = matrix.values
        
        model = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
                                      cv=5,
                                      penalty='l2',
                                      max_iter=10000, 
                                      multi_class='ovr',
                                      scoring='neg_log_loss',
                                      class_weight='balanced'
                                     )
    
    else:
        X = remove_single_mut(matrix, mutation).values
    
        model = LogisticRegression(C=reg_param, 
                                   penalty='l2',
                                   max_iter=10000, 
                                   multi_class='ovr',
                                   class_weight='balanced'
                                  )
    X = scaler.fit_transform(X)
    model.fit(X, y)
    
    if mutation is None:
        pickle.dump(model, open(os.path.join(out_dir, f"tiers={'+'.join(tiers_lst)}_model.sav"), "wb"))
    
    # get positive class probabilities and predicted classes after determining the binarization threshold
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = get_threshold_val_and_classes(y_prob, y)

    # log-likelihood is the negative of the log-loss. Y_PRED MUST BE THE PROBABILITIES. set normalize=False to get sum of the log-likelihoods
    log_like = -sklearn.metrics.log_loss(y_true=y, y_pred=y_prob, normalize=False)
    
    if mutation is None:        
        chi_stat = np.nan
        pval = np.nan
        idx = "FULL"
    else:
        chi_stat = 2 * (results_df.loc["FULL", "log_like"] - log_like)
        pval = 1 - st.chi2.cdf(x=chi_stat, df=1)
        idx = mutation
        
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=y, y_pred=y_pred).ravel()
     
    results_df.loc[idx, :] = [log_like, chi_stat, pval,
                               sklearn.metrics.roc_auc_score(y_true=y, y_score=y_prob),
                               tp / (tp + fn),
                               tn / (tn + fp),
                               sklearn.metrics.accuracy_score(y_true=y, y_pred=y_pred),
                              ]
    
    return results_df


# run regressions for the full model
results_df = run_regression_for_LRT(matrix, y, results_df, mutation=None, reg_param=None)
model = pickle.load(open(os.path.join(out_dir, f"tiers={'+'.join(tiers_lst)}_model.sav"), "rb"))
reg_param = model.C_[0]

    
############# STEP 4: GET ALL MUTATIONS TO PERFORM THE LRT FOR AND FIT L2-PENALIZED REGRESSIONS FOR ALL MODELS WITH 1 FEATURE REMOVED #############


print(f"Performing LRT for {len(mutations_lst)} mutations\n")
    
for i, mutation in enumerate(mutations_lst):
    
    results_df = run_regression_for_LRT(matrix, y, results_df, mutation, reg_param)
    
    if i % 100 == 0:
        print(f"Finished {mutation}")
    
results_df.to_csv(os.path.join(out_dir, f"tier={tiers_lst[-1]}_LRT_results.csv"))

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")