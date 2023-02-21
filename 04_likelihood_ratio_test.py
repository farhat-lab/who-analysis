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

_, config_file, drug = sys.argv

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

# run likelihood ratio test only for BINARY models currently
if not binary or atu_analysis:
    exit()
    
if binary:
    if atu_analysis:
        out_dir = os.path.join(analysis_dir, drug, "ATU", f"tiers={'+'.join(tiers_lst)}", model_prefix)
        
        # the CC and CC-ATU models are in the same folder, but the output files (i.e. regression_coef.csv have different suffixes to distinguish)
        model_suffix = kwargs["atu_analysis_type"]
        assert model_suffix == "CC" or model_suffix == "CC-ATU"
    else:
        out_dir = os.path.join(analysis_dir, drug, "BINARY", f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}", model_prefix)
        model_suffix = ""

print(f"Saving results to {out_dir}")
    
if os.path.isfile(os.path.join(out_dir, "LRT_results.csv")):
    print("LRT was already performed for this model")
    exit()
       

############# STEP 1: READ IN THE ORIGINAL DATA: MODEL_MATRIX PICKLE FILE FOR A GIVEN MODEL #############


phenos_file = os.path.join(analysis_dir, drug, "phenos_binary.csv")
df_phenos = pd.read_csv(phenos_file).set_index("sample_id")

# different matrices, depending on the phenotypes. Get all mutations 
# no model (basically just for Pretomanid because there are no WHO phenotypes, so some models don't exist)
if not os.path.isfile(os.path.join(out_dir, "model_matrix.pkl")):
    print("There is no model for this LRT")
    exit()
else:
    matrix = pd.read_pickle(os.path.join(out_dir, "model_matrix.pkl"))
    
mutations_lst = matrix.columns

# if this is for a tier 1 + 2 model, only compute LRT for tier 2 mutations because the script takes a while to run, so remove the tier 1 mutations
if len(tiers_lst) == 2:
    tier1_mutations = pd.read_pickle(os.path.join(out_dir.replace("tiers=1+2", "tiers=1"), "model_matrix.pkl")).columns
    mutations_lst = list(set(mutations_lst) - set(tier1_mutations))
 
# Read in the PC coordinates dataframe, then keep only the desired number of principal components
eigenvec_df = pd.read_csv("data/eigenvec_10PC.csv", index_col=[0]).iloc[:, :num_PCs]

# concatenate the eigenvectors to the matrix and check the index ordering against the phenotypes matrix
matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True, how="inner")
print(f"{matrix.shape[0]} samples and {matrix.shape[1]} variables in the largest {phenos_name} model")

df_phenos = df_phenos.loc[matrix.index]
assert sum(matrix.index != df_phenos.index.values) == 0
y = df_phenos["phenotype"].values


def remove_single_mut(matrix, mutation):
    '''
    Function to remove a single mutation from the matrix
    '''
    
    if mutation not in matrix.columns:
        raise ValueError(f"{mutation} is not in the argument matrix!")
    
    small_matrix = matrix.loc[:, matrix.columns != mutation]
    assert small_matrix.shape[1] + 1 == matrix.shape[1]
    return small_matrix
    

############# STEP 2: COMPUTE METRICS FOR THE LARGEST MODEL USING THE PREVIOUSLY SAVED MODEL #############


def run_regression_for_LRT(matrix, y, results_df, mutation, reg_param):
    
    # remove mutation from the matrix
    X = remove_single_mut(matrix, mutation).values
    X = scaler.fit_transform(X)

    # fit model using the specified regularization parameter
    model = LogisticRegression(C=reg_param, 
                               penalty='l2',
                               max_iter=10000, 
                               multi_class='ovr',
                               class_weight='balanced'
                              )
    model.fit(X, y)

    # get positive class probabilities and predicted classes after determining the binarization threshold
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = get_threshold_val_and_classes(y_prob, y)

    # log-likelihood is the negative of the log-loss. Y_PRED MUST BE THE PROBABILITIES. set normalize=False to get sum of the log-likelihoods
    log_like = -sklearn.metrics.log_loss(y_true=y, y_pred=y_prob, normalize=False)
    
    # null hypothesis is that full model log-like > alt model log-like. Larger log-like is a better model
    # positive chi_stat: mutation increases AUC. so removing it from the model DECREASES log_like from the FULL log-like
    # negative chi_stat: mutation decreases AUC, so removing it from the model INCREASES log_like over the FULL log-like
    chi_stat = 2 * (results_df.loc["FULL", "log_like"] - log_like)
    pval = 1 - st.chi2.cdf(x=chi_stat, df=1)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=y, y_pred=y_pred).ravel()
     
    results_df.loc[mutation, :] = [log_like, chi_stat, pval,
                                   sklearn.metrics.roc_auc_score(y_true=y, y_score=y_prob),
                                   tp / (tp + fn),
                                   tn / (tn + fp),
                                   sklearn.metrics.accuracy_score(y_true=y, y_pred=y_pred),
                                  ]
    return results_df


# load in original model to get the regularization term
model = pickle.load(open(os.path.join(out_dir, "model.sav"), "rb"))
reg_param = model.C_[0]

# get positive class probabilities and predicted classes after determining the binarization threshold
y_prob = model.predict_proba(scaler.fit_transform(matrix.values))[:, 1]
y_pred = get_threshold_val_and_classes(y_prob, y)

# log-likelihood is the negative of the log-loss. Y_PRED MUST BE THE PROBABILITIES. set normalize=False to get sum of the log-likelihoods
log_like = -sklearn.metrics.log_loss(y_true=y, y_pred=y_prob, normalize=False)
tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=y, y_pred=y_pred).ravel()
        
# add the results for the full model
results_df = pd.DataFrame(columns=["log_like", "chi_stat", "pval", "AUC", "Sens", "Spec", "accuracy"])
results_df.loc["FULL", :] = [log_like, np.nan, np.nan,
                               sklearn.metrics.roc_auc_score(y_true=y, y_score=y_prob),
                               tp / (tp + fn),
                               tn / (tn + fp),
                               sklearn.metrics.accuracy_score(y_true=y, y_pred=y_pred),
                              ]

    
############# STEP 3: GET ALL MUTATIONS TO PERFORM THE LRT FOR AND FIT L2-PENALIZED REGRESSIONS FOR ALL MODELS WITH 1 FEATURE REMOVED #############


print(f"Performing LRT for {len(mutations_lst)} mutations\n")
    
for i, mutation in enumerate(mutations_lst):
    
    results_df = run_regression_for_LRT(matrix, y, results_df, mutation, reg_param)
    
    if i % 100 == 0:
        print(f"Finished {mutation}")
    
results_df.to_csv(os.path.join(out_dir, "LRT_results.csv"))

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")