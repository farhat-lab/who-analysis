import numpy as np
import pandas as pd
import statsmodels.api as sm
import glob, os, yaml, sparse, sys
import scipy.stats as st
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import tracemalloc, pickle

# utils files are in a separate folder
sys.path.append("utils")
from stats_utils import *


############# STEP 0: READ IN PARAMETERS FILE AND GET DIRECTORIES #############

    
# starting the memory monitoring
tracemalloc.start()

_, config_file, drug = sys.argv

kwargs = yaml.safe_load(open(config_file))    
binary = kwargs["binary"]
tiers_lst = kwargs["tiers_lst"]
silent = kwargs["silent"]
alpha = kwargs["alpha"]
model_prefix = kwargs["model_prefix"]
pheno_category_lst = kwargs["pheno_category_lst"]
atu_analysis = kwargs["atu_analysis"]
atu_analysis_type = kwargs["atu_analysis_type"]
analysis_dir = kwargs["output_dir"]
num_PCs = kwargs["num_PCs"]
eigenvec_df = pd.read_csv("PCA/eigenvec_100PC.csv", usecols=["sample_id"] + [f"PC{num+1}" for num in np.arange(num_PCs)]).set_index("sample_id")

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

if not os.path.isfile(os.path.join(out_dir, "model.sav")):
    print("There is no regression model for this analysis")
    exit()

    
############# STEP 1: READ IN THE ORIGINAL DATA: MODEL_MATRIX PICKLE FILE FOR A GIVEN MODEL #############


phenos_file = os.path.join(analysis_dir, drug, "phenos_binary.csv")
df_phenos = pd.read_csv(phenos_file).set_index("sample_id")

# no model (basically just for Pretomanid because there are no WHO phenotypes, so some models don't exist)
if not os.path.isfile(os.path.join(out_dir, "model_matrix.pkl")):
    print("There is no model for this LRT")
    exit()
else:
    matrix = pd.read_pickle(os.path.join(out_dir, "model_matrix.pkl"))

print(matrix.shape)
mutations_lst = matrix.columns

# if this is for a tier 1 + 2 model, only compute LRT for tier 2 mutations because the script takes a while to run, so remove the tier 1 mutations
if len(tiers_lst) == 2:
    tier1_mutations = pd.read_pickle(os.path.join(out_dir.replace("tiers=1+2", "tiers=1"), "model_matrix.pkl")).columns
    mutations_lst = list(set(mutations_lst) - set(tier1_mutations))
    
# # if this is for a +silent model, only compute LRT for the silent because the script takes a while to run, so remove the nonsyn mutations
# # LRT will have already been computed for these in the corresponding noSyn model
# if silent:
#     nonsyn_mutations = pd.read_pickle(os.path.join(out_dir.replace("withSyn", "noSyn"), "model_matrix.pkl")).columns
#     mutations_lst = list(set(mutations_lst) - set(nonsyn_mutations))
   
# # only compute LRT for the pooled mutations because the unpooled mutation data will come from the unpooled models
# if "poolSeparate" in model_prefix or "poolLoF" in model_prefix:
#     unpooled_mutations = pd.read_pickle(os.path.join(out_dir.replace("poolSeparate", "unpooled").replace("poolLoF", "unpooled"), "model_matrix.pkl")).columns
#     mutations_lst = list(set(mutations_lst) - set(unpooled_mutations))
    
# merge with eigenvectors
print(f"{matrix.shape[0]} samples and {matrix.shape[1]} variants in the largest model")
matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True, how="inner")

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
    

############# STEP 2: COMPUTE METRICS FOR THE LARGEST MODEL #############


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

    # log-likelihood is the negative of the log-loss. Y_PRED MUST BE THE PROBABILITIES. set normalize=False to get sum of the log-likelihoods
    y_prob = model.predict_proba(X)[:, 1]
    log_like = -sklearn.metrics.log_loss(y_true=y, y_pred=y_prob, normalize=False)
    
    # null hypothesis is that full model log-like > alt model log-like. Larger log-like is a better model
    chi_stat = 2 * (results_df.loc["FULL", "log_like"] - log_like)

    # get positive class probabilities and predicted classes after determining the binarization threshold
    y_pred = get_threshold_val_and_classes(y_prob, y)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=y, y_pred=y_pred).ravel()
     
    results_df.loc[mutation, :] = [log_like, 
                                   chi_stat, 
                                   st.chi2.sf(x=chi_stat, df=1), # p-value. p-value + neutral p-value = 1
                                   st.chi2.cdf(x=chi_stat, df=1), # neutral p-value testing the hypothesis that a mutation is NOT relevant
                                   sklearn.metrics.roc_auc_score(y_true=y, y_score=y_prob),
                                   tp / (tp + fn),
                                   tn / (tn + fp),
                                   sklearn.metrics.accuracy_score(y_true=y, y_pred=y_pred),
                                   sklearn.metrics.balanced_accuracy_score(y_true=y, y_pred=y_pred),
                                  ]
    return results_df


model = pickle.load(open(os.path.join(out_dir, "model.sav"), "rb"))
reg_param = model.C_[0]
print(f"Regularization parameter: {reg_param}")

# get positive class probabilities and predicted classes after determining the binarization threshold
y_prob = model.predict_proba(scaler.fit_transform(matrix.values))[:, 1]
y_pred = get_threshold_val_and_classes(y_prob, y)

# log-likelihood is the negative of the log-loss. Y_PRED MUST BE THE PROBABILITIES. set normalize=False to get sum of the log-likelihoods
log_like = -sklearn.metrics.log_loss(y_true=y, y_pred=y_prob, normalize=False)
tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=y, y_pred=y_pred).ravel()
        
    
############# STEP 3: GET ALL MUTATIONS TO PERFORM THE LRT FOR AND FIT L2-PENALIZED REGRESSIONS FOR ALL MODELS WITH 1 FEATURE REMOVED #############


if os.path.isfile(os.path.join(out_dir, "LRT_results.csv")):
    print("Results dataframe exists. Appending additional mutations")
    results_df = pd.read_csv(os.path.join(out_dir, "LRT_results.csv"), index_col=[0])
    finished_mutations = results_df.index.values
    mutations_lst = list(set(mutations_lst) - set(finished_mutations) - set(["FULL"]))

else:
    print("Creating LRT results dataframe")
    # create the results dataframe and add the results for the full model
    results_df = pd.DataFrame(columns=["log_like", "chi_stat", "LRT_pval", "LRT_neutral_pval", "AUC", "Sens", "Spec", "accuracy", "balanced_accuracy"])
    results_df.loc["FULL", :] = [log_like, 
                                 np.nan, np.nan, np.nan,
                                 sklearn.metrics.roc_auc_score(y_true=y, y_score=y_prob),
                                 tp / (tp + fn),
                                 tn / (tn + fp),
                                 sklearn.metrics.accuracy_score(y_true=y, y_pred=y_pred),
                                 sklearn.metrics.balanced_accuracy_score(y_true=y, y_pred=y_pred)
                                ]

if len(mutations_lst) == 0:
    print("Finished LRT for all mutations in this model")
else:
    print(f"Performing LRT for {len(mutations_lst)} mutations\n")
    
for i, mutation in enumerate(mutations_lst):
    
    results_df = run_regression_for_LRT(matrix, y, results_df, mutation, reg_param)
    
    if i % 100 == 0:
        print(f"Finished {mutation}")
        results_df.reset_index().rename(columns={"index":"mutation"}).to_csv(os.path.join(out_dir, "LRT_results.csv"), index=False)
    
results_df.reset_index().rename(columns={"index":"mutation"}).to_csv(os.path.join(out_dir, "LRT_results.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")