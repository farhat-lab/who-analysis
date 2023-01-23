import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
import scipy.stats as st
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
import tracemalloc, pickle
from stats_utils import *


############# STEP 0: READ IN PARAMETERS FILE AND GET DIRECTORIES #############

    
# starting the memory monitoring
tracemalloc.start()

_, config_file, drug = sys.argv

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
    os.makedirs(os.path.join(analysis_dir, drug, "BINARY/interaction"))
    

############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############


phenos_file = os.path.join(analysis_dir, drug, "phenos_binary.csv")
df_phenos = pd.read_csv(phenos_file).set_index("sample_id")

# different matrices, depending on the phenotypes
matrix = pd.read_pickle(os.path.join(analysis_dir, drug, "BINARY", f"tiers=1+2/phenos={phenos_name}", model_prefix, "model_matrix.pkl"))

# Read in the PC coordinates dataframe, then keep only the desired number of principal components
eigenvec_df = pd.read_csv("data/eigenvec_10PC.csv", index_col=[0]).iloc[:, :num_PCs]


############# STEP 2: READ IN THE ORIGINAL DATA: MODEL_MATRIX PICKLE FILE FOR A GIVEN MODEL #############


def read_in_data(matrix, df_phenos):
        
    # keep only eigenvec coordinates for samples in the matrix dataframe
    eigenvec = eigenvec_df.loc[matrix.index]
    df_phenos = df_phenos.loc[matrix.index]
    
    assert sum(matrix.merge(eigenvec, left_index=True, right_index=True).index != df_phenos.index.values) == 0

    # concatenate the eigenvectors to the matrix and check the index ordering against the phenotypes matrix
    matrix = matrix.merge(eigenvec, left_index=True, right_index=True)

    return matrix, df_phenos["phenotype"].values
    

    
def remove_single_mut(large_matrix, mutation):
    
    if mutation not in large_matrix.columns:
        raise ValueError(f"{mutation} is not in the argument matrix!")
    
    small_matrix = large_matrix.loc[:, large_matrix.columns != mutation]
    assert small_matrix.shape[1] + 1 == large_matrix.shape[1]
    return small_matrix
    
    

large_matrix, y = read_in_data(matrix, df_phenos)

print(f"{large_matrix.shape[0]} samples and {large_matrix.shape[1]} variables in the largest {phenos_name} model")

results_df = pd.DataFrame(columns=["penalty", "log_like", "chi_stat", "pval", "AUC", "Sens", "Spec", "accuracy"])


############# STEP 3: FIT L2-PENALIZED REGRESSION FOR THE LARGEST MODEL #############


def run_regression_for_LRT(matrix, y, results_df, mutation=None):
    
    if mutation is None:
        print("Fitting full regression...")
        X = matrix.values
    else:
        X = remove_single_mut(matrix, mutation).values
        
    model = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
                                  cv=5,
                                  penalty='l2',
                                  max_iter=10000, 
                                  multi_class='ovr',
                                  scoring='neg_log_loss',
                                  class_weight='balanced'
                                 )
    
    X = scaler.fit_transform(X)
    model.fit(X, y)
    
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
     
    results_df.loc[idx, :] = [model.C_[0], log_like, chi_stat, pval,
                               sklearn.metrics.roc_auc_score(y_true=y, y_score=y_prob),
                               tp / (tp + fn),
                               tn / (tn + fp),
                               sklearn.metrics.accuracy_score(y_true=y, y_pred=y_pred),
                              ]
    
    return results_df


# run regressions for the full models
results_df = run_regression_for_LRT(large_matrix, y, results_df, mutation=None)
print(results_df)

    
############# STEP 4: GET ALL MUTATIONS TO PERFORM THE LRT FOR AND FIT L2-PENALIZED REGRESSIONS FOR ALL MODELS WITH 1 FEATURE REMOVED #############


tier2_mutations = get_tier2_mutations_of_interest(analysis_dir, drug, phenos_name)

    
for i, mutation in enumerate(tier2_mutations):
    
    print(f"\nWorking on {mutation}")
    results_df = run_regression_for_LRT(large_matrix, y, results_df, mutation)

    if i % 10 == 0:
        results_df.to_csv(os.path.join(analysis_dir, drug, "BINARY/LRT", f"{phenos_name}phenos_results.csv"))


results_df.to_csv(os.path.join(analysis_dir, drug, "BINARY/LRT", f"{phenos_name}phenos_results.csv"))
print(f"{len(results_df['penalty'].unique())} unique penalty terms in the {phenos_name} analysis")

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")