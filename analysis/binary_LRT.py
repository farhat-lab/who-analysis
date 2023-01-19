import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
import scipy.stats as st
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
import tracemalloc, pickle, warnings
warnings.filterwarnings("ignore")


############# STEP 0: READ IN PARAMETERS FILE AND GET DIRECTORIES #############

    
# starting the memory monitoring
tracemalloc.start()

_, config_file, drug, drug_WHO_abbr = sys.argv

kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
binary = kwargs["binary"]
atu_analysis = kwargs["atu_analysis"]
assert binary
assert not atu_analysis
analysis_dir = kwargs["output_dir"]

model_prefix = kwargs["model_prefix"]
num_PCs = kwargs["num_PCs"]
num_bootstrap = kwargs["num_bootstrap"]

scaler = StandardScaler()
    

############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############


phenos_file = os.path.join(analysis_dir, drug, "phenos_binary.csv")
df_phenos = pd.read_csv(phenos_file).set_index("sample_id")

# different matrices, depending on the phenotypes
who_matrix = pd.read_pickle(os.path.join(analysis_dir, drug, "BINARY", f"tiers={'+'.join(tiers_lst)}", "phenos=WHO", model_prefix, "model_matrix.pkl"))
all_matrix = pd.read_pickle(os.path.join(analysis_dir, drug, "BINARY", f"tiers={'+'.join(tiers_lst)}", "phenos=ALL", model_prefix, "model_matrix.pkl"))

# Read in the PC coordinates dataframe, then keep only the desired number of principal components
eigenvec_df = pd.read_csv("data/eigenvec_10PC.csv", index_col=[0])
eigenvec_df = eigenvec_df.iloc[:, :num_PCs]


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
    
    

who_large_matrix, who_y = read_in_data(who_matrix, df_phenos)
all_large_matrix, all_y = read_in_data(all_matrix, df_phenos)

print(f"{who_large_matrix.shape[0]} samples and {who_large_matrix.shape[1]} variables in the largest WHO model")
print(f"{all_large_matrix.shape[0]} samples and {all_large_matrix.shape[1]} variables in the largest ALL model")

results_who = pd.DataFrame(columns=["penalty", "log_like", "chi_stat", "pval", "AUC", "accuracy", "balanced_acc"])
results_all = pd.DataFrame(columns=["penalty", "log_like", "chi_stat", "pval", "AUC", "accuracy", "balanced_acc"])


############# STEP 3: FIT L2-PENALIZED REGRESSION FOR THE LARGEST MODEL #############


def get_threshold_val_and_classes(y_prob, y_test):
    
    # Compute number resistant and sensitive
    num_samples = len(y_test)
    num_resistant = np.sum(y_test).astype(int)
    num_sensitive = num_samples - num_resistant
    
    pred_df = pd.DataFrame({"y_prob": y_prob, "y_test": y_test})

    # Test thresholds from 0 to 1, in 0.01 increments
    thresholds = np.linspace(0, 1, 101)
    
    fpr_ = []
    tpr_ = []

    for thresh in thresholds:
        
        # binarize using the threshold, then compute true and false positives
        pred_df["y_pred"] = (pred_df["y_prob"] > thresh).astype(int)
        
        tp = len(pred_df.loc[(pred_df["y_pred"] == 1) & (pred_df["y_test"] == 1)])
        fp = len(pred_df.loc[(pred_df["y_pred"] == 1) & (pred_df["y_test"] == 0)])

        # Compute FPR and TPR. FPR = FP / N. TPR = TP / P
        fpr_.append(fp / num_sensitive)
        tpr_.append(tp / num_resistant)

    fpr_ = np.array(fpr_)
    tpr_ = np.array(tpr_)

    sens_spec_sum = (1 - fpr_) + tpr_

    # get index of highest sum(s) of sens and spec. Arbitrarily take the first threshold when there are multiple
    best_sens_spec_sum_idx = np.where(sens_spec_sum == np.max(sens_spec_sum))[0][0]
    select_thresh = thresholds[best_sens_spec_sum_idx]
    print(f"Binarization threshold: {select_thresh}")

    # return the predicted class labels
    return (pred_df["y_prob"] > select_thresh).astype(int).values


def run_regression_for_LRT(matrix, y, results_df, mutation=None, save_name="who_model.sav"):
    
    if mutation is None:
        X = matrix.values
    # remove the mutation from large_matrix and extract the values
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
        pickle.dump(open(os.path.join(analysis_dir, drug, "BINARY", save_name), 'rb'))
        
        chi_stat = np.nan
        pval = np.nan
        idx = "FULL"
    else:
        chi_stat = 2 * (results_df.loc["FULL", "log_like"] - log_like)
        pval = 1 - st.chi2.cdf(x=chi_stat, df=1)
        idx = mutation
     
    results_df.loc[idx, :] = [model.C_[0], log_like, chi_stat, pval,
                               sklearn.metrics.roc_auc_score(y_true=y, y_score=y_prob),
                               sklearn.metrics.accuracy_score(y_true=y, y_pred=y_pred),
                               sklearn.metrics.balanced_accuracy_score(y_true=y, y_pred=y_pred),
                              ]
    return results_df


# run regressions for the full models
print("Fitting full regressions...")
results_who = run_regression_for_LRT(who_large_matrix, who_y, results_who, mutation=None, save_name="who_model.sav")
results_all = run_regression_for_LRT(all_large_matrix, all_y, results_all, mutation=None, save_name="all_model.sav")

    
############# STEP 4: GET ALL MUTATIONS TO PERFORM THE LIKELIHOOD RATIO TEST FOR #############


# model_7 = tiers 1+2, phenos=WHO only, no synonymous, unpooled. model_15 = same, but phenos=ALL
rif_binary_results = pd.read_excel(f"results/BINARY/{drug}.xlsx", sheet_name=["Model_7", "Model_15"])

model_7 = rif_binary_results["Model_7"]
model_15 = rif_binary_results["Model_15"]
print(model_7.shape, model_15.shape)

# glpK_p.Val192fs has an OR_LB of 1.03
or_thresh = 1.1
lrt_mut7 = model_7.query("OR_LB > @or_thresh & BH_pval < 0.01 & confidence not in ['1) Assoc w R', '2) Assoc w R - Interim']")
lrt_mut15 = model_15.query("OR_LB > @or_thresh & BH_pval < 0.01 & confidence not in ['1) Assoc w R', '2) Assoc w R - Interim']")

lrt_mut7 = list(set(lrt_mut7["mutation"]).intersection(lrt_mut15["mutation"]).union(lrt_mut7["mutation"]))
lrt_mut15 = list(set(lrt_mut15["mutation"]) - set(lrt_mut7))
print(f"Performing LRT for {len(lrt_mut7) + len(lrt_mut15)} mutations with OR lower bounds above {or_thresh}!")


############# STEP 5: FIT L2-PENALIZED REGRESSIONS FOR ALL MODELS WITH 1 FEATURE REMOVED #############

    
for i, mutation in enumerate(lrt_mut7):
    results_who = run_regression_for_LRT(who_large_matrix, who_y, results_who, mutation)
    print(f"Finished {mutation}")
    
    if i == 0:
        print(results_who)
    
for i, mutation in enumerate(lrt_mut15):
    results_all = run_regression_for_LRT(all_large_matrix, all_y, results_all, mutation)
    print(f"Finished {mutation}")
    
    if i == 0:
        print(results_all)

results_who.to_csv(os.path.join(analysis_dir, drug, "BINARY", f"{drug_WHO_abbr}_LRT_WHO_results.csv"))
print(f"{len(results_who['penalty'].unique())} unique penalty terms in the WHO analysis")
results_all.to_csv(os.path.join(analysis_dir, drug, "BINARY", f"{drug_WHO_abbr}_LRT_ALL_results.csv"))
print(f"{len(results_all['penalty'].unique())} unique penalty terms in the ALL analysis")

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")