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
    os.makedirs(os.path.join(analysis_dir, drug, "BINARY/LRT"))
    

############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############


phenos_file = os.path.join(analysis_dir, drug, "phenos_binary.csv")
df_phenos = pd.read_csv(phenos_file).set_index("sample_id")

# different matrices, depending on the phenotypes
matrix = pd.read_pickle(os.path.join(analysis_dir, drug, "BINARY", f"tiers=1+2/phenos={phenos_name}", model_prefix, "model_matrix.pkl"))

# Read in the PC coordinates dataframe, then keep only the desired number of principal components
eigenvec_df = pd.read_csv("data/eigenvec_10PC.csv", index_col=[0]).iloc[:, :num_PCs]

# concatenate the eigenvectors to the matrix and check the index ordering against the phenotypes matrix
matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True)
df_phenos = df_phenos.loc[matrix.index]
assert sum(matrix.index != df_phenos.index.values) == 0
y = df_phenos["phenotype"].values
print(f"{matrix.shape[0]} samples and {matrix.shape[1]} variables in the largest {phenos_name} model")


############# STEP 2: READ IN THE ORIGINAL DATA: MODEL_MATRIX PICKLE FILE FOR A GIVEN MODEL #############

    
def remove_single_mut(matrix, mutation):
    
    if mutation not in matrix.columns:
        raise ValueError(f"{mutation} is not in the argument matrix!")
    
    small_matrix = matrix.loc[:, matrix.columns != mutation]
    assert small_matrix.shape[1] + 1 == matrix.shape[1]
    return small_matrix
    
    
    
def fit_logReg_known_penalty(matrix, y, num_bootstrap, reg_param):
    
    X = matrix.values
    X_small = remove_single_mut(matrix, mutation).values
    
    diff_stats = []
    
    for i in range(num_bootstrap):

        # randomly draw sample indices
        sample_idx = np.random.choice(np.arange(0, len(y)), size=len(y), replace=True)
        y_bs = y[sample_idx]

        null_model = LogisticRegression(C=reg_param, penalty='l2', max_iter=10000, multi_class='ovr', class_weight='balanced')
        alt_model = LogisticRegression(C=reg_param, penalty='l2', max_iter=10000, multi_class='ovr', class_weight='balanced')
        
        # fit original model on the bootstrap sample containing all features
        X_null = scaler.fit_transform(X[sample_idx, :])
        null_model.fit(X_null, y_bs)
        
        # fit the model with one feature dropped on the other model
        X_alt = scaler.fit_transform(X_small[sample_idx, :])
        alt_model.fit(X_alt, y_bs)
        
        null_stats = get_binary_metrics_from_model(null_model, X_null, y_bs)
        alt_stats = get_binary_metrics_from_model(alt_model, X_alt, y_bs)
        
        # null hypothesis is that the reference model 
        diff_stats.append(null_stats - alt_stats)
        assert len(diff_stats[i]) == len(null_stats)
        
    diff_stats_df = pd.DataFrame(diff_stats)
    diff_stats_df.columns = ["AUC", "Sens", "Spec", "accuracy"]
    return diff_stats_df
        
        

num_bootstrap = 100

# tier2_mutations = get_tier2_mutations_of_interest(analysis_dir, drug, phenos_name)
# mutations_lst = ["rpoB_p.Ser450Leu", 'Rv2752c_p.Asn30Ser', 'rpoC_p.Glu1092Asp', 'rpoC_p.Ile491Thr', 'rpoC_p.Asn698Ser', 'rpoC_p.Pro1040Arg']

mutations_lst = ['glpK_p.Leu152Arg', 'Rv2752c_p.Val396Gly', 'lpqB_p.Asp370Glu', 'rpoA_c.-316G>A', 'Rv1129c_p.Ser362Thr']
# mutations_lst = tier2_mutations
print(f"Bootstrapping {len(mutations_lst)} mutations with {num_bootstrap} replicates\n")
summary_stats_all = pd.DataFrame(columns = ["AUC", "Sens", "Spec", "accuracy"])

already_done = []
# already_done = ["rpoB_p.Ser450Leu", 'Rv2752c_p.Asn30Ser', 'rpoC_p.Glu1092Asp', 'rpoC_p.Ile491Thr', 'rpoC_p.Asn698Ser', 'rpoC_p.Pro1040Arg']

for i, mutation in enumerate(mutations_lst):
    
    if mutation not in already_done:
    
        print(f"Working on {mutation}")

        diff_stats_df = fit_logReg_known_penalty(matrix, y, num_bootstrap, 0.01)
        diff_stats_df["mutation"] = mutation
        summary_stats_all = pd.concat([summary_stats_all, diff_stats_df], axis=0)

    # if i % 25 == 0:
    #     summary_stats_all.to_csv(os.path.join(analysis_dir, drug, "BINARY/LRT", f"{phenos_name}phenos_bootstrap_StatsDiff.csv.gz"), 
    #                      compression="gzip",
    #                      index=False)
        
summary_stats_all.to_csv(os.path.join(analysis_dir, drug, "BINARY/LRT", f"{phenos_name}phenos_bootstrap_StatsDiff_2.csv.gz"), compression="gzip", index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")