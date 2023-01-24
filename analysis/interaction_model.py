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
matrix = pd.read_pickle(os.path.join(analysis_dir, drug, f"BINARY/interaction/model_matrix_{phenos_name}_LRT05_sig.pkl"))
print(matrix.shape)

# Read in the PC coordinates dataframe, then keep only the desired number of principal components
eigenvec_df = pd.read_csv("data/eigenvec_10PC.csv", index_col=[0]).iloc[:, :num_PCs]
matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True)

df_phenos = df_phenos.loc[matrix.index]
assert sum(matrix.index != df_phenos.index.values) == 0

X = scaler.fit_transform(matrix.values)
y = df_phenos["phenotype"].values

############# STEP 2: FIT L2-PENALIZED REGRESSION FOR THE ORIGINAL MODEL #############


print(f"{matrix.shape[0]} samples and {matrix.shape[1]} variables in the {phenos_name} model")
model_file = os.path.join(analysis_dir, drug, f"BINARY/interaction/{phenos_name}_model.sav")

if not os.path.isfile(model_file):

    model = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
                                  cv=5,
                                  penalty='l2',
                                  max_iter=10000, 
                                  multi_class='ovr',
                                  scoring='neg_log_loss',
                                  class_weight='balanced'
                                 )
    model.fit(X, y)

    # get positive class probabilities and predicted classes after determining the binarization threshold
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = get_threshold_val_and_classes(y_prob, y)

    print([model.C_[0], 
            sklearn.metrics.roc_auc_score(y_true=y, y_score=y_prob),
            sklearn.metrics.accuracy_score(y_true=y, y_pred=y_pred),
            sklearn.metrics.balanced_accuracy_score(y_true=y, y_pred=y_pred),
           ])

    pickle.dump(model, open(model_file, "wb"))
    pd.DataFrame({"mutation": matrix.columns, "coef": np.squeeze(model.coef_)}).to_csv(os.path.join(analysis_dir, drug, f"BINARY/interaction/{phenos_name}_LRT05_sig_coef.csv"), index=False)
    
else:
    model = pickle.load(open(model_file, "rb"))


############# STEP 3: PERFORM BOOTSTRAPPING #############


# bootstrap both the coefficients and the summary stats
num_bootstrap = 100
coef_df, summary_stats_df = perform_bootstrapping(model, X, y, num_bootstrap, binary=True, save_summary_stats=True)
coef_df.columns = matrix.columns
coef_df.to_csv(os.path.join(analysis_dir, drug, f"BINARY/interaction/{phenos_name}_LRT05_sig_BScoef.csv"), index=False)
             
summary_stats_df.to_csv(os.path.join(analysis_dir, drug, f"BINARY/interaction/{phenos_name}_LRT05_sig_BSsummaryStats.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")