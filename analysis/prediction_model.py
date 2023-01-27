import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
import scipy.stats as st
import sklearn.metrics
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
import tracemalloc, pickle
from stats_utils import *


_, config_file, drug = sys.argv


kwargs = yaml.safe_load(open(config_file))

analysis_dir = kwargs["output_dir"]
num_PCs = kwargs["num_PCs"]
tiers_lst = kwargs["tiers_lst"]
pheno_category_lst = kwargs["pheno_category_lst"]

# make sure that both phenotypes are included
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
    pheno_category_lst = ["ALL", "WHO"]
else:
    phenos_name = "WHO"
    
mutations = pd.read_csv(os.path.join(analysis_dir, drug, "BINARY", "significant_tiers=1+2_variants.csv"))
mutations["Tier"] = mutations["Tier"].astype(str)
mutations = mutations.query("Tier in @tiers_lst").mutation.values

    
############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############
    
    
df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv")).query("phenotypic_category in @pheno_category_lst").set_index("sample_id")
print(f"Fitting model for {len(df_phenos)} {phenos_name} phenotypes on {len(mutations)} Tier {'+'.join(tiers_lst)} mutations!")

# use the model matrix from the unpooled model
matrix = pd.read_pickle(os.path.join(analysis_dir, drug, f"BINARY/tiers={'+'.join(tiers_lst)}/phenos={phenos_name}/dropAF_noSyn_unpooled/model_matrix.pkl"))

if len(set(mutations) - set(matrix.columns)) > 0:
    print(f"Mutations {set(mutations) - set(matrix.columns)} are not in the model matrix")
    mutations = list(set(mutations).intersection(matrix.columns))
    
matrix = matrix[mutations]

# Read in the PC coordinates dataframe, then keep only the desired number of principal components
eigenvec_df = pd.read_csv("data/eigenvec_10PC.csv", index_col=[0]).iloc[:, :num_PCs]
matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True, how="left")

df_phenos = df_phenos.loc[matrix.index]
assert sum(matrix.index != df_phenos.index.values) == 0


########################## STEP 2: FIT MODEL ##########################


scaler = StandardScaler()

X = scaler.fit_transform(matrix.values)
y = df_phenos["phenotype"].values

model_file = os.path.join(analysis_dir, drug, f"tiers={'+'.join(tiers_lst)}_phenos={phenos_name}_predict_model.sav")

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
    print(f"Regularization parameter: {model.C_[0]}\n")
    pickle.dump(model, open(model_file, "wb"))
else:
    model = pickle.load(open(model_file, "rb"))
    

cv_model = LogisticRegression(C=model.C_[0], 
                              penalty='l2', 
                              max_iter=10000, 
                              multi_class='ovr',
                              class_weight='balanced'
                             )

scores = cross_validate(cv_model, 
                         X, 
                         y, 
                         cv=5,
                         scoring=['roc_auc', 'recall', 'accuracy', 'balanced_accuracy'] # recall = sensitivity. balanced_accuracy = 1/2(sens + spec)
                        )

print(scores)


# # get positive class probabilities and predicted classes after determining the binarization threshold
# y_prob = model.predict_proba(X)[:, 1]
# y_pred = get_threshold_val_and_classes(y_prob, y)

# tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=y, y_pred=y_pred).ravel()

# print(f"AUC: {sklearn.metrics.roc_auc_score(y_true=y, y_score=y_prob)}")
# print(f"Sens: {tp / (tp + fn)}")
# print(f"Spec: {tn / (tn + fp)}")
# print(f"Acc: {sklearn.metrics.accuracy_score(y_true=y, y_pred=y_pred)}\n")