import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm

import glob, os, yaml, subprocess, itertools, sparse, sys
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

eigenvec_df = pd.read_csv("data/eigenvec_original_10PC.csv", index_col=[0])
who_variants_combined = pd.read_csv("analysis/who_confidence_2021.csv")
analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'

import warnings
warnings.filterwarnings("ignore")

sys.path.append("utils")
from stats_utils import *



def get_pvals_no_PCA(drug, drug_WHO_abbr, model_matrix_path, binary=True, num_bootstrap=1000, alpha=0.05):
    
    if binary:
        folder = "BINARY"
        df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv"))
    else:
        folder = "MIC"
        df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_mic.csv"))
        df_phenos = process_multiple_MICs(df_phenos)

    model_matrix = pd.read_pickle(os.path.join(analysis_dir, drug, folder, model_matrix_path, "model_matrix.pkl"))
    print(f"Original matrix: {model_matrix.shape}")

    df_phenos = df_phenos.set_index("sample_id")
    df_phenos = df_phenos.loc[model_matrix.index]

    # check that the sample ordering is the same in the genotype and phenotype matrices
    assert sum(model_matrix.index != df_phenos.index) == 0
    X = model_matrix.values

    # scale inputs
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # binary vs. quant (MIC) phenotypes
    if binary:
        y = df_phenos["phenotype"].values
        assert len(np.unique(y)) == 2
    else:
        y = np.log2(df_phenos["mic_value"].values)

    if len(y) != X.shape[0]:
        raise ValueError(f"Shapes of model inputs {X.shape} and outputs {len(y)} are incompatible")
    print(f"{X.shape[0]} samples and {X.shape[1]} variables in the model")
    
    if binary:
        model = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
                                     cv=5,
                                     penalty='l2',
                                     max_iter=10000, 
                                     multi_class='ovr',
                                     scoring='neg_log_loss',
                                     class_weight='balanced'
                                    )


    else:
        model = RidgeCV(alphas=np.logspace(-6, 6, 13),
                        cv=5,
                        scoring='neg_root_mean_squared_error'
                       )
    model.fit(X, y)

    coef_df = pd.DataFrame({"mutation": model_matrix.columns, "coef": np.squeeze(model.coef_)})
    
    if num_bootstrap == 0:
        raise ValueError("Number of replicates must be positive for the permutation test!")
    else:
        print(f"Performing permutation test with {num_bootstrap} replicates")
        permute_df = perform_permutation_test(model, X, y, num_bootstrap, binary=binary, progress_bar=True)
        permute_df.columns = model_matrix.columns
        return get_coef_and_confidence_intervals(alpha, binary, who_variants_combined, drug_WHO_abbr, coef_df, permute_df=permute_df, bootstrap_df=None)      
    

# drug = "Bedaquiline"
# drug_WHO_abbr = "BDQ"
_, drug, drug_WHO_abbr = sys.argv

print(f"Fitting model with no PCs for {drug}")
no_PCA = get_pvals_no_PCA(drug, drug_WHO_abbr, "tiers=1+2/phenos=WHO/dropAF_noSyn_unpooled", binary=True, num_bootstrap=1000, alpha=0.05)
no_PCA.to_csv(f"{drug_WHO_abbr}_tiers=1+2_no_PCA.csv", index=False)