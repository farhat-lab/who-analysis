import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['figure.dpi'] = 150
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
from functools import reduce

import glob, os, yaml, subprocess, itertools, sparse, sys, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# eigenvec_df = pd.read_csv("../data/eigenvec_original_10PC.csv", index_col=[0])
# eigenvec_df = pd.read_csv("../data/eigenvec_1000PC.csv", index_col=[0])
eigenvec_df = pd.read_csv("data/eigenvec_100PC.csv", index_col=[0])
who_variants = pd.read_csv("analysis/who_confidence_2021.csv")
drug_gene_mapping = pd.read_csv("data/drug_gene_mapping.csv")
analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'

sys.path.append("utils")
from stats_utils import *



def get_log_like_from_model(matrix, y):
    
    X = scaler.fit_transform(matrix.values)
    
    model = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
                         cv=5,
                         penalty='l2',
                         max_iter=10000, 
                         multi_class='ovr',
                         scoring='neg_log_loss',
                         class_weight='balanced'
                        )

    model.fit(X, y)
    print(f"Regularization parameter: {model.C_[0]}")

    # get positive class probabilities and predicted classes after determining the binarization threshold
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = get_threshold_val_and_classes(y_prob, y, maximize="sens_spec", spec_thresh=0.9)

    # log-likelihood is the negative of the log-loss. Y_PRED MUST BE THE PROBABILITIES. set normalize=False to get sum of the log-likelihoods
    return -sklearn.metrics.log_loss(y_true=y, y_pred=y_prob, normalize=False)



def likelihood_ratio_test_tiers(drug, phenos_name):
    
    # make sure that both phenotypes are included
    if phenos_name == "ALL":
        pheno_category_lst = ["ALL", "WHO"]
    else:
        pheno_category_lst = ["WHO"]

    results_excel = pd.read_excel(f"results/NEW/{drug}.xlsx", sheet_name=None)

    if drug == "Pretomanid":
        # pick any other drug to get the keys for
        model_abbrev = pd.read_excel(f"results/NEW/Bedaquiline.xlsx", sheet_name=None).keys()
    else:
        model_abbrev = results_excel.keys()

    tier1_mutations = pd.read_csv(os.path.join(analysis_dir, drug, "BINARY", "tiers=1", f"phenos={phenos_name}", "mutations_lst.txt"), sep="\t", header=None)[0].values
    
    if not os.path.isfile(os.path.join(analysis_dir, drug, "BINARY", "tiers=1+2", f"phenos={phenos_name}", "mutations_lst.txt")):
        print("There are no significant Tier 2 mutations. Quitting this model...")
        exit()
        
    tier12_mutations = pd.read_csv(os.path.join(analysis_dir, drug, "BINARY", "tiers=1+2", f"phenos={phenos_name}", "mutations_lst.txt"), sep="\t", header=None)[0].values

    # check that Tier 1 mutations are a subset of Tier 1 and 2 mutations
    assert len(set(tier1_mutations) - set(tier12_mutations)) == 0
    tier2_mutations = list(set(tier12_mutations) - set(tier1_mutations))
    print(f"Tier 1: {len(tier1_mutations)} mutations\nTier 2: {len(tier2_mutations)} mutations")
    
    # read in only the genotypes files for the tiers for this model
    print("Reading in input data")
    df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv")).query("phenotypic_category in @pheno_category_lst")

    df_genos = pd.concat([pd.read_csv(os.path.join(analysis_dir, drug, f"genos_{num}.csv.gz"), compression="gzip") for num in ["1", "2"]], axis=0)
    df_genos["mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
    df_genos = df_genos.query("mutation in @tier12_mutations & sample_id in @df_phenos.sample_id.values").drop_duplicates()
    del df_genos["resolved_symbol"]
    del df_genos["variant_category"]

    drop_isolates = df_genos.query("variant_allele_frequency > 0.25 & variant_allele_frequency < 0.75").sample_id.unique()
    print(f"    Dropped {len(drop_isolates)} isolates with any intermediate AFs. Remainder are binary")
    df_genos = df_genos.query("sample_id not in @drop_isolates")    

    # check after this step that the only NaNs left are truly missing data --> NaN in variant_binary_status must also be NaN in variant_allele_frequency
    assert len(df_genos.loc[(~pd.isnull(df_genos["variant_allele_frequency"])) & (pd.isnull(df_genos["variant_binary_status"]))]) == 0

    # pivot and drop any rows with NaNs
    model_matrix = df_genos.pivot(index="sample_id", columns="mutation", values="variant_binary_status").dropna(how="any", axis=0)

    # there should not be any more NaNs
    assert sum(pd.isnull(np.unique(model_matrix.values))) == 0

    # remove any features that have no signal (should be nothing though because we kept only mutations that are significant)
    model_matrix = model_matrix.loc[:, model_matrix.nunique() > 1]

    # in this case, only 2 possible values -- 0 (ref), 1 (alt) because we already dropped NaNs
    assert len(np.unique(model_matrix.values)) <= 2

    # keep only samples (rows) that are in matrix and use loc with indices to ensure they are in the same order
    df_phenos = df_phenos.set_index("sample_id")
    df_phenos = df_phenos.loc[model_matrix.index]

    # check that the sample ordering is the same in the genotype and phenotype matrices
    assert sum(model_matrix.index != df_phenos.index) == 0

    tier1_matrix = model_matrix[tier1_mutations]
    tier12_matrix = model_matrix[tier12_mutations]
    y = df_phenos["phenotype"].values
    
    # get log-likelihoods of the two models
    tier1_log_like = get_log_like_from_model(tier1_matrix, y)
    tier12_log_like = get_log_like_from_model(tier12_matrix, y)

    # chi-statistic is the log-like of the bigger model minus the log-like of the smaller model 
    chi_stat = 2 * (tier12_log_like - tier1_log_like)
    dof = tier12_matrix.shape[1] - tier1_matrix.shape[1]
    assert dof == len(tier2_mutations)
    return st.chi2.sf(x=chi_stat, df=dof) 



_, drug, phenos_name = sys.argv

print(f"Performing LRT for {drug} on {phenos_name} phenotypes")
lrt_pval = likelihood_ratio_test_tiers(drug, phenos_name)
print(f"{lrt_pval}\n")