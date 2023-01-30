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


# starting the memory monitoring
tracemalloc.start()

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
    
# fit separate models for both the set of mutations that were significant in both Ridge AND LRT, or mutations that were significant in either
# AND_mutations is a subset of OR_mutations
OR_mutations = pd.read_csv(os.path.join(analysis_dir, drug, "BINARY/prediction_models/ridge_OR_LRT_variants.csv"))
AND_mutations = pd.read_csv(os.path.join(analysis_dir, drug, "BINARY/prediction_models/ridge_AND_LRT_variants.csv"))

if not os.path.isdir(os.path.join(analysis_dir, drug, f"BINARY/prediction_models/phenos={phenos_name}")):
    os.makedirs(os.path.join(analysis_dir, drug, f"BINARY/prediction_models/phenos={phenos_name}"))

OR_mutations["Tier"] = OR_mutations["Tier"].astype(str)
OR_mutations = OR_mutations.query("Tier in @tiers_lst").mutation.values
print(f"{len(OR_mutations)} Tier {'+'.join(tiers_lst)} mutations significant in Ridge or LRT")

AND_mutations["Tier"] = AND_mutations["Tier"].astype(str)
AND_mutations = AND_mutations.query("Tier in @tiers_lst").mutation.values
print(f"{len(AND_mutations)} Tier {'+'.join(tiers_lst)} mutations significant in Ridge and LRT")

    
############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############
    

# read in only the genotypes files for the tiers for this model
df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv")).query("phenotypic_category in @pheno_category_lst")
df_model = pd.concat([pd.read_csv(os.path.join(analysis_dir, drug, f"genos_{num}.csv.gz"), compression="gzip", low_memory=False) for num in tiers_lst])

# then keep only samples of the desired phenotype
df_model = df_model.loc[df_model["sample_id"].isin(df_phenos["sample_id"])]

# keep only the mutations of interest
df_model["mutation"] = df_model["resolved_symbol"] + "_" + df_model["variant_category"]


def prepare_model_inputs(df_genos, df_phenos, mutations_lst):

    df_model = df_genos.query("mutation in @mutations_lst")

    drop_isolates = df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] <= 0.75)].sample_id.unique()
    print(f"Dropped {len(drop_isolates)} isolates with any intermediate AFs. Remainder are binary")
    df_model = df_model.query("sample_id not in @drop_isolates")    

    df_model = df_model.dropna(subset="variant_binary_status", axis=0, how="any")
    df_model = df_model.sort_values("variant_binary_status", ascending=False).drop_duplicates(["sample_id", "mutation"], keep="first")
    matrix = df_model.pivot(index="sample_id", columns="mutation", values="variant_binary_status")
    del df_model

    matrix = matrix.dropna(axis=0, how="any")
    matrix = matrix[matrix.columns[~((matrix == 0).all())]]

    matrix.to_pickle(os.path.join(analysis_dir, drug, f"tiers={'+'.join(tiers_lst)}_phenos={phenos_name}_significant.pkl"))
    print(f"Fitting prediction model on {matrix.shape[0]} samples and {matrix.shape[1]} features")   

    if len(set(mutations_lst) - set(matrix.columns)) > 0:
        raise ValueError(f"Mutations {set(mutations_lst) - set(matrix.columns)} are not in the model matrix")

    # Read in the PC coordinates dataframe, then keep only the desired number of principal components
    eigenvec_df = pd.read_csv("../data/eigenvec_10PC.csv", index_col=[0]).iloc[:, :num_PCs]
    matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True, how="inner")

    df_phenos = df_phenos.set_index("sample_id").loc[matrix.index]
    assert sum(matrix.index != df_phenos.index.values) == 0
    
    return matrix, df_phenos


########################## STEP 2: FIT MODEL ##########################


scaler = StandardScaler()

def fit_model_save_results(matrix, df_phenos, prefix):

    X = scaler.fit_transform(matrix.values)
    y = df_phenos["phenotype"].values

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

    scores["test_spec"] = 2 * scores['test_balanced_accuracy'] - scores['test_recall']
    scores["test_sens"] = scores.pop("test_recall")

    res_df = pd.DataFrame.from_dict(scores)
    del res_df["fit_time"]
    del res_df["score_time"]
    res_df.to_csv(os.path.join(analysis_dir, drug, f"BINARY/prediction_models/phenos={phenos_name}/tiers={'+'.join(tiers_lst)}_{prefix}_CVresults.csv"), index=False)
    
    
AND_matrix, AND_phenos = prepare_model_inputs(df_model, df_phenos, AND_mutations)
fit_model_save_results(AND_matrix, AND_phenos, "AND")

del AND_matrix
del AND_phenos

OR_matrix, OR_phenos = prepare_model_inputs(df_model, df_phenos, OR_mutations)
fit_model_save_results(OR_matrix, OR_phenos, "OR")

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")