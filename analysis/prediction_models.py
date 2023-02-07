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
num_bootstrap = kwargs["num_bootstrap"]
    
out_dir = os.path.join(analysis_dir, drug, f"BINARY/prediction_models")
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

df_results = pd.read_excel(f"results/BINARY/{drug}.xlsx", sheet_name=["Model_3", "Model_7", "Model_11", "Model_15"])
    
# # Tiers 1 + 2
# if len(tiers_lst) == 2:
#     sig_mutations = list(set(df_results["Model_3"].query("BH_pval < 0.05")["mutation"]).union(
#                              df_results["Model_7"].query("BH_pval < 0.01")["mutation"]).union(
#                              df_results["Model_11"].query("BH_pval < 0.05")["mutation"]).union(
#                              df_results["Model_15"].query("BH_pval < 0.01")["mutation"])
#                         )
# # Tier 1 only
# else:
#     sig_mutations = list(set(df_results["Model_3"].query("BH_pval < 0.05")["mutation"]).union(
#                              df_results["Model_11"].query("BH_pval < 0.05")["mutation"])
#                         )
# Tiers 1 + 2
if len(tiers_lst) == 2:
    sig_mutations = list(set(df_results["Model_3"].query("BH_pval < 0.05 & (OR_LB > 1 | OR_UB < 1)")["mutation"]).union(
                             df_results["Model_7"].query("BH_pval < 0.01 & (OR_LB > 1 | OR_UB < 1)")["mutation"]).union(
                             df_results["Model_11"].query("BH_pval < 0.05 & (OR_LB > 1 | OR_UB < 1)")["mutation"]).union(
                             df_results["Model_15"].query("BH_pval < 0.01 & (OR_LB > 1 | OR_UB < 1)")["mutation"])
                        )
# Tier 1 only
else:
    sig_mutations = list(set(df_results["Model_3"].query("BH_pval < 0.05 & (OR_LB > 1 | OR_UB < 1)")["mutation"]).union(
                             df_results["Model_11"].query("BH_pval < 0.05 & (OR_LB > 1 | OR_UB < 1)")["mutation"])
                        )


############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############
    

df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv"))
matrix_file = os.path.join(out_dir, f"tiers={'+'.join(tiers_lst)}.pkl")
    
# read in only the genotypes files for the tiers for this model
if not os.path.isfile(matrix_file):
    
    print(f"Creating input matrices for tiers={'+'.join(tiers_lst)} prediction models")
    df_genos = pd.concat([pd.read_csv(os.path.join(analysis_dir, drug, f"genos_{num}.csv.gz"), compression="gzip", low_memory=False) for num in tiers_lst])
    df_genos["mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
    
    # keep only the significant mutations and samples in the desired phenotypes category
    df_model = df_genos.query("mutation in @sig_mutations")

    drop_isolates = df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] <= 0.75)].sample_id.unique()
    print(f"Dropped {len(drop_isolates)} isolates with any intermediate AFs. Remainder are binary")
    df_model = df_model.query("sample_id not in @drop_isolates")    

    df_model = df_model.dropna(subset="variant_binary_status", axis=0, how="any")
    df_model = df_model.sort_values("variant_binary_status", ascending=False).drop_duplicates(["sample_id", "mutation"], keep="first")
    matrix = df_model.pivot(index="sample_id", columns="mutation", values="variant_binary_status")
    del df_model

    matrix = matrix.dropna(axis=0, how="any")
    matrix = matrix[matrix.columns[~((matrix == 0).all())]]
    matrix.to_pickle(matrix_file)

matrix = pd.read_pickle(matrix_file)
print(f"Fitting prediction model on {matrix.shape[0]} samples and {matrix.shape[1]} features")   

if len(set(sig_mutations) - set(matrix.columns)) > 0:
    raise ValueError(f"Mutations {set(sig_mutations) - set(matrix.columns)} are not in the model matrix")

# Read in the PC coordinates dataframe, then keep only the desired number of principal components
eigenvec_df = pd.read_csv("data/eigenvec_10PC.csv", index_col=[0]).iloc[:, :num_PCs]
matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True, how="inner")

df_phenos = df_phenos.set_index("sample_id").loc[matrix.index]
assert sum(matrix.index != df_phenos.index.values) == 0

# stratify train and test sets by the phenotypic category (WHO vs. ALL) and the phenotype
stratify_col = df_phenos["phenotypic_category"] + "_" + df_phenos["phenotype"].astype(str)
train_samples, test_samples = sklearn.model_selection.train_test_split(matrix.index.values, test_size=0.2, stratify=stratify_col)

X_train = scaler.fit_transform(matrix.loc[train_samples, :].values)
X_test = scaler.fit_transform(matrix.loc[test_samples, :].values)

y_train = df_phenos.loc[train_samples, :]["phenotype"].values
y_test = df_phenos.loc[test_samples, :]["phenotype"].values
print(f"Training set shape: {X_train.shape}, {len(y_train)}")
print(f"Testing set shape: {X_test.shape}, {len(y_test)}")


########################## STEP 2: FIT MODEL ##########################


scaler = StandardScaler()

# fit the full model
model = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
                             cv=5,
                             penalty='l2',
                             max_iter=10000, 
                             multi_class='ovr',
                             scoring='neg_log_loss',
                             class_weight='balanced'
                            )

model.fit(X_train, y_train)
print(f"Regularization parameter: {model.C_[0]}\n")

# add AUC, SENS, SPEC, and ACCURACY from the model trained on all WHO samples into the 
summary_stats_df = pd.DataFrame(columns=["AUC", "Sens", "Spec", "accuracy", "balanced_acc"])
summary_stats_df.loc[0, :] = get_binary_metrics_from_model(model, X_test, y_test, [0, 1, 2, 3, 4])
print(summary_stats_df)
summary_stats_df.to_csv(os.path.join(out_dir, f"tiers={'+'.join(tiers_lst)}_bootstrap_binary_stats.csv"), index=False)

print(f"Bootstrapping binary summary statistics with {num_bootstrap} replicates")
for i in range(num_bootstrap):

    # randomly draw sample indices
    sample_idx = np.random.choice(np.arange(0, len(y_train)), size=len(y_train), replace=True)

    # get the X and y matrices
    X_bs = scaler.fit_transform(X_train[sample_idx, :])
    y_bs = y_train[sample_idx]

    bs_model = LogisticRegression(C=model.C_[0], penalty='l2', max_iter=10000, multi_class='ovr', class_weight='balanced')
    bs_model.fit(X_bs, y_bs)
    summary_stats_df.loc[i+1, :] = get_binary_metrics_from_model(bs_model, X_test, y_test, [0, 1, 2, 3, 4])
    
    if i % 100 == 0:
        print(i)
        
summary_stats_df.to_csv(os.path.join(out_dir, f"tiers={'+'.join(tiers_lst)}_bootstrap_binary_stats.csv"), index=False)
#     cv_model = LogisticRegression(C=model.C_[0], 
#                                   penalty='l2', 
#                                   max_iter=10000, 
#                                   multi_class='ovr',
#                                   class_weight='balanced'
#                                  )

#     scores = cross_validate(cv_model, 
#                              X, 
#                              y, 
#                              cv=5,
#                              scoring=['roc_auc', 'recall', 'accuracy', 'balanced_accuracy'] # recall = sensitivity. balanced_accuracy = 1/2(sens + spec)
#                             )

#     scores["test_spec"] = 2 * scores['test_balanced_accuracy'] - scores['test_recall']
#     scores["test_sens"] = scores.pop("test_recall")

#     res_df = pd.DataFrame.from_dict(scores)
#     del res_df["fit_time"]
#     del res_df["score_time"]
#     res_df.to_csv(os.path.join(analysis_dir, drug, f"BINARY/prediction_models/phenos={phenos_name}/tiers={'+'.join(tiers_lst)}_{prefix}_CVresults.csv"), index=False)
    

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")