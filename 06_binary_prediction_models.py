import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
import scipy.stats as st
import sklearn.metrics
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
import tracemalloc, pickle, warnings
warnings.filterwarnings("ignore")

# utils files are in a separate folder
sys.path.append("utils")
from data_utils import *
from stats_utils import *


# starting the memory monitoring
tracemalloc.start()

_, config_file, drug, _, use_old = sys.argv

kwargs = yaml.safe_load(open(config_file))

analysis_dir = kwargs["output_dir"]
num_PCs = kwargs["num_PCs"]
tiers_lst = kwargs["tiers_lst"]
pheno_category_lst = kwargs["pheno_category_lst"]
model_prefix = kwargs["model_prefix"]
num_bootstrap = kwargs["num_bootstrap"]
amb_mode = kwargs["amb_mode"]

# make sure that both phenotypes are included
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
    pheno_category_lst = ["ALL", "WHO"]
else:
    phenos_name = "WHO"
    
out_dir = os.path.join(analysis_dir, drug, "BINARY", f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}", model_prefix)
print(f"Saving results to {out_dir}")
    
    
# if os.path.isfile(os.path.join(out_dir, "model_stats_with_bootstrap.csv")):
#     print("Binary model analysis was already performed for this model\n")
#     exit()
    

if use_old == "True":
    if phenos_name == "WHO":

        results_df = pd.read_excel(f"results/BINARY/{drug}.xlsx", sheet_name=["Model_3", "Model_7"])

        if len(tiers_lst) == 1:
            mutations_lst = results_df["Model_3"].query("BH_pval < 0.05")["mutation"].values
        else:
            mutations_lst = list(set(results_df["Model_3"].query("BH_pval < 0.05")["mutation"].values).union(set(results_df["Model_7"].query("BH_pval < 0.01")["mutation"].values)))
    else:

        results_df = pd.read_excel(f"results/BINARY/{drug}.xlsx", sheet_name=["Model_11", "Model_15"])

        if len(tiers_lst) == 1:
            mutations_lst = results_df["Model_11"].query("BH_pval < 0.05")["mutation"].values
        else:
            mutations_lst = list(set(results_df["Model_11"].query("BH_pval < 0.05")["mutation"].values).union(set(results_df["Model_15"].query("BH_pval < 0.01")["mutation"].values)))
else:
    model_1 = pd.read_csv(os.path.join(analysis_dir, drug, "BINARY", f"tiers=1/phenos={phenos_name}/dropAF_noSyn_unpooled", "model_analysis_new.csv"))

    if len(tiers_lst) == 1:
        mutations_lst = list(model_1.query("BH_pval < 0.05 & ~mutation.str.contains('PC')")["mutation"].values)
    else:
        model_2 = pd.read_csv(os.path.join(analysis_dir, drug, "BINARY", f"tiers=1+2/phenos={phenos_name}/dropAF_noSyn_unpooled", "model_analysis_new.csv"))
        mutations_lst = list(set(model_1.query("BH_pval < 0.05 & ~mutation.str.contains('PC')")["mutation"].values).union(set(model_2.query("BH_pval < 0.01 & ~mutation.str.contains('PC')")["mutation"].values)))

print(f"Fitting predictive model on {len(mutations_lst)} significant mutations from {phenos_name}, tiers {'+'.join(tiers_lst)}")
    
    
############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############
    
    
# read in only the genotypes files for the tiers for this model
df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv")).query("phenotypic_category in @pheno_category_lst")

df_genos = pd.concat([pd.read_csv(os.path.join(analysis_dir, drug, f"genos_{num}.csv.gz"), compression="gzip") for num in tiers_lst], axis=0)
df_genos["mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
df_genos = df_genos.query("mutation in @mutations_lst & sample_id in @df_phenos.sample_id.values").drop_duplicates()
del df_genos["resolved_symbol"]
del df_genos["variant_category"]


# set variants with AF <= the threshold as wild-type and AF > the threshold as alternative
if amb_mode == "BINARY":
    print(f"Binarizing ambiguous variants with AF threshold of {AF_thresh}")
    df_genos.loc[(pd.isnull(df_genos["variant_binary_status"])) & (df_genos["variant_allele_frequency"] <= AF_thresh), "variant_binary_status"] = 0
    df_genos.loc[(pd.isnull(df_genos["variant_binary_status"])) & (df_genos["variant_allele_frequency"] > AF_thresh), "variant_binary_status"] = 1

# use ambiguous AF as the matrix value for variants with AF > 0.25. Below 0.25, the AF measurements aren't reliable
elif amb_mode == "AF":
    print("Encoding ambiguous variants with their AF")
    # encode all variants with AF > 0.25 with their AF
    df_genos.loc[df_genos["variant_allele_frequency"] > 0.25, "variant_binary_status"] = df_genos.loc[df_genos["variant_allele_frequency"] > 0.25, "variant_allele_frequency"].values
   
# drop all isolates with ambiguous variants with ANY AF below the threshold. DON'T DROP FEATURES BECAUSE MIGHT DROP SOMETHING RELEVANT
elif amb_mode == "DROP":    
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
if amb_mode.upper() in ["BINARY", "DROP"]:
    assert len(np.unique(model_matrix.values)) <= 2
# the smallest value will be 0. Check that the second smallest value is greater than 0.25 (below this, AFs are not really reliable)
else:
    assert np.sort(np.unique(model_matrix.values))[1] > 0.25
    
# keep only samples (rows) that are in matrix and use loc with indices to ensure they are in the same order
df_phenos = df_phenos.set_index("sample_id")
df_phenos = df_phenos.loc[model_matrix.index]

# check that the sample ordering is the same in the genotype and phenotype matrices
assert sum(model_matrix.index != df_phenos.index) == 0
X = model_matrix.values
print(f"{X.shape[0]} isolates and {X.shape[1]} features in the model")


########################## STEP 2: FIT MODEL ##########################


scaler = StandardScaler()

X = scaler.fit_transform(model_matrix.values)
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
reg_param = model.C_[0]
print(f"Regularization parameter: {reg_param}")    

# create dataframe for the results
results_df = pd.DataFrame(columns=["AUC", "Sens", "Spec", "accuracy", "balanced_acc"])

# AUC, Sens, Spec, Acc, Balanced_Acc
results_df.loc["FULL", :] = get_binary_metrics_from_model(model, X, y, [0, 1, 2, 3, 4])
results_df.to_csv(os.path.join(out_dir, "model_stats_with_bootstrap.csv"))

# print(f"Bootstrapping binary summary statistics with {num_bootstrap} samples\n")
# for i in range(num_bootstrap):
    
#     # randomly draw sample indices
#     sample_idx = np.random.choice(np.arange(0, len(y)), size=len(y), replace=True)

#     # get the X and y matrices
#     X_bs = scaler.fit_transform(X[sample_idx, :])
#     y_bs = y[sample_idx]
    
#     bs_model = LogisticRegression(C=reg_param, penalty='l2', max_iter=10000, multi_class='ovr', class_weight='balanced')
#     bs_model.fit(X_bs, y_bs)
    
#     # add binary summary stats to the dataframe
#     results_df.loc[i, :] = get_binary_metrics_from_model(bs_model, X_bs, y_bs, [0, 1, 2, 3, 4])

    
# results_df.to_csv(os.path.join(out_dir, "model_stats_with_bootstrap.csv"))

################################# CROSS-VALIDATION PROCEDURE FOR COMPARING MODEL PERFORMANCES #################################

# cv_model = LogisticRegression(C=model.C_[0], 
#                               penalty='l2', 
#                               max_iter=10000, 
#                               multi_class='ovr',
#                               class_weight='balanced'
#                              )

# scores = cross_validate(cv_model, 
#                          X, 
#                          y, 
#                          cv=5,
#                          scoring=['roc_auc', 'recall', 'accuracy', 'balanced_accuracy'] # recall = sensitivity. balanced_accuracy = 1/2(sens + spec)
#                         )

# scores["test_spec"] = 2 * scores['test_balanced_accuracy'] - scores['test_recall']
# scores["test_sens"] = scores.pop("test_recall")

# res_df = pd.DataFrame.from_dict(scores)
# del res_df["fit_time"]
# del res_df["score_time"]
# print(res_df)

# res_df.to_csv(os.path.join(analysis_dir, drug, f"tiers={'+'.join(tiers_lst)}_phenos={phenos_name}_CV_results.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")