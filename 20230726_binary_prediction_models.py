import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
import scipy.stats as st
import sklearn.metrics
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, KFold
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

_, config_file, drug = sys.argv

kwargs = yaml.safe_load(open(config_file))

analysis_dir = kwargs["output_dir"]
tiers_lst = kwargs["tiers_lst"]
pheno_category_lst = kwargs["pheno_category_lst"]
num_bootstrap = kwargs["num_bootstrap"]
amb_mode = kwargs["amb_mode"]

# make sure that both phenotypes are included
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
    pheno_category_lst = ["ALL", "WHO"]
else:
    phenos_name = "WHO"
    
out_dir = os.path.join(analysis_dir, drug, "BINARY", f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}")
print(f"Saving results to {out_dir}")
assert os.path.isdir(out_dir)
    
if drug == "Pretomanid":
    if phenos_name == "WHO":
        print("There are no WHO phenotypes for Pretomanid. Quitting this model...\n")
        exit()
    elif len(tiers_lst) == 2:
        print("There are no Tier 2 genes for Pretomanid. Quitting this model...\n")
        exit()
        
if amb_mode == "BINARY":
    model_prefix = "_binarized"
elif amb_mode == "AF":
    model_prefix = "_HET"
elif amb_mode == "DROP":
    model_prefix = ""
        
if os.path.isfile(os.path.join(out_dir, f"model_stats_{model_prefix}_bootstrap.csv")):
    print("Prediction models weree already fit\n")
    exit()

results_df = pd.read_csv(f"results/FINAL/{drug}.csv")
results_df["Tier"] = results_df["Tier"].astype(str)

if len(tiers_lst) == 2:
    if len(results_df.query("regression_confidence != 'Uncertain' & Tier == '2'")) == 0:
        print("There are no Tier 2 mutations that are not Uncertain.\n")
        exit()
    
mutations_lst = results_df.query("regression_confidence != 'Uncertain' & Tier in @tiers_lst & ~mutation.str.contains('|'.join(['lof', 'inframe']))")["mutation"].values
cat1_mutations = results_df.query("regression_confidence == 'Assoc w R - strict' & Tier in @tiers_lst & ~mutation.str.contains('|'.join(['lof', 'inframe']))")["mutation"].values

if len(mutations_lst) == 0:
    print("There are no significant mutations for this model\n")
    exit()
else:
    print(f"Fitting binary prediction models on {len(mutations_lst)} tiers={'+'.join(tiers_lst)} mutations for {phenos_name} phenotypes")
    
    
############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############
    
    
# read in only the genotypes files for the tiers for this model
df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv")).query("phenotypic_category in @pheno_category_lst")

df_genos = pd.concat([pd.read_csv(os.path.join(analysis_dir, drug, f"genos_{num}.csv.gz"), compression="gzip") for num in tiers_lst], axis=0)
df_genos["mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]

if len(cat1_mutations) > 0:

    print(f"Performing catalog-based classification with {len(cat1_mutations)} mutations")    
    pred_df = df_genos[["sample_id", "mutation", "variant_binary_status"]].merge(df_phenos, on="sample_id")
    pred_df["variant_binary_status"] = pred_df["variant_binary_status"].fillna(0)
    
    cat1_isolates = pred_df.loc[(pred_df["mutation"].isin(cat1_mutations)) & (pred_df["variant_binary_status"]==1)].sample_id.unique()
    pred_df.loc[pred_df["sample_id"].isin(cat1_isolates), "pred_phenotype"] = 1
    pred_df["pred_phenotype"] = pred_df["pred_phenotype"].fillna(0).astype(int)
    pred_df = pred_df.drop_duplicates("sample_id")

    y = pred_df["phenotype"].values
    y_pred = pred_df["pred_phenotype"].values
    
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=y, y_pred=y_pred).ravel()

    catalog_results = pd.DataFrame({"AUC": np.nan,
                                    "Sens": tp / (tp + fn),
                                    "Spec": tn / (tn + fp),
                                    "Precision": tp / (tp + fp),
                                    "Accuracy": sklearn.metrics.accuracy_score(y_true=y, y_pred=y_pred),
                                    "Balanced_Acc": sklearn.metrics.balanced_accuracy_score(y_true=y, y_pred=y_pred),
                                    "Model": "Catalog",
                                    "CV": 0
                                   }, index=[0])
    print(catalog_results)
else:
    catalog_results = pd.DataFrame(columns=["AUC", "Sens", "Spec", "Precision", "Accuracy", "Balanced_Acc", "Catalog", "CV"])
    
df_genos = df_genos.query("mutation in @mutations_lst & sample_id in @df_phenos.sample_id.values").drop_duplicates()
del df_genos["resolved_symbol"]
del df_genos["variant_category"]


# set variants with AF <= the threshold as wild-type and AF > the threshold as alternative
if amb_mode == "BINARY":
    print(f"Binarizing ambiguous variants with AF threshold of {AF_thresh}")
    df_genos.loc[(df_genos["variant_allele_frequency"] <= AF_thresh), "variant_binary_status"] = 0
    df_genos.loc[(df_genos["variant_allele_frequency"] > AF_thresh), "variant_binary_status"] = 1

# use ambiguous AF as the matrix value for variants with AF > 0.25. Below 0.25, the AF measurements aren't reliable
elif amb_mode == "AF":
    print("Encoding all variants with AF >= 0.25 with their AF")
    # encode all variants with AF > 0.25 with their AF
    df_genos.loc[df_genos["variant_allele_frequency"] >= 0.25, "variant_binary_status"] = df_genos.loc[df_genos["variant_allele_frequency"] >= 0.25, "variant_allele_frequency"].values

# drop all isolates with ambiguous variants with ANY AF below the threshold. DON'T DROP FEATURES BECAUSE MIGHT DROP SOMETHING RELEVANT
elif amb_mode == "DROP":    
    drop_isolates = df_genos.query("variant_allele_frequency > 0.25 & variant_allele_frequency < 0.75").sample_id.unique()
    print(f"    Dropped {len(drop_isolates)}/{len(df_genos.sample_id.unique())} isolates with any intermediate AFs. Remainder are binary")
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
    assert np.sort(np.unique(model_matrix.values))[1] >= 0.25
    print(np.sort(np.unique(model_matrix.values))[1], np.sort(np.unique(model_matrix.values))[-1])
    
# keep only samples (rows) that are in matrix and use loc with indices to ensure they are in the same order
df_phenos = df_phenos.set_index("sample_id")
df_phenos = df_phenos.loc[model_matrix.index]

# check that the sample ordering is the same in the genotype and phenotype matrices
assert sum(model_matrix.index != df_phenos.index) == 0

print(model_matrix.shape)
scaler = StandardScaler()
X = scaler.fit_transform(model_matrix.values)
print(f"{X.shape[0]} isolates and {X.shape[1]} features in the model")
y = df_phenos["phenotype"].values


########################## STEP 2: FIT MODEL ##########################


def bootstrap_binary_metrics(X, y, num_bootstrap=None):
    
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

    if drug in ["Clofazimine", "Delamanid", "Pretomanid"]:
        print("Setting specificity minimum at 0.9")
        spec_thresh = 0.9
    else:
        spec_thresh = None
        
    all_model_results = [pd.DataFrame(get_binary_metrics_from_model(model, X, y, spec_thresh=spec_thresh), index=[0])]
    print(all_model_results[0])
    
    if num_bootstrap is not None:
        
        print(f"Performing bootstrapping with {num_bootstrap} replicates")
        for i in range(num_bootstrap):

            bs_idx = np.random.choice(np.arange(len(y)), size=len(y), replace=True)

            X_bs = X[bs_idx, :]
            y_bs = y[bs_idx]

            model = LogisticRegression(C=reg_param, 
                                       penalty='l2',
                                       max_iter=10000, 
                                       multi_class='ovr',
                                       class_weight='balanced'
                                      )

            model.fit(X_bs, y_bs)        
            all_model_results.append(pd.DataFrame(get_binary_metrics_from_model(model, X_bs, y_bs, spec_thresh=spec_thresh), index=[0]))

            if i % int(num_bootstrap / 10) == 0:
                print(i)

    df_combined = pd.concat(all_model_results, axis=0).reset_index(drop=True)
    df_combined["CV"] = df_combined.index.values
    df_combined["Model"] = "L2_Reg"
    return df_combined


reg_results = bootstrap_binary_metrics(X, y, num_bootstrap=None)
pd.concat([catalog_results, reg_results], axis=0).to_csv(os.path.join(out_dir, f"model_stats{model_prefix}_bootstrap.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")