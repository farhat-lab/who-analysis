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
num_bootstrap = kwargs["num_bootstrap"]
amb_mode = kwargs["amb_mode"]
alpha = kwargs["alpha"]

if drug == "Pretomanid":
    phenos_name = "WHO"
else:
    phenos_name= "ALL"
    
out_dir = os.path.join(analysis_dir, drug, "BINARY", f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}")
print(f"Saving results to {out_dir}")
assert os.path.isdir(out_dir)

if amb_mode == "BINARY":
    model_prefix = "_binarized"
elif amb_mode == "AF":
    model_prefix = "_HET"
elif amb_mode == "DROP":
    model_prefix = ""
        
# if os.path.isfile(os.path.join(out_dir, f"model_stats_{model_prefix}_bootstrap.csv")):
#     print("Prediction models weree already fit\n")
#     exit()

results_df = pd.read_csv(f"results/FINAL/{drug}.csv")    
print(results_df.regression_confidence.unique())
mutations_lst = results_df.query("regression_confidence not in ['Neutral', 'Uncertain']")["mutation"].values
R_assoc = results_df.query("regression_confidence.str.contains('Assoc w R')")["mutation"].values

if len(R_assoc) == 0:
    print("There are no significant R-associated mutations for this model\n")
    exit()
else:
    print(f"Predicting binary {phenos_name} phenotypes from {len(mutations_lst)} tiers={'+'.join(tiers_lst)} mutations")
    print(mutations_lst)
    
    
#################################################### STEP 1: GET GENOTYPES, CREATE LOF AND INFRAME FEATURES ####################################################
    
    
# read in only the genotypes files for the tiers for this model
df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv"))
df_genos = pd.concat([pd.read_csv(os.path.join(analysis_dir, drug, f"genos_{num}.csv.gz"), compression="gzip") for num in tiers_lst], axis=0)
df_genos["mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]

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
    print(f"    Dropped {len(drop_isolates)}/{df_genos.sample_id.nunique()} isolates with any intermediate AFs. Remainder are binary")
    df_genos = df_genos.query("sample_id not in @drop_isolates")    
                
# check after this step that the only NaNs left are truly missing data --> NaN in variant_binary_status must also be NaN in variant_allele_frequency
assert len(df_genos.loc[(~pd.isnull(df_genos["variant_allele_frequency"])) & (pd.isnull(df_genos["variant_binary_status"]))]) == 0

lof_effect_list = ["frameshift", "start_lost", "stop_gained", "feature_ablation"]
inframe_effect_list = ["inframe_insertion", "inframe_deletion"]

df_genos.loc[df_genos["predicted_effect"].isin(lof_effect_list), "pooled_variable"] = "LoF"
df_genos.loc[df_genos["predicted_effect"].isin(inframe_effect_list), "pooled_variable"] = "inframe"

df_genos["pooled_mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["pooled_variable"]

# variant_binary_status is NaN for missing variants (low-quality), so drop those samples
pooled_matrix = df_genos.loc[~pd.isnull(df_genos["pooled_variable"])].sort_values(by=["variant_binary_status", "variant_allele_frequency"], ascending=False, na_position="last").drop_duplicates(subset=["sample_id", "pooled_mutation"], keep="first")

# keep only variants that are in the list of associated with R or S
pooled_matrix = pooled_matrix.query("pooled_mutation in @mutations_lst")
unpooled_matrix = df_genos.query("mutation in @mutations_lst")

pooled_matrix = pooled_matrix.pivot(index="sample_id", columns="pooled_mutation", values="variant_binary_status")
unpooled_matrix = unpooled_matrix.pivot(index="sample_id", columns="mutation", values="variant_binary_status")

model_matrix = pd.concat([pooled_matrix, unpooled_matrix], axis=1).dropna(how="any", axis=0)
assert model_matrix.shape[1] == len(mutations_lst)

# there should not be any more NaNs
assert sum(pd.isnull(np.unique(model_matrix.values))) == 0

# in this case, only 2 possible values -- 0 (ref), 1 (alt) because we already dropped NaNs
if amb_mode.upper() in ["BINARY", "DROP"]:
    assert len(np.unique(model_matrix.values)) <= 2
# the smallest value will be 0. Check that the second smallest value is greater than 0.25 (below this, AFs are not really reliable)
else:
    assert np.sort(np.unique(model_matrix.values))[1] >= 0.25
    print(np.sort(np.unique(model_matrix.values))[1], np.sort(np.unique(model_matrix.values))[-1])

print(model_matrix.shape)


#################################################### STEP 2: PERFORM CATALOG-BASED CLASSIFICATION USING R-ASSOCIATED MUTATIONS ONLY ####################################################


print(f"Performing catalog-based classification with {len(R_assoc)} R-associated mutations")    
catalog_pred_df = pd.DataFrame(model_matrix[R_assoc].sum(axis=1)).reset_index()
catalog_pred_df.columns = ["sample_id", "y_pred"]
catalog_pred_df["y_pred"] = (catalog_pred_df["y_pred"] > 0).astype(int)
catalog_pred_df = catalog_pred_df.merge(df_phenos[["sample_id", "phenotype"]], on="sample_id").drop_duplicates("sample_id")


def get_sens_spec_PPV_CI(df, pred_col, true_col):

    # make the TP, FP, TN, and FN columns
    df.loc[(df[pred_col]==1) & (df[true_col]==1), "Mut_R"] = 1
    df.loc[(df[pred_col]==1) & (df[true_col]==0), "Mut_S"] = 1
    df.loc[(df[pred_col]==0) & (df[true_col]==1), "NoMut_R"] = 1
    df.loc[(df[pred_col]==0) & (df[true_col]==0), "NoMut_S"] = 1
    
    df[["Mut_R", "Mut_S", "NoMut_S", "NoMut_R"]] = df[["Mut_R", "Mut_S", "NoMut_S", "NoMut_R"]].fillna(0).astype(int)
    
    assert len(np.unique(df[["Mut_R", "Mut_S", "NoMut_S", "NoMut_R"]].sum(axis=1))) == 1
    assert np.unique(df[["Mut_R", "Mut_S", "NoMut_S", "NoMut_R"]].sum(axis=1))[0] == 1
    
    # get the total numbers across the whole dataset
    Mut_R = df["Mut_R"].sum()
    Mut_S = df["Mut_S"].sum()
    NoMut_R = df["NoMut_R"].sum()
    NoMut_S = df["NoMut_S"].sum()
    
    results_df = pd.DataFrame({"Sens": Mut_R / (Mut_R + NoMut_R),
                               "Spec": NoMut_S / (NoMut_S + Mut_S),
                               "PPV":  Mut_R / (Mut_R + Mut_S),
                              }, index=["Test"])

    Sens_ci = st.binomtest(k=Mut_R, n=Mut_R + NoMut_R, p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
    Spec_ci = st.binomtest(k=NoMut_S, n=NoMut_S + Mut_S, p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
    PPV_ci = st.binomtest(k=Mut_R, n=Mut_R + Mut_S, p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
    
    results_df.loc["Lower", :] = [Sens_ci.low, Spec_ci.low, PPV_ci.low]
    results_df.loc["Upper", :] = [Sens_ci.high, Spec_ci.high, PPV_ci.high]
    
    assert sum(results_df.loc["Lower"] > results_df.loc["Test"]) == 0
    assert sum(results_df.loc["Upper"] < results_df.loc["Test"]) == 0
    return results_df
    


catalog_results = get_sens_spec_PPV_CI(catalog_pred_df, "y_pred", "phenotype")
catalog_results["Model"] = "Catalog"


#################################################### STEP 3: FIT REGRESSION MODELS USING ALL SIGNIFICANT MUTATIONS NOT UNCERTAIN OR NEUTRAL ####################################################


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

# get the classes after binarizing the probabilities
y_prob = model.predict_proba(X)[:, 1]
reg_pred_classes = get_threshold_val_and_classes(y_prob, y, spec_thresh=0.9)

reg_pred_df = df_phenos.reset_index()
reg_pred_df["y_pred"] = reg_pred_classes
reg_results = get_sens_spec_PPV_CI(reg_pred_df, "y_pred", "phenotype")
reg_results["Model"] = "Regression"

# need to perform bootstrapping to get confidence interval for AUC because it does not follow a binomial distribution



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

    all_model_results = [pd.DataFrame(get_binary_metrics_from_model(model, X, y, spec_thresh=0.9), index=[0])]
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