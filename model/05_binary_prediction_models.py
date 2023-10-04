import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys, pickle, tracemalloc, warnings
import scipy.stats as st
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
warnings.filterwarnings("ignore")

# utils files are in a separate folder
sys.path.append("utils")
from data_utils import *
from stats_utils import *


# starting the memory monitoring
tracemalloc.start()

_, config_file, drug, AF_thresh = sys.argv

kwargs = yaml.safe_load(open(config_file))

analysis_dir = kwargs["output_dir"]
tiers_lst = kwargs["tiers_lst"]
alpha = kwargs["alpha"]
AF_thresh = float(AF_thresh)

# AF_thresh needs to be a float between 0 and 1
if AF_thresh > 1:
    AF_thresh /= 100

if drug == "Pretomanid":
    phenos_name = "WHO"
else:
    phenos_name= "ALL"
    
out_dir = os.path.join(analysis_dir, drug, "BINARY", f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}")
print(f"Saving results to {out_dir}")
assert os.path.isdir(out_dir)

results_df = pd.read_csv(f"results/FINAL/{drug}.csv")   
print(results_df['FINAL CONFIDENCE GRADING'].unique())

mutations_lst = results_df.loc[~results_df['FINAL CONFIDENCE GRADING'].isin(['Neutral', 'Uncertain', 'Neutral - Interim'])]["mutation"].values
R_assoc = results_df.loc[results_df['FINAL CONFIDENCE GRADING'].str.contains('Assoc w R')]["mutation"].values

if len(R_assoc) == 0:
    print("There are no significant R-associated mutations for this model\n")
    exit()
else:
    print(f"Predicting binary {phenos_name} phenotypes from {len(mutations_lst)} tiers={'+'.join(tiers_lst)} mutations")
    print(mutations_lst)
    
    
#################################################### STEP 1: GET GENOTYPES, CREATE LOF AND INFRAME FEATURES ####################################################
    
    
# read in only the genotypes files for the tiers for this model
df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv"))
df_genos = pd.concat([pd.read_csv(os.path.join(analysis_dir, drug, f"genos_{num}.csv.gz"), usecols=["sample_id", "resolved_symbol", "variant_category", "predicted_effect", "variant_allele_frequency", "variant_binary_status"], compression="gzip") for num in tiers_lst], axis=0)
df_genos["mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
del df_genos["variant_category"]

# set variants with AF <= threshold --> ABSENT, and AF > threshold = PRESENT
df_genos.loc[(df_genos["variant_allele_frequency"] <= AF_thresh), "variant_binary_status"] = 0
df_genos.loc[(df_genos["variant_allele_frequency"] > AF_thresh), "variant_binary_status"] = 1

# then drop variants with intermediate AFs. In the HET case when AF_thresh = 0.25, nothing will be dropped, and these samples are considered as having that mutation present
drop_isolates = df_genos.query("variant_allele_frequency > 0.25 & variant_allele_frequency < @AF_thresh").sample_id.unique()
print(f"    Dropped {len(drop_isolates)}/{df_genos.sample_id.nunique()} isolates with any AFs in the range (0.25, {AF_thresh}). Remainder have been binarized")
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

# think it's just a data issue that sometimes there are duplicates. Just drop those samples
pooled_matrix = pooled_matrix.pivot(index="sample_id", columns="pooled_mutation", values="variant_binary_status")
unpooled_matrix = unpooled_matrix.drop_duplicates(["sample_id", "mutation"], keep=False).pivot(index="sample_id", columns="mutation", values="variant_binary_status")

model_matrix = pd.concat([pooled_matrix, unpooled_matrix], axis=1).dropna(how="any", axis=0)
assert model_matrix.shape[1] == len(mutations_lst)

# there should not be any more NaNs
assert sum(pd.isnull(np.unique(model_matrix.values))) == 0

# in this case, only 2 possible values -- 0 (ref), 1 (alt) because we already dropped NaNs
assert len(np.unique(model_matrix.values)) <= 2
print(model_matrix.shape)


#################################################### STEP 2: PERFORM CATALOG-BASED CLASSIFICATION USING R-ASSOCIATED MUTATIONS ONLY ####################################################


print(f"Performing catalog-based classification with {len(R_assoc)} R-associated mutations")    
catalog_pred_df = pd.DataFrame(model_matrix[R_assoc].sum(axis=1)).reset_index()
catalog_pred_df.columns = ["sample_id", "y_pred"]
catalog_pred_df["y_pred"] = (catalog_pred_df["y_pred"] > 0).astype(int)
catalog_pred_df = catalog_pred_df.merge(df_phenos[["sample_id", "phenotype"]], on="sample_id").drop_duplicates("sample_id")


def get_stats_with_CI(df, pred_col, true_col):

    # make the TP, FP, TN, and FN columns
    df.loc[(df[pred_col]==1) & (df[true_col]==1), "TP"] = 1
    df.loc[(df[pred_col]==1) & (df[true_col]==0), "FP"] = 1
    df.loc[(df[pred_col]==0) & (df[true_col]==1), "FN"] = 1
    df.loc[(df[pred_col]==0) & (df[true_col]==0), "TN"] = 1
    
    df[["TP", "FP", "FN", "TN"]] = df[["TP", "FP", "FN", "TN"]].fillna(0).astype(int)
    
    assert len(np.unique(df[["TP", "FP", "FN", "TN"]].sum(axis=1))) == 1
    assert np.unique(df[["TP", "FP", "FN", "TN"]].sum(axis=1))[0] == 1
    
    # get the total numbers across the whole dataset
    TP = df["TP"].sum()
    FP = df["FP"].sum()
    FN = df["FN"].sum()
    TN = df["TN"].sum()

    Sens_ci = st.binomtest(k=TP, n=TP + FN, p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
    Spec_ci = st.binomtest(k=TN, n=TN + FP, p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
    PPV_ci = st.binomtest(k=TP, n=TP + FP, p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
    NPV_ci = st.binomtest(k=TN, n=TN + FN, p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
    PropR_ci = st.binomtest(k=TP + FN, n=TP + FN + FP + TN, p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
    Accuracy_ci =  st.binomtest(k=TP + TN, n=TP + TN + FP + FN, p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
    
    # Balanced Accuracy
    balanced_acc_numerator = TP * (TN + FP) + TN * (TP + FN)
    balanced_acc_denominator = 2 * (TN + FP) * (TP + FN)    
    balanced_acc_ci = st.binomtest(k=balanced_acc_numerator, n=balanced_acc_denominator, p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')

    results_df = pd.DataFrame({"Sens": TP / (TP + FN),
                               "Sens_lb": Sens_ci.low,
                               "Sens_ub": Sens_ci.high,
                               
                               "Spec": TN / (TN + FP),
                               "Spec_lb": Spec_ci.low,
                               "Spec_ub": Spec_ci.high,
                               
                               "PPV":  TP / (TP + FP),
                               "PPV_lb": PPV_ci.low,
                               "PPV_ub": PPV_ci.high,
                               
                               "NPV": TN / (TN + FN),
                               "NPV_lb": NPV_ci.low,
                               "NPV_ub": NPV_ci.high,
                               
                               "PropR": (TP + FN) / (TP + FN + FP + TN),
                               "PropR_lb": PropR_ci.low,
                               "PropR_ub": PropR_ci.high,

                               "Accuracy": sklearn.metrics.accuracy_score(y_true=df[true_col], y_pred=df[pred_col]),
                               "Accuracy_lb": Accuracy_ci.low,
                               "Accuracy_ub": Accuracy_ci.high,
                               
                               "BalancedAcc": sklearn.metrics.balanced_accuracy_score(y_true=df[true_col], y_pred=df[pred_col]),
                               "BalancedAcc_lb": balanced_acc_ci.low,
                               "BalancedAcc_ub": balanced_acc_ci.high,

                               "TP": TP,
                               "TN": TN,
                               "FP": FP,
                               "FN": FN
                              }, index=[0])
    return results_df
    

catalog_results = get_stats_with_CI(catalog_pred_df, "y_pred", "phenotype")
catalog_results["Model"] = "Catalog"


#################################################### STEP 3: FIT REGRESSION MODELS USING ALL SIGNIFICANT MUTATIONS NOT UNCERTAIN OR NEUTRAL ####################################################


# keep only samples (rows) that are in matrix and use loc with indices to ensure they are in the same order
df_phenos = df_phenos.set_index("sample_id")
df_phenos = df_phenos.loc[model_matrix.index]

# check that the sample ordering is the same in the genotype and phenotype matrices
assert sum(model_matrix.index != df_phenos.index) == 0

scaler = StandardScaler()
X = scaler.fit_transform(model_matrix.values)
print(f"{X.shape[0]} isolates and {X.shape[1]} features in the regression model")
y = df_phenos["phenotype"].values

# keep L2 penalization because there is still collinearity, especially when including LoF mutations and the component mutations UNLESS THERE IS A SINGLE FEATURE (PRETOMANID)
if model_matrix.shape[1] == 1:
    print("Fitting unpenalized regression model") 
    model = LogisticRegression(max_iter=10000, 
                               multi_class='ovr',
                               class_weight='balanced'
                              )
    model.fit(X, y)
else:
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

pickle.dump(model, open(os.path.join(out_dir, "logReg_classification.sav"), "wb"))

# get the classes after binarizing the probabilities
y_prob = model.predict_proba(X)[:, 1]
reg_pred_classes = get_threshold_val_and_classes(y_prob, y, spec_thresh=0.9)

reg_pred_df = df_phenos.reset_index()
reg_pred_df["y_pred"] = reg_pred_classes
reg_results = get_stats_with_CI(reg_pred_df, "y_pred", "phenotype")

# add AUC to the regression results
def compute_ci_for_AUC(auc, y, alpha=0.05):
    
    # Compute standard error
    n1 = np.sum(y) # number of positives (resistant)
    n2 = len(y) - n1 # number of negatives (susceptible)
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    se_auc = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 - 1) * (q2 - auc ** 2)) / (n1 * n2))
    
    # two-tailed test, so 0.025 on each side for a 95% confidence interval
    z = st.norm.ppf(1-alpha/2)
    
    # Compute confidence interval
    lower_bound = auc - z * se_auc
    upper_bound = auc + z * se_auc
    
    return [lower_bound, upper_bound]


auc = sklearn.metrics.roc_auc_score(y_true=y, y_score=y_prob)
reg_results["AUC"] = auc

auc_ci = compute_ci_for_AUC(auc, y, alpha)
reg_results["AUC_lb"] = auc_ci[0]
reg_results["AUC_ub"] = auc_ci[1]
reg_results["Model"] = "Regression"
pd.concat([catalog_results, reg_results], axis=0).set_index("Model").to_csv(os.path.join(out_dir, f"model_stats_AF{int(AF_thresh*100)}.csv"))

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")