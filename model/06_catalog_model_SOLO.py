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

_, drug, AF_thresh, solo_col = sys.argv

solo_col = solo_col.upper()
assert solo_col in ['INITIAL', 'FINAL']

analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'
alpha = 0.05
AF_thresh = float(AF_thresh)
tiers_lst = ['1']

who_variants = pd.read_csv("results/WHO-catalog-V2.csv", header=[2])
del who_variants['mutation']
who_variants['tier'] = who_variants['tier'].astype(str)

who_variants = who_variants.query("tier in @tiers_lst").reset_index(drop=True)

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

who_variants = who_variants.loc[~pd.isnull(who_variants[f'{solo_col} CONFIDENCE GRADING'])].reset_index(drop=True)
R_assoc = who_variants.loc[who_variants[f'{solo_col} CONFIDENCE GRADING'].str.contains('Assoc w R')].query("drug == @drug")["variant"].values

if len(R_assoc) == 0:
    print("There are no significant R-associated mutations for this model\n")
    exit()


#################################################### STEP 1: GET GENOTYPES, CREATE LOF AND INFRAME FEATURES ####################################################
    
    
# read in only the genotypes files for the tiers for this model
df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv"))
df_genos = pd.concat([pd.read_csv(os.path.join(analysis_dir, drug, f"genos_{num}.csv.gz"), usecols=["sample_id", "resolved_symbol", "variant_category", "predicted_effect", "variant_allele_frequency", "variant_binary_status"], compression="gzip") for num in tiers_lst], axis=0)
df_genos["mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
del df_genos["variant_category"]

# set variants with AF <= threshold --> ABSENT, and AF > threshold = PRESENT
df_genos.loc[(df_genos["variant_allele_frequency"] <= AF_thresh), "variant_binary_status"] = 0
df_genos.loc[(df_genos["variant_allele_frequency"] > AF_thresh), "variant_binary_status"] = 1
del df_genos["variant_allele_frequency"]

lof_effect_list = ["frameshift", "start_lost", "stop_gained", "feature_ablation"]
# inframe_effect_list = ["inframe_insertion", "inframe_deletion"]

df_genos.loc[df_genos["predicted_effect"].isin(lof_effect_list), "pooled_variable"] = "LoF"
# df_genos.loc[df_genos["predicted_effect"].isin(inframe_effect_list), "pooled_variable"] = "inframe"

# get the pooled mutation column so that it's gene + inframe/LoF
df_genos["pooled_mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["pooled_variable"]

# variant_binary_status is NaN for missing variants (low-quality), so drop those samples
pooled_matrix = df_genos.sort_values(by=["variant_binary_status"], ascending=[False], na_position="last").drop_duplicates(subset=["sample_id", "pooled_mutation"], keep="first")

# keep only variants that are in the list of R-associated mutations
pooled_matrix = pooled_matrix.query("pooled_mutation in @R_assoc")
unpooled_matrix = df_genos.query("mutation in @R_assoc")

pooled_matrix = pooled_matrix.pivot(index="sample_id", columns="pooled_mutation", values="variant_binary_status")
unpooled_matrix = unpooled_matrix.drop_duplicates(["sample_id", "mutation"], keep=False).pivot(index="sample_id", columns="mutation", values="variant_binary_status")

# keep all isolates (i.e., no dropping due to NaNs)
model_matrix = pd.concat([pooled_matrix, unpooled_matrix], axis=1)
assert model_matrix.shape[1] == len(R_assoc)

# in this case, only 3 possible values -- 0 (ref), 1 (alt), and NaN
assert len(np.unique(model_matrix.values)) <= 3
print(f"Full matrix: {model_matrix.shape}, unique values: {np.unique(model_matrix.values)}")


#################################################### STEP 2: PERFORM CATALOG-BASED CLASSIFICATION USING R-ASSOCIATED MUTATIONS ONLY ####################################################


print(f"Performing catalog-based classification with {len(R_assoc)} tiers={'+'.join(tiers_lst)} R-associated mutations by SOLO {solo_col} at an AF > {AF_thresh}")    
print(R_assoc)

# can take the sum because variant_binary_status (the column being used) has been converted to binary everywhere
catalog_pred_df = pd.DataFrame(model_matrix[R_assoc].sum(axis=1)).reset_index()
catalog_pred_df.columns = ["sample_id", "y_pred"]
catalog_pred_df["y_pred"] = (catalog_pred_df["y_pred"] > 0).astype(int)

assert catalog_pred_df["y_pred"].nunique() == 2
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
    F1_ci = st.binomtest(k=2*TP, n=2*TP + FP + FN, p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
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

                               "F1": 2*TP / (2*TP + FP + FN),
                               "F1_lb": F1_ci.low,
                               "F1_ub": F1_ci.high,
                               
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
catalog_results["Model"] = f"SOLO {solo_col.lower().capitalize()}"
catalog_results.set_index("Model").to_csv(os.path.join(out_dir, f"model_stats_AF{int(AF_thresh*100)}_SOLO_{solo_col.lower()}_withLoF.csv"))

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")