import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys, argparse, pickle, tracemalloc, warnings, shutil
import scipy.stats as st
import sklearn.metrics
warnings.filterwarnings("ignore")

analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'
who_variants = pd.read_csv("./results/WHO-catalog-V2.csv", header=[2])
del who_variants['mutation']
who_variants['tier'] = who_variants['tier'].astype(str)

alpha = 0.05
tiers_lst = ['1']
who_variants = who_variants.query("tier in @tiers_lst").reset_index(drop=True)
results_final = pd.read_csv("results/Regression_Final_June2024_Tier1.csv")

# utils files are in a separate folder
sys.path.append("utils")
from data_utils import *
from stats_utils import *


# starting the memory monitoring
tracemalloc.start()

parser = argparse.ArgumentParser()

# Add a required string argument for the config file
parser.add_argument("--drug", dest='drug', type=str, required=True)

parser.add_argument('--AF', dest='AF_thresh', type=float, default=0.75, help='Alternative allele frequency threshold (exclusive) to consider variants present')

parser.add_argument('--remove-mut', dest='remove_mut', action='store_true', help='Remove mutations in Groups 1-2 regression and Groups 4-5 SOLO that are major discrepancies')

parser.add_argument('--grading-rules', dest='grading_rules', action='store_true', help='Add mutations that would be upgraded by grading rules')

parser.add_argument('--S-assoc', dest='S_assoc', action='store_true', help='Predict susceptible isolates with S-assoc mutations')

cmd_line_args = parser.parse_args()
drug = cmd_line_args.drug
AF_thresh = cmd_line_args.AF_thresh
remove_mut = cmd_line_args.remove_mut
grading_rules = cmd_line_args.grading_rules
S_assoc = cmd_line_args.S_assoc

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

model_suffix = ''

if grading_rules:
    results_col = 'REGRESSION + GRADING RULES'
    model_suffix += '_gradingRules'
else:
    results_col = 'REGRESSION FINAL CONFIDENCE GRADING'

R_assoc = results_final.loc[results_final[results_col].str.contains('Assoc w R', case=True)].query("Drug==@drug")["mutation"].values

if remove_mut:

    remove_muts_lst = results_final.loc[(results_final[results_col].str.contains('Assoc w R', case=True)) & (results_final['SOLO FINAL CONFIDENCE GRADING'].str.contains('Not assoc w R', case=True))].query("Drug==@drug")["mutation"].values

    print(f"Removed major discrepancies {','.join(remove_muts_lst)} from the regression classification list")
    
    R_assoc = list(set(R_assoc) - set(remove_muts_lst))
    model_suffix += '_remove_discrepancies'

if S_assoc:
    
    # mutations that abrogate the effects of an R-associated mutation: only for BDQ, CFZ, AMK, and KAN. Checked that they have "Abrogates" in the Comment column
    negating_muts = who_variants.dropna(subset="Comment").query("drug==@drug & Comment.str.contains('Abrogates')").variant.values
    print(f"{len(negating_muts)} resistance-abrogating mutations for {drug}")

    model_suffix += '_R_abrogating_muts'
    
    if len(negating_muts) == 0:
        
        # copy the statistics for the model without R abrogating mutations
        shutil.copy(os.path.join(out_dir, f"model_stats_AF{int(AF_thresh*100)}_withLoF{model_suffix.replace('_R_abrogating_muts', '')}.csv"),
                    os.path.join(out_dir, f"model_stats_AF{int(AF_thresh*100)}_withLoF{model_suffix}.csv")
                   )
        
        # then exit
        exit()

else:
    negating_muts = []

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
df_genos.loc[df_genos["predicted_effect"].isin(lof_effect_list), "pooled_variable"] = "LoF"

# get the pooled mutation column so that it's gene + inframe/LoF
df_genos["pooled_mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["pooled_variable"]

# variant_binary_status is NaN for missing variants (low-quality), so drop those samples
pooled_matrix = df_genos.sort_values(by=["variant_binary_status"], ascending=[False], na_position="last").drop_duplicates(subset=["sample_id", "pooled_mutation"], keep="first")

# keep only variants that are in the list of R-associated mutations
# including variants that are components of pooled LoF variants 
pooled_matrix = pooled_matrix.query("pooled_mutation in @R_assoc | pooled_mutation in @negating_muts")
unpooled_matrix = df_genos.query("mutation in @R_assoc | mutation in @negating_muts")

del df_genos

pooled_matrix = pooled_matrix.pivot(index="sample_id", columns="pooled_mutation", values="variant_binary_status")
unpooled_matrix = unpooled_matrix.drop_duplicates(["sample_id", "mutation"], keep=False).pivot(index="sample_id", columns="mutation", values="variant_binary_status")

# keep all isolates (i.e., no dropping due to NaNs)
model_matrix = pd.concat([pooled_matrix, unpooled_matrix], axis=1)

# some variants are not present because they were not in the dataset -- these are the "Selection evidence" variants that had no data-driven results because they're not in any isolate
R_assoc = [variant for variant in R_assoc if variant in model_matrix.columns]
negating_muts = [variant for variant in negating_muts if variant in model_matrix.columns]
assert model_matrix.shape[1] == len(R_assoc) + len(negating_muts)

# in this case, only 3 possible values -- 0 (ref), 1 (alt), and NaN. Don't need to drop NaNs because we're looking for presence/absence of R-assoc variants
assert len(np.unique(model_matrix.values)) <= 3
print(f"Full matrix: {model_matrix.shape}, unique values: {np.unique(model_matrix.values)}")


#################################################### STEP 2: PERFORM CATALOG-BASED CLASSIFICATION USING R-ASSOCIATED MUTATIONS ONLY ####################################################


print(R_assoc)
print(f"Performing catalog-based classification with {len(R_assoc)} tiers={'+'.join(tiers_lst)} R-associated mutations at an AF > {AF_thresh}")

# can take the sum because variant_binary_status (the column being used) has been converted to binary everywhere
catalog_pred_df = pd.DataFrame(model_matrix[R_assoc].sum(axis=1)).reset_index()
catalog_pred_df.columns = ["sample_id", "y_pred"]
catalog_pred_df["y_pred"] = (catalog_pred_df["y_pred"] > 0).astype(int)

assert catalog_pred_df["y_pred"].nunique() == 2
catalog_pred_df = catalog_pred_df.merge(df_phenos[["sample_id", "phenotype"]], on="sample_id").drop_duplicates("sample_id")

negated_resistance = pd.DataFrame(model_matrix[negating_muts].sum(axis=1)).reset_index()
negated_resistance.columns = ['sample_id', 'susceptible']

# merge with the dataframe of negated resistance
catalog_pred_df = catalog_pred_df.merge(negated_resistance, on='sample_id')

# isolates where susceptible != 0 should all be predicted susceptible
catalog_pred_df.loc[catalog_pred_df['susceptible'] > 0, 'y_pred'] = 0
assert len(catalog_pred_df.query("susceptible > 0 & y_pred==1")) == 0


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
catalog_results["Model"] = "Regression"
catalog_results.set_index("Model").to_csv(os.path.join(out_dir, f"model_stats_AF{int(AF_thresh*100)}_withLoF{model_suffix}.csv"))

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")