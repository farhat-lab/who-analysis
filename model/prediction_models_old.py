import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
import scipy.stats as st
import sklearn.metrics
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, KFold
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

# make sure that both phenotypes are included
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
    pheno_category_lst = ["ALL", "WHO"]
else:
    phenos_name = "WHO"
    
out_dir = os.path.join(analysis_dir, drug, "BINARY", f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}")
print(f"Saving results to {out_dir}")

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
    
if drug == "Pretomanid":
    if phenos_name == "ALL":
        print("There are no ALL phenotypes for Pretomanid\n")
        exit()
    elif len(tiers_lst) == 2:
        print("There are no Tier 2 genes for Pretomanid\n")
        exit()


def get_genos_phenos(analysis_dir, drug, pheno_category_lst):
    '''
    This function gets annotations (predicted effect and position) for mutations to merge them into the final analysis dataframes
    '''

    # shouldn't be any duplicates, but just to be safe
    df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv")).query("phenotypic_category in @pheno_category_lst").drop_duplicates()
    
    # drop duplicates and NaNs (NaNs not relevant here because we only want presence vs. absence)
    df_genos = pd.read_csv(os.path.join(analysis_dir, drug, "genos_1.csv.gz"), compression="gzip", low_memory=False, usecols=["sample_id", "resolved_symbol", "variant_category", "predicted_effect", "variant_binary_status"]).drop_duplicates().dropna()

    df_phenos = df_phenos.query("sample_id in @df_genos.sample_id.values")
    df_genos = df_genos.query("sample_id in @df_phenos.sample_id.values")

    # add another column for pooled mutations to use later
    df_genos.loc[df_genos["predicted_effect"].isin(["frameshift", "start_lost", "stop_gained", "feature_ablation"]), "pooled_mutation"] = df_genos["resolved_symbol"] + "_lof"
    df_genos.loc[df_genos["predicted_effect"].str.contains("inframe"), "pooled_mutation"] = df_genos["resolved_symbol"] + "_inframe"

    # get annotations for mutations to combine later. Exclude lof and inframe, these will be manually replaced later
    df_genos["mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
    return df_genos[["sample_id", "mutation", "pooled_mutation", "variant_binary_status", "predicted_effect"]], df_phenos

    


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

    
    
def compute_ci_for_binary_stats(TP, FP, TN, FN, AUC, y, alpha=0.05):

    ci_dict = {}
    
    # AUC
    ci_dict["AUC"] = compute_ci_for_AUC(AUC, y, alpha=alpha)
    
    # Sensitivity: TP / (TP + FN)
    ci = st.binomtest(k=TP, n=TP + FN, p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
    ci_dict["Sens"] = [ci.low, ci.high]
    
    # Specificity: TN / (TN + FP)
    ci = st.binomtest(k=TN, n=TN + FP, p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
    ci_dict["Spec"] = [ci.low, ci.high]

    # Precision = PPV: TP / (TP + FP)
    ci = st.binomtest(k=TP, n=TP + FP, p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
    ci_dict["Precision"] = [ci.low, ci.high]

    # Accuracy
    ci = st.binomtest(k=TP + TN, n=TP + TN + FP + FN, p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
    ci_dict["Accuracy"] = [ci.low, ci.high]
    
    # Balanced Accuracy
    balanced_acc_numerator = TP * (TN + FP) + TN * (TP + FN)
    balanced_acc_denominator = 2 * (TN + FP) * (TP + FN)    
    
    ci = st.binomtest(k=balanced_acc_numerator, n=balanced_acc_denominator, p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
    ci_dict["Balanced_Acc"] = [ci.low, ci.high]

    return ci_dict


    
def compute_mutation_catalog_stats(df_genos, df_phenos, mutation_lst):

    pred_R = list(set(df_genos.query("(mutation in @mutation_lst | pooled_mutation in @mutation_lst) & variant_binary_status == 1")["sample_id"]))

    df_phenos.loc[df_phenos["sample_id"].isin(pred_R), "prediction"] = 1
    df_phenos.loc[~df_phenos["sample_id"].isin(pred_R), "prediction"] = 0
    print(f"{len(df_phenos)}, {len(mutation_lst)}")

    tp = len(df_phenos.query("prediction == 1 & phenotype == 1"))
    fp = len(df_phenos.query("prediction == 1 & phenotype == 0"))
    tn = len(df_phenos.query("prediction == 0 & phenotype == 0"))
    fn = len(df_phenos.query("prediction == 0 & phenotype == 1"))

    AUC = sklearn.metrics.roc_auc_score(y_true=df_phenos["phenotype"], y_score=df_phenos["prediction"])
    
    results = np.array([AUC, # AUC
                        tp / (tp + fn), # SENSITIVITY
                        tn / (tn + fp), # SPECIFICITY,
                        tp / (tp + fp), # PRECISION / PPV
                        sklearn.metrics.accuracy_score(y_true=df_phenos["phenotype"], y_pred=df_phenos["prediction"]), # ACCURACY
                        sklearn.metrics.balanced_accuracy_score(y_true=df_phenos["phenotype"], y_pred=df_phenos["prediction"]), # BALANCED ACCURACY
                       ])
        
    res_dict = dict(zip(["AUC", "Sens", "Spec", "Precision", "Accuracy", "Balanced_Acc"], results))
    ci_dict = compute_ci_for_binary_stats(tp, fp, tn, fn, AUC, df_phenos["phenotype"].values, alpha=0.05)

    results_df = pd.concat([pd.DataFrame(res_dict, index=["value"]), pd.DataFrame(ci_dict, index=["lower", "upper"])], axis=0)

    # check that the lower bound is less than or equal to the sample value and the upper bound is greater than or equal to the sample value
    assert sum(results_df.loc["lower"] > results_df.loc["value"]) == 0
    assert sum(results_df.loc["upper"] < results_df.loc["value"]) == 0
    return results_df.loc[["lower", "value", "upper"]]    


    
def compute_logReg_model_stats(df_genos, df_phenos, mutation_lst):

    # drop samples with NaNs (they are NaN because when creating df_genos, we dropped all samples with NaNs in the variant_binary_status column)
    unpooled_matrix = df_genos.query("mutation in @mutation_lst").pivot(index="sample_id", columns="mutation", values="variant_binary_status").dropna(axis=0)
    
    # preferentially keep variant_binary_status = 1 because of cases where one lof mutation is present and another is absent. The overall feature = present in that case
    pooled_matrix = df_genos.query("pooled_mutation in @mutation_lst").sort_values("variant_binary_status", ascending=False).drop_duplicates(["sample_id", "pooled_mutation"], keep="first").pivot(index="sample_id", columns="pooled_mutation", values="variant_binary_status")

    # make sure that there are samples. If not, the merged dataframe will be empty
    if len(unpooled_matrix) > 0:
        if len(pooled_matrix) > 0:
            matrix = unpooled_matrix.merge(pooled_matrix, left_index=True, right_index=True)
        else:
            matrix = unpooled_matrix
    else:
        if len(pooled_matrix) > 0:
            matrix = pooled_matrix
        else:
            print("There are no variants for this model. Quitting...")
            return None

    print(f"Dropped {matrix.loc[:, matrix.nunique() == 1].shape[1]} variants with no signal")
    matrix = matrix.loc[:, matrix.nunique() > 1]
    print(matrix.shape)
        
    df_phenos = df_phenos.set_index("sample_id").loc[matrix.index.values]
    assert sum(df_phenos.index.values != matrix.index.values) == 0
    X = matrix.values
    y = df_phenos["phenotype"].values

    model = LogisticRegression(fit_intercept=True)
    model.fit(X, y)
    pickle.dump(model, open(os.path.join(out_dir, "logReg_classification.sav"), "wb"))

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = get_threshold_val_and_classes(y_prob, y, spec_thresh=0.9)
    df_pred = pd.DataFrame({"y_true": y, "y_pred": y_pred})
    AUC = sklearn.metrics.roc_auc_score(y_true=y, y_score=y_prob)

    TP = len(df_pred.query("y_pred == 1 & y_true == 1"))
    FP = len(df_pred.query("y_pred == 1 & y_true == 0"))
    TN = len(df_pred.query("y_pred == 0 & y_true == 0"))
    FN = len(df_pred.query("y_pred == 0 & y_true == 1"))
    assert TP + FP + TN + FN == len(df_pred)

    res_dict = get_binary_metrics_from_model(model, X, y, spec_thresh=0.9)
    ci_dict = compute_ci_for_binary_stats(TP, FP, TN, FN, AUC, y, alpha=0.05)

    results_df = pd.concat([pd.DataFrame(res_dict, index=["value"]), pd.DataFrame(ci_dict, index=["lower", "upper"])], axis=0)

    # check that the lower bound is less than or equal to the sample value and the upper bound is greater than or equal to the sample value
    assert sum(results_df.loc["lower"] > results_df.loc["value"]) == 0
    assert sum(results_df.loc["upper"] < results_df.loc["value"]) == 0
    return results_df.loc[["lower", "value", "upper"]]




df = pd.read_csv(f"results/FINAL/{drug}.csv")

# if os.path.isfile(os.path.join(out_dir, "catalog_results.csv")) and os.path.isfile(os.path.join(out_dir, "logReg_results.csv")):
#     print("Both the catalog and logistic regression classifier were already fit for this model")
#     exit()

# get dataframes of genotypes and phenotypes
print("Getting genotype and phenotype dataframes")
df_genos, df_phenos = get_genos_phenos(analysis_dir, drug, pheno_category_lst)

catalog_mutation_lst = df.query("regression_confidence.str.contains('Assoc w R')")["mutation"].values
# logReg_mutation_lst = df.query("regression_confidence not in ['Uncertain', 'Neutral']")["mutation"].values

if len(catalog_mutation_lst) > 0:
    print(f"Building catalog classifier with {len(catalog_mutation_lst)} mutations")
    catalog_results = compute_mutation_catalog_stats(df_genos, df_phenos, catalog_mutation_lst)
    catalog_results.to_csv(os.path.join(out_dir, "catalog_results.csv"))


if os.path.isfile(os.path.join(out_dir, "logReg_results.csv")):
    os.remove(os.path.join(out_dir, "logReg_results.csv"))
    os.remove(os.path.join(out_dir, "logReg_classification.sav"))
# if len(logReg_mutation_lst) > 0:
#     print(f"Building logistic regression classifier with {len(logReg_mutation_lst)} mutations")
#     logReg_results = compute_logReg_model_stats(df_genos, df_phenos, logReg_mutation_lst)

#     if logReg_results is not None:
#         logReg_results.to_csv(os.path.join(out_dir, "logReg_results.csv"))
    

########################## STEP 2: FIT MODEL ##########################


# def bootstrap_binary_metrics(X, y, num_bootstrap=None):
    
#     model = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
#                              cv=5,
#                              penalty='l2',
#                              max_iter=10000, 
#                              multi_class='ovr',
#                              scoring='neg_log_loss',
#                              class_weight='balanced'
#                             )

#     model.fit(X, y)
#     reg_param = model.C_[0]
#     print(f"Regularization parameter: {reg_param}") 

#     if drug in ["Clofazimine", "Delamanid", "Pretomanid"]:
#         print("Setting specificity minimum at 0.9")
#         spec_thresh = 0.9
#     else:
#         spec_thresh = None
        
#     all_model_results = [pd.DataFrame(get_binary_metrics_from_model(model, X, y, maximize="sens_spec", spec_thresh=spec_thresh), index=[0])]
#     print(all_model_results[0])
    
#     if num_bootstrap is not None:
        
#         print(f"Performing bootstrapping with {num_bootstrap} replicates")
#         for i in range(num_bootstrap):

#             bs_idx = np.random.choice(np.arange(len(y)), size=len(y), replace=True)

#             X_bs = X[bs_idx, :]
#             y_bs = y[bs_idx]

#             model = LogisticRegression(C=reg_param, 
#                                        penalty='l2',
#                                        max_iter=10000, 
#                                        multi_class='ovr',
#                                        class_weight='balanced'
#                                       )

#             model.fit(X_bs, y_bs)        
#             all_model_results.append(pd.DataFrame(get_binary_metrics_from_model(model, X_bs, y_bs, maximize="sens_spec", spec_thresh=spec_thresh), index=[0]))

#             if i % int(num_bootstrap / 10) == 0:
#                 print(i)

#     df_combined = pd.concat(all_model_results, axis=0).reset_index(drop=True)
#     df_combined["CV"] = df_combined.index.values
#     return df_combined


# results_df = bootstrap_binary_metrics(X, y, num_bootstrap)
# results_df.to_csv(os.path.join(out_dir, f"model_stats_CV{model_prefix}_bootstrap.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")