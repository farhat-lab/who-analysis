import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
import scipy.stats as st
import sys, pickle
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

import glob, os, yaml
import warnings
warnings.filterwarnings("ignore")



def compute_predictive_values(combined_df, return_stats=[]):
    '''
    Compute positive predictive value. 
    Compute sensitivity, specificity, and positive and negative likelihood ratios. 
    
    PPV = true_positive / all_positive. NPV = true_negative / all_negative
    Sens = true_positive / (true_positive + false_negative)
    Spec = true_negative / (true_negative + false_positive)
    
    Also return the number of isolates with each variant = all_positive
    
    Positive LR = sens / (1 – spec)
    Negative LR = (1 – sens) / spec

    '''
    # make a copy to keep sample_id in one dataframe
    melted = combined_df.melt(id_vars=["sample_id", "phenotype"])
    melted_2 = melted.copy()
    del melted_2["sample_id"]
    
    # get counts of isolates grouped by phenotype and variant -- so how many isolates have a variant and have a phenotype (all 4 possibilities)
    grouped_df = pd.DataFrame(melted_2.groupby(["phenotype", "variable"]).value_counts()).reset_index()
    grouped_df = grouped_df.rename(columns={"variable": "orig_variant", "value": "variant", 0:"count"})
    
    # dataframes of the counts of the 4 values
    true_pos_df = grouped_df.query("variant == 1 & phenotype == 1").rename(columns={"count": "TP"})
    false_pos_df = grouped_df.query("variant == 1 & phenotype == 0").rename(columns={"count": "FP"})
    true_neg_df = grouped_df.query("variant == 0 & phenotype == 0").rename(columns={"count": "TN"})
    false_neg_df = grouped_df.query("variant == 0 & phenotype == 1").rename(columns={"count": "FN"})

    assert len(true_pos_df) + len(false_pos_df) + len(true_neg_df) + len(false_neg_df) == len(grouped_df)
    
    # combine the 4 dataframes into a single dataframe (concatenating on axis = 1)
    final = true_pos_df[["orig_variant", "TP"]].merge(
            false_pos_df[["orig_variant", "FP"]], on="orig_variant", how="outer").merge(
            true_neg_df[["orig_variant", "TN"]], on="orig_variant", how="outer").merge(
            false_neg_df[["orig_variant", "FN"]], on="orig_variant", how="outer").fillna(0)

    assert len(final) == len(melted["variable"].unique())
    assert len(final) == len(final.drop_duplicates("orig_variant"))
    
    # check that all feature rows have the same number of samples    
    assert len(np.unique(final[["TP", "FP", "TN", "FN"]].sum(axis=1))) == 1
        
    final["Num_Isolates"] = final["TP"] + final["FP"]
    final["Total_Isolates"] = final["TP"] + final["FP"] + final["TN"] + final["FN"]
    final["PPV"] = final["TP"] / (final["TP"] + final["FP"])
    final["Sens"] = final["TP"] / (final["TP"] + final["FN"])
    final["Spec"] = final["TN"] / (final["TN"] + final["FP"])
    final["LR+"] = final["Sens"] / (1 - final["Spec"])
    final["LR-"] = (1 - final["Sens"]) / final["Spec"]
    #final["NPV"] = final["TN"] / (final["TN"] + final["FN"])
    
    if len(return_stats) == 0:
        return final[["orig_variant", "Num_Isolates", "Total_Isolates", "TP", "FP", "TN", "FN", "PPV", "Sens", "Spec", "LR+", "LR-"]]
    else:
        return final[return_stats]
    
    

def compute_univariate_stats(**kwargs):
    
    tiers_lst = kwargs["tiers_lst"]
    pheno_category_lst = kwargs["pheno_category_lst"]
    model_prefix = kwargs["model_prefix"]
    het_mode = kwargs["het_mode"]
    synonymous = kwargs["synonymous"]
    pool_lof = kwargs["pool_lof"]
    AF_thresh = kwargs["AF_thresh"]

    num_PCs = kwargs["num_PCs"]
    num_bootstrap = kwargs["num_bootstrap"]
    alpha = kwargs["alpha"]

    # model_analysis file with all nominally significant variants
    res_df = pd.read_csv(os.path.join(out_dir, "model_analysis.csv"))
    
    # read in all genotypes and phenotypes and combine into a single dataframe
    model_inputs = pd.read_pickle(os.path.join(out_dir, "model_matrix.pkl"))
    df_phenos = pd.read_csv(os.path.join(out_dir, "phenos.csv"))
    combined = model_inputs.merge(df_phenos[["sample_id", "phenotype"]], on="sample_id").reset_index(drop=True)

    # compute univariate stats for only the lof variable
    if pool_lof:
        keep_variants = list(res_df.loc[res_df["orig_variant"].str.contains("lof")]["orig_variant"].values)
    else:
        keep_variants = list(res_df.loc[~res_df["orig_variant"].str.contains("PC")]["orig_variant"].values)
        
    # check that all samples were preserved
    combined_small = combined[["sample_id", "phenotype"] + keep_variants]
    assert len(combined_small) == len(combined)
    
    #### Compute univariate statistics only for cases where genotypes are binary (no AF), synonymous are included, all features ####
    #### For LOF, only compute univariate stats for the LOF variables. Otherwise, the corresponding non-LOF model contains everything #### 
    #### In the LOF case, if no LOF variants (there is 1 LOF per gene) are significant, then keep_variants = [], and we don't run this block of code ####
    #if (het_mode != "AF") & (synonymous == True) and (len(tiers_lst) > 1) and (len(keep_variants) > 0):
    if (het_mode != "AF") & (len(keep_variants) > 0):
        
        # get dataframe of predictive values for the non-zero coefficients and add them to the results dataframe
        full_predict_values = compute_predictive_values(combined_small)
        res_df = res_df.merge(full_predict_values, on="orig_variant", how="outer")

        print(f"Computing and bootstrapping predictive values with {num_bootstrap} replicates")
        bs_results = pd.DataFrame(columns = keep_variants)

        # need confidence intervals for 5 stats: PPV, sens, spec, + likelihood ratio, - likelihood ratio
        for i in range(num_bootstrap):

            # get bootstrap sample
            bs_idx = np.random.choice(np.arange(0, len(combined_small)), size=len(combined_small), replace=True)
            bs_combined = combined_small.iloc[bs_idx, :]

            # check ordering of features because we're just going to append bootstrap dataframes
            assert sum(bs_combined.columns[2:] != bs_results.columns) == 0

            # get predictive values from the dataframe of bootstrapped samples. Only return the 5 we want CI for, and the variant
            bs_values = compute_predictive_values(bs_combined, return_stats=["orig_variant", "PPV", "Sens", "Spec", "LR+", "LR-"])
            bs_results = pd.concat([bs_results, bs_values.set_index("orig_variant").T], axis=0)

        # add the confidence intervals to the dataframe
        for variable in ["PPV", "Sens", "Spec", "LR+", "LR-"]:

            lower, upper = np.nanpercentile(bs_results.loc[variable], q=[2.5, 97.5], axis=0)

            # LR+ can be infinite if spec is 1, and after percentile, it will be NaN, so replace with infinity
            if variable == "LR+":
                res_df[variable] = res_df[variable].fillna(np.inf)
                lower[np.isnan(lower)] = np.inf
                upper[np.isnan(upper)] = np.inf

            res_df = res_df.merge(pd.DataFrame({"orig_variant": bs_results.columns, 
                                f"{variable}_LB": lower,
                                f"{variable}_UB": upper,
                               }), on="orig_variant", how="outer")

            # sanity checks -- lower bounds should be <= true values, and upper bounds should be >= true values
            assert sum(res_df[variable] < res_df[f"{variable}_LB"]) == 0
            assert sum(res_df[variable] > res_df[f"{variable}_UB"]) == 0
            
            
            
def get_threshold_val(y, y_proba):

    # Compute true number resistant and sensitive
    num_samples = len(y)
    num_resistant = np.sum(y).astype(int)
    num_sensitive = num_samples - num_resistant

    # Test thresholds from 0 to 1, in 0.01 increments
    thresholds = np.linspace(0, 1, 101)

    fpr_ = []
    tpr_ = []

    # put in dataframe for easy slicing
    pred_df = pd.DataFrame({"y_proba": y_proba, "y": y})

    for thresh in thresholds:

        # binarize using the threshold, then compute true and false positives
        pred_df["y_pred"] = (pred_df["y_proba"] > thresh).astype(int)

        tp = len(pred_df.query("y_pred == 1 & y == 1"))
        fp = len(pred_df.query("y_pred == 1 & y == 0"))

        # Compute FPR and TPR. FPR = FP / N. TPR = TP / P
        fpr_.append(fp / num_sensitive)
        tpr_.append(tp / num_resistant)

    fpr_ = np.array(fpr_)
    tpr_ = np.array(tpr_)

    sens_spec_sum = (1 - fpr_) + tpr_

    # get index of highest sum(s) of sens and spec. Arbitrarily take the first threshold when there are multiple
    best_sens_spec_sum_idx = np.where(sens_spec_sum == np.max(sens_spec_sum))[0][0]
    select_thresh = thresholds[best_sens_spec_sum_idx]
    print(f"    Selected threshold: {select_thresh}")

    # return the labels determined using the selected threshold
    return (pred_df["y_proba"] > select_thresh).astype(int).values




# def compute_balanced_accuracy_score_single_variant(model_matrix, model_analysis, y, variant):

#     matrix = model_matrix.copy()

#     coef = model_analysis.query("orig_variant == @variant")["coef"].values[0]
#     if coef > 0:
#         matrix.loc[model_matrix[variant] == 1, "assoc"] = 1
#         matrix.loc[model_matrix[variant] != 1, "assoc"] = 0
#     else:
#         matrix.loc[model_matrix[variant] == 1, "assoc"] = 0
#         matrix.loc[model_matrix[variant] != 1, "assoc"] = 1

#     return sklearn.metrics.accuracy_score(y, matrix["assoc"]), sklearn.metrics.balanced_accuracy_score(y, matrix["assoc"])




def compute_downselected_logReg_model(out_dir, tiers_lst, het_mode, synonymous):
    '''
    This model computes a logistic regression model using the significant predictors from the first model.
    
    The original model was used to assign coefficients/odds ratios and p-values. Using the significant predictors (p < 0.05 after FDR), this function
    builds another L2-penalized logistic regression to compute sensitivity, specificity, AUC, accuracy, and balanced accuracy. 
    '''
    model_analysis = pd.read_csv(os.path.join(out_dir, "model_analysis.csv"))
    model_matrix = pd.read_pickle(os.path.join(out_dir, "model_matrix.pkl"))

    eigenvec_df = pd.read_pickle(os.path.join(out_dir, "model_eigenvecs.pkl"))
    eigenvec_df.columns = [f"PC{num}" for num in eigenvec_df.columns]

    df_phenos = pd.read_csv(os.path.join(out_dir, "phenos.csv")).sort_values("sample_id").reset_index(drop=True)
    y = df_phenos.phenotype.values
    
    # get all significant features
    downselect_matrix = model_matrix.merge(eigenvec_df, left_index=True, right_index=True)[model_analysis["orig_variant"]]
    assert sum(df_phenos["sample_id"] != downselect_matrix.index) == 0
    assert len(model_analysis) == downselect_matrix.shape[1]
    del model_matrix
    
    scaler = StandardScaler()
    X = scaler.fit_transform(downselect_matrix.values)
    
    # fit a logistic regression model on the downselected data (only variants with non-zero coefficients and significant p-values after FDR)
    small_model = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
                                 cv=5,
                                 penalty='l2', 
                                 max_iter=10000, 
                                 multi_class='ovr',
                                 scoring='neg_log_loss',
                                 class_weight='balanced'
                                )

    # fit and save the baseline model
    small_model.fit(X, y)
    print(f"    Regularization parameter: {small_model.C_[0]}")
    pickle.dump(small_model, open(os.path.join(out_dir, 'logReg_model'),'wb'))

#     # update the model_analysis dataframe with accuracy metrics (sens, spec, ppv are already there)
#     for i, row in model_analysis.iterrows():

#         if "PC" not in row["orig_variant"]:
#             model_analysis.loc[i, ["accuracy", "balanced_accuracy"]] = compute_balanced_accuracy_score_single_variant(downselect_matrix, model_analysis, y, row["orig_variant"])

#     # save the updated dataframe
#     model_analysis.to_csv(os.path.join(out_dir, "model_analysis.csv"), index=False)

    # get predicted probabilities. The output is N x k dimensions, where N = number of samples, and k = number of classes
    # the second column is the probability of being in the class 1, so compute the classification threshold using that
    y_proba = small_model.predict_proba(X)[:, 1]
    y_hat = get_threshold_val(y, y_proba)

    # compute sensitivity, specificity, and accuracy scores (balanced and unbalanced)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y, y_hat).ravel()
    sens = tp / (tp+fn)
    spec = tn / (tn+fp)

    # return a dataframe of the summary stats
    return pd.DataFrame({"Sens": sens,
                         "Spec": spec,
                         "AUC": sklearn.metrics.roc_auc_score(y, y_hat),
                         "F1": sklearn.metrics.f1_score(y, y_hat),
                         "accuracy": sklearn.metrics.accuracy_score(y, y_hat),
                         "balanced_accuracy": sklearn.metrics.balanced_accuracy_score(y, y_hat),
                        }, index=[0]
                       )



_, config_file, drug = sys.argv

kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
pheno_category_lst = kwargs["pheno_category_lst"]
model_prefix = kwargs["model_prefix"]
het_mode = kwargs["het_mode"]
synonymous = kwargs["synonymous"]

out_dir = '/n/data1/hms/dbmi/farhat/ye12/who/analysis'
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
else:
    phenos_name = "WHO"

out_dir = os.path.join(out_dir, drug, f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}", model_prefix)

if not os.path.isdir(out_dir):
    print("No model for this analysis")
    exit()

# run logistic regression model using only significant predictors saved in the model_analysis.csv file
#if (het_mode != "AF") & (synonymous == True) and (len(tiers_lst) > 1):
summary_df = compute_downselected_logReg_model(out_dir, tiers_lst, het_mode, synonymous)
summary_df.to_csv(os.path.join(out_dir, "logReg_summary.csv"), index=False)