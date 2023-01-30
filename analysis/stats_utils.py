import numpy as np
import pandas as pd
import glob, os, yaml, sys
import scipy.stats as st
import sklearn.metrics
import statsmodels.stats.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
import tracemalloc, pickle


scaler = StandardScaler()



def get_binary_metrics_from_model(model, X, y):
        
    # get positive class probabilities and predicted classes after determining the binarization threshold
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = get_threshold_val_and_classes(y_prob, y)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=y, y_pred=y_pred).ravel()
    
    return np.array([sklearn.metrics.roc_auc_score(y_true=y, y_score=y_prob),
                    tp / (tp + fn),
                    tn / (tn + fp),
                    sklearn.metrics.accuracy_score(y_true=y, y_pred=y_pred),
                   ])



def get_threshold_val_and_classes(y_prob, y_test):
    
    # Compute number resistant and sensitive
    num_samples = len(y_test)
    num_resistant = np.sum(y_test).astype(int)
    num_sensitive = num_samples - num_resistant
    
    pred_df = pd.DataFrame({"y_prob": y_prob, "y_test": y_test})

    # Test thresholds from 0 to 1, in 0.01 increments
    thresholds = np.linspace(0, 1, 101)
    
    fpr_ = []
    tpr_ = []

    for thresh in thresholds:
        
        # binarize using the threshold, then compute true and false positives
        pred_df["y_pred"] = (pred_df["y_prob"] > thresh).astype(int)
        
        tp = len(pred_df.loc[(pred_df["y_pred"] == 1) & (pred_df["y_test"] == 1)])
        fp = len(pred_df.loc[(pred_df["y_pred"] == 1) & (pred_df["y_test"] == 0)])

        # Compute FPR and TPR. FPR = FP / N. TPR = TP / P
        fpr_.append(fp / num_sensitive)
        tpr_.append(tp / num_resistant)

    fpr_ = np.array(fpr_)
    tpr_ = np.array(tpr_)

    sens_spec_sum = (1 - fpr_) + tpr_

    # get index of highest sum(s) of sens and spec. Arbitrarily take the first threshold when there are multiple
    best_sens_spec_sum_idx = np.where(sens_spec_sum == np.max(sens_spec_sum))[0][0]
    select_thresh = thresholds[best_sens_spec_sum_idx]

    # return the predicted class labels
    return (pred_df["y_prob"] > select_thresh).astype(int).values




# use the regularization parameter determined above
def perform_bootstrapping(model, X, y, num_bootstrap, binary=True, save_summary_stats=False):
    
    if type(model) == float or type(model) == int:
        reg_param = model
    else:
        if binary:
            reg_param = model.C_[0]
        else:
            reg_param = model.alpha_
    
    coefs = []
    summary_stats_df = pd.DataFrame(columns=["AUC", "Sens", "Spec", "accuracy"])
    
    for i in range(num_bootstrap):

        # randomly draw sample indices
        sample_idx = np.random.choice(np.arange(0, len(y)), size=len(y), replace=True)

        # get the X and y matrices
        X_bs = scaler.fit_transform(X[sample_idx, :])
        y_bs = y[sample_idx]

        if binary:
            bs_model = LogisticRegression(C=reg_param, penalty='l2', max_iter=10000, multi_class='ovr', class_weight='balanced')
        else:
            bs_model = Ridge(alpha=reg_param, max_iter=10000)
        
        bs_model.fit(X_bs, y_bs)
        coefs.append(np.squeeze(bs_model.coef_))
        
        # also output a dataframe with the AUC, sensitivity, specificity, and accuracy of the model
        if save_summary_stats:
            
            y_prob = bs_model.predict_proba(X_bs)[:, 1]
            y_pred = get_threshold_val_and_classes(y_prob, y_bs)
            
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=y_bs, y_pred=y_pred).ravel()
            
            summary_stats_df.loc[i, :] = [sklearn.metrics.roc_auc_score(y_true=y_bs, y_score=y_prob),
                                           tp / (tp + fn),
                                           tn / (tn + fp),
                                           sklearn.metrics.accuracy_score(y_true=y_bs, y_pred=y_pred),
                                          ]

    if save_summary_stats:        
        return pd.DataFrame(coefs), summary_stats_df
    else:
        return pd.DataFrame(coefs)

    
    
    
# use the regularization parameter determined above
def perform_permutation_test(model, X, y, num_reps, binary=True):
    
    if type(model) == float or type(model) == int:
        reg_param = model
    else:
        if binary:
            reg_param = model.C_[0]
        else:
            reg_param = model.alpha_
    
    coefs = []    
    for i in range(num_reps):

        # shuffle phenotypes. np.random.shuffle works in-place
        y_permute = y.copy()
        np.random.shuffle(y_permute)

        if binary:
            rep_model = LogisticRegression(C=reg_param, penalty='l2', max_iter=10000, multi_class='ovr', class_weight='balanced')
        else:
            rep_model = Ridge(alpha=reg_param, max_iter=10000)
        
        rep_model.fit(X, y_permute)
        coefs.append(np.squeeze(rep_model.coef_))
        
    return pd.DataFrame(coefs)

    



def compute_univariate_stats(combined_df, variant_coef_dict):
    '''
    Compute positive predictive value. 
    Compute sensitivity, specificity, and positive and negative likelihood ratios. 
    
    PPV = true_positive / all_positive
    NPV = true_negative / all_negative
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
    
    # get counts of isolates grouped by phenotype and mutation -- so how many isolates have a mutation and have a phenotype (all 4 possibilities)
    grouped_df = pd.DataFrame(melted_2.groupby(["phenotype", "variable"]).value_counts()).reset_index()
    grouped_df = grouped_df.rename(columns={"variable": "mutation", "value": "present", 0:"count"})
    
    # add coefficients, create new column for the switched phenotypes (keep the old ones in actual_pheno)
    grouped_df["coef"] = grouped_df["mutation"].map(variant_coef_dict)
    grouped_df["actual_pheno"] = grouped_df["phenotype"].copy()
    assert sum(grouped_df["phenotype"] != grouped_df["actual_pheno"]) == 0

    # switch sign of the phenotypes for the negative coefficients and check
    grouped_df.loc[grouped_df["coef"] < 0, "phenotype"] = 1 - grouped_df.loc[grouped_df["coef"] < 0, "actual_pheno"]
    assert sum(grouped_df["phenotype"] != grouped_df["actual_pheno"]) == len(grouped_df.query("coef < 0"))
    
    # dataframes of the counts of the 4 values
    true_pos_df = grouped_df.query("present == 1 & phenotype == 1").rename(columns={"count": "TP"})
    false_pos_df = grouped_df.query("present == 1 & phenotype == 0").rename(columns={"count": "FP"})
    true_neg_df = grouped_df.query("present == 0 & phenotype == 0").rename(columns={"count": "TN"})
    false_neg_df = grouped_df.query("present == 0 & phenotype == 1").rename(columns={"count": "FN"})

    assert len(true_pos_df) + len(false_pos_df) + len(true_neg_df) + len(false_neg_df) == len(grouped_df)
    
    # combine the 4 dataframes into a single dataframe (concatenating on axis = 1)
    final = true_pos_df[["mutation", "TP"]].merge(
            false_pos_df[["mutation", "FP"]], on="mutation", how="outer").merge(
            true_neg_df[["mutation", "TN"]], on="mutation", how="outer").merge(
            false_neg_df[["mutation", "FN"]], on="mutation", how="outer").fillna(0)

    assert len(final) == len(melted["variable"].unique())
    assert len(final) == len(final.drop_duplicates("mutation"))
        
    # LR+ ranges from 1 to infinity. LR- ranges from 0 to 1
    final["Num_Isolates"] = final["TP"] + final["FP"]
    final["Total_Isolates"] = final[["TP", "FP", "TN", "FN"]].sum(axis=1)
    final["PPV"] = final["TP"] / (final["TP"] + final["FP"])
    final["NPV"] = final["TN"] / (final["TN"] + final["FN"])
    final["Sens"] = final["TP"] / (final["TP"] + final["FN"])
    final["Spec"] = final["TN"] / (final["TN"] + final["FP"])
    final["LR+"] = final["Sens"] / (1 - final["Spec"])
    final["LR-"] = (1 - final["Sens"]) / final["Spec"]
    
    return final[["mutation", "Num_Isolates", "Total_Isolates", "TP", "FP", "TN", "FN", "PPV", "NPV", "Sens", "Spec", "LR+", "LR-"]]
    


def compute_exact_confidence_intervals(res_df, alpha):
    
    res_df = res_df.reset_index(drop=True)
    
    # add exact binomial confidence intervals for the binomial variables. The other two will be done in another function
    for i, row in res_df.iterrows():
        
        # will be null for the principal components, so skip them
        if not pd.isnull(row["TP"]):
            
            # binomtest requires the numbers to be integers
            row[["TP", "FP", "TN", "FN"]] = row[["TP", "FP", "TN", "FN"]].astype(int)
        
            # PPV
            ci = st.binomtest(k=row["TP"], n=row["TP"] + row["FP"], p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
            res_df.loc[i, ["PPV_LB", "PPV_UB"]] = [ci.low, ci.high]
            
            # NPV
            ci = st.binomtest(k=row["TN"], n=row["TN"] + row["FN"], p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
            res_df.loc[i, ["NPV_LB", "NPV_UB"]] = [ci.low, ci.high]
            
            # Sensitivity
            ci = st.binomtest(k=row["TP"], n=row["TP"] + row["FN"], p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
            res_df.loc[i, ["Sens_LB", "Sens_UB"]] = [ci.low, ci.high]
            
            # Specificity
            ci = st.binomtest(k=row["TN"], n=row["TN"] + row["FP"], p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
            res_df.loc[i, ["Spec_LB", "Spec_UB"]] = [ci.low, ci.high]
    
    return res_df


    
def compute_likelihood_ratio_confidence_intervals(res_df, alpha):
    
    z = np.abs(st.norm.ppf(q=alpha/2))
    
    LRpos_error = np.exp(z * np.sqrt(1/res_df["TP"] - 1/(res_df["TP"] + res_df["FN"]) + 1/res_df["FP"] - 1/(res_df["FP"] + res_df["TN"])))
    LRneg_error = np.exp(z * np.sqrt(1/res_df["FN"] - 1/(res_df["TP"] + res_df["FN"]) + 1/res_df["TN"] - 1/(res_df["FP"] + res_df["TN"])))
    
    res_df["LR+_LB"] = res_df["LR+"] / LRpos_error
    res_df["LR+_UB"] = res_df["LR+"] * LRpos_error
    
    res_df["LR-_LB"] = res_df["LR-"] / LRneg_error
    res_df["LR-_UB"] = res_df["LR-"] * LRneg_error

    return res_df


    
    
def get_model_inputs_exclude_cooccur(variant, exclude_variants_dict, samples_highConf_tier1, df_model, df_phenos, eigenvec_df):
    '''
    Use this function to get the input matrix (dataframe) and phenotypes for a model, dropping all isolates that contain both the variant of interest 
    
    
    sample_id must be the index of BOTH df_phenos and eigenvec_df
    '''
    
    samples_with_variant = exclude_variants_dict[variant]
    samples_to_exclude = set(samples_with_variant).intersection(samples_highConf_tier1)
    print(f"{len(samples_to_exclude)} samples will be excluded")
    df_model = df_model.query("sample_id not in @samples_to_exclude")

    # drop more duplicates, but I think this might be because we have multiple data pulls at a time
    # NaN is larger than any number, so sort ascending and keep first
    
    matrix = df_model.pivot(index="sample_id", columns="mutation", values="variant_binary_status")

    # drop any isolate with missingness (because the RIF genes are well-sequenced areas), and any features that are 0 everywhere
    matrix = matrix.dropna(axis=0, how="any")
    matrix = matrix[matrix.columns[~((matrix == 0).all())]]
    
    if variant not in matrix.columns:
        print(f"{variant} was dropped from in the matrix")
        return None, None
    else:
        # combine with eigenvectors
        eigenvec_df = eigenvec_df.loc[matrix.index]
        matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True)
        assert sum(matrix.index != df_phenos.index.values) == 0

        y = df_phenos.loc[matrix.index]["phenotype"].values
    
    return matrix, y




def get_tier2_mutations_of_interest(analysis_dir, drug, phenos_name):
    
    
    # get all mutations from the Tiers 1+2 model with positive coefficients and significant p-values
    model_tiers12 = pd.read_csv(f"{analysis_dir}/{drug}/BINARY/tiers=1+2/phenos={phenos_name}/dropAF_noSyn_unpooled/model_analysis.csv").query("coef > 0 & BH_pval < 0.01")
    model_tier1 = pd.read_csv(f"{analysis_dir}/{drug}/BINARY/tiers=1/phenos={phenos_name}/dropAF_noSyn_unpooled/model_analysis.csv")

    # drop mutations that are in the Tier 1 model, only want to consider Tier 2 mutations
    tier2_mutations_of_interest = list(set(model_tiers12["mutation"]) - set(model_tier1["mutation"]))
    print(f"{len(tier2_mutations_of_interest)} significant tier 2 mutations associated with {phenos_name} resistance")
    return tier2_mutations_of_interest




def add_pval_corrections(df, col="pval"):
    '''
    Implement Benjamini-Hochberg FDR and Bonferroni corrections.
    '''
    
    # LINK TO HOW TO USE: https://tedboy.github.io/statsmodels_doc/generated/statsmodels.stats.multitest.fdrcorrection.html#statsmodels.stats.multitest.fdrcorrection

    if sum(pd.isnull(df[col])) > 0:
        raise ValueError("There are NaNs in the p-value column!")
        
    # alpha argument doesn't matter because we threshold later. It's only relevant if you keep the first argument, which is a boolean of reject vs. not reject
    _, bh_pvals, _, _ = sm.multipletests(df[col], method='fdr_bh', is_sorted=False, returnsorted=False)
    _, bonferroni_pvals, _, _ = sm.multipletests(df[col], method='bonferroni', is_sorted=False, returnsorted=False)

    df["BH_pval"] = bh_pvals
    df["Bonferroni_pval"] = bonferroni_pvals
    
    return df.reset_index(drop=True)