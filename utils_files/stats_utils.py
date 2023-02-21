import numpy as np
import pandas as pd
import glob, os, yaml, sys
import scipy.stats as st
import sklearn.metrics
import statsmodels.stats.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression, Lasso
import tracemalloc, pickle


scaler = StandardScaler()



def get_binary_metrics_from_model(model, X, y, return_idx):
        
    # get positive class probabilities and predicted classes after determining the binarization threshold
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = get_threshold_val_and_classes(y_prob, y)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=y, y_pred=y_pred).ravel()
    
    results = np.array([sklearn.metrics.roc_auc_score(y_true=y, y_score=y_prob),
                        tp / (tp + fn),
                        tn / (tn + fp),
                        sklearn.metrics.accuracy_score(y_true=y, y_pred=y_pred),
                        sklearn.metrics.balanced_accuracy_score(y_true=y, y_pred=y_pred),
                       ])
    
    return results[return_idx]



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
            if reg_param == 0:
                bs_model = LogisticRegression(penalty='none', max_iter=10000, multi_class='ovr', class_weight='balanced')
            else:
                bs_model = LogisticRegression(C=reg_param, penalty='l2', max_iter=10000, multi_class='ovr', class_weight='balanced')
        else:
            if reg_param == 0:
                bs_model = LinearRegression()
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
def perform_permutation_test(model, X, y, num_reps, binary=True, penalty_type='l2'):
    
    if penalty_type not in ["l1", "l2", None]:
        raise ValueError(f"{penalty_type} is a not a valid penalty type!")
        
    if type(model) == float or type(model) == int:
        reg_param = model
    else:
        if penalty_type is not None:
            if binary:
                reg_param = model.C_[0]
            else:
                reg_param = model.alpha_
        else:
            reg_param = 1
    
    coefs = []    
    for i in range(num_reps):

        # shuffle phenotypes. np.random.shuffle works in-place
        y_permute = y.copy()
        np.random.shuffle(y_permute)

        if binary:
            # HAVE SPLIT THIS UP BECAUSE THE UNPENALIZED REGRESSION CONVERGES BETTER WITH THE DEFAULT LBFGS SOLVER THAN WITH SAGA
            if penalty_type is None:
                rep_model = LogisticRegression(penalty=None, max_iter=10000, multi_class='ovr', class_weight='balanced')
            else:
                rep_model = LogisticRegression(C=reg_param, penalty=penalty_type, max_iter=10000, multi_class='ovr', class_weight='balanced', solver='saga')
        else:
            if penalty_type is None:
                rep_model = LinearRegression()
            else:
                if penalty_type == 'l1':
                    rep_model = Lasso(alpha=reg_param, max_iter=10000)
                else:
                    rep_model = Ridge(alpha=reg_param, max_iter=10000)
        
        rep_model.fit(X, y_permute)
        coefs.append(np.squeeze(rep_model.coef_))
        
    return pd.DataFrame(coefs)





def get_coef_and_confidence_intervals(alpha, binary, who_variants_combined, drug_WHO_abbr, coef_df, permute_df, bootstrap_df):
    
    # get dataframe of 2021 WHO confidence gradings
    who_variants_single_drug = who_variants_combined.query("drug==@drug_WHO_abbr")
    del who_variants_single_drug["drug"]
    del who_variants_combined

    # add confidence intervals for the coefficients for all mutation. first check ordering of mutations
    ci = (1-alpha)*100
    diff = (100-ci)/2
    assert sum(coef_df["mutation"].values != bootstrap_df.columns) == 0
    lower, upper = np.percentile(bootstrap_df, axis=0, q=(diff, 100-diff))
    coef_df["coef_LB"] = lower
    coef_df["coef_UB"] = upper

    # assess significance using the results of the permutation test
    for i, row in coef_df.iterrows():
        # p-value is the proportion of permutation coefficients that are AT LEAST AS EXTREME as the test statistic
        # ONE-SIDED because we are interested in the sign of the coefficient
        if row["coef"] > 0:
            coef_df.loc[i, "pval"] = np.mean(permute_df[row["mutation"]] >= row["coef"])
        else:
            coef_df.loc[i, "pval"] = np.mean(permute_df[row["mutation"]] <= row["coef"])
            
        # add p-value testing whether a mutation is neutral: computes the proportion of permuted coefficients that have smaller magnitude (are closer to 0) than the test stat
        # TWO-SIDED because a mutation can be neutral if its coef is slightly above or below 0
        coef_df.loc[i, "neutral_pval"] = np.mean(np.abs(permute_df[row["mutation"]]) < np.abs(row["coef"]))

    # Benjamini-Hochberg and Bonferroni corrections
    coef_df = add_pval_corrections(coef_df)

    # adjusted p-values are larger so that fewer null hypotheses (coef = 0) are rejected
    assert len(coef_df.query("pval > BH_pval")) == 0
    assert len(coef_df.query("pval > Bonferroni_pval")) == 0

    # convert to odds ratios
    if binary:
        coef_df["Odds_Ratio"] = np.exp(coef_df["coef"])
        coef_df["OR_LB"] = np.exp(coef_df["coef_LB"])
        coef_df["OR_UB"] = np.exp(coef_df["coef_UB"])

    # add in the WHO 2021 catalog confidence levels, using the dataframe with 2021 to 2022 mapping and save
    final_df = coef_df.merge(who_variants_single_drug, on="mutation", how="left")
    assert len(final_df) == len(coef_df)
    return final_df
    



def compute_univariate_stats(combined_df):
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

    # dataframes of the counts of the 4 values
    true_pos_df = grouped_df.query("present == 1 & phenotype == 1").rename(columns={"count": "Mut_R"})
    false_pos_df = grouped_df.query("present == 1 & phenotype == 0").rename(columns={"count": "Mut_S"})
    true_neg_df = grouped_df.query("present == 0 & phenotype == 0").rename(columns={"count": "NoMut_S"})
    false_neg_df = grouped_df.query("present == 0 & phenotype == 1").rename(columns={"count": "NoMut_R"})

    assert len(true_pos_df) + len(false_pos_df) + len(true_neg_df) + len(false_neg_df) == len(grouped_df)
    
    # combine the 4 dataframes into a single dataframe (concatenating on axis = 1)
    final = true_pos_df[["mutation", "Mut_R"]].merge(
            false_pos_df[["mutation", "Mut_S"]], on="mutation", how="outer").merge(
            true_neg_df[["mutation", "NoMut_S"]], on="mutation", how="outer").merge(
            false_neg_df[["mutation", "NoMut_R"]], on="mutation", how="outer").fillna(0)

    assert len(final) == len(melted["variable"].unique())
    assert len(final) == len(final.drop_duplicates("mutation"))
        
    # LR+ ranges from 1 to infinity. LR- ranges from 0 to 1
    final["Num_Isolates"] = final["Mut_R"] + final["Mut_S"]
    final["Total_Isolates"] = final[["Mut_R", "Mut_S", "NoMut_S", "NoMut_R"]].sum(axis=1)
    
    final["PPV"] = final["Mut_R"] / (final["Mut_R"] + final["Mut_S"])
    final["NPV"] = final["NoMut_S"] / (final["NoMut_S"] + final["NoMut_R"])
    
    final["Sens"] = final["Mut_R"] / (final["Mut_R"] + final["NoMut_R"])
    final["Spec"] = final["NoMut_S"] / (final["NoMut_S"] + final["Mut_S"])
    final["LR+"] = final["Sens"] / (1 - final["Spec"])
    final["LR-"] = (1 - final["Sens"]) / final["Spec"]
    
    return final[["mutation", "Num_Isolates", "Total_Isolates", "Mut_R", "Mut_S", "NoMut_S", "NoMut_R", "PPV", "NPV", "Sens", "Spec", "LR+", "LR-"]]
    


def compute_exact_confidence_intervals(res_df, alpha):
    
    res_df = res_df.reset_index(drop=True)
    
    # add exact binomial confidence intervals for the binomial variables. The other two will be done in another function
    for i, row in res_df.iterrows():
        
        # will be null for the principal components, so skip them
        if not pd.isnull(row["Mut_R"]):
            
            # binomtest requires the numbers to be integers
            row[["Mut_R", "Mut_S", "NoMut_S", "NoMut_R"]] = row[["Mut_R", "Mut_S", "NoMut_S", "NoMut_R"]].astype(int)
        
            # PPV
            ci = st.binomtest(k=row["Mut_R"], n=row["Mut_R"] + row["Mut_S"], p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
            res_df.loc[i, ["PPV_LB", "PPV_UB"]] = [ci.low, ci.high]
            
            # NPV
            ci = st.binomtest(k=row["NoMut_S"], n=row["NoMut_S"] + row["NoMut_R"], p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
            res_df.loc[i, ["NPV_LB", "NPV_UB"]] = [ci.low, ci.high]
            
            # Sensitivity
            ci = st.binomtest(k=row["Mut_R"], n=row["Mut_R"] + row["NoMut_R"], p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
            res_df.loc[i, ["Sens_LB", "Sens_UB"]] = [ci.low, ci.high]
            
            # Specificity
            ci = st.binomtest(k=row["NoMut_S"], n=row["NoMut_S"] + row["Mut_S"], p=0.5).proportion_ci(confidence_level=1-alpha, method='exact')
            res_df.loc[i, ["Spec_LB", "Spec_UB"]] = [ci.low, ci.high]
    
    return res_df


    
def compute_likelihood_ratio_confidence_intervals(res_df, alpha):
    
    z = np.abs(st.norm.ppf(q=alpha/2))
    
    LRpos_error = np.exp(z * np.sqrt(1/res_df["Mut_R"] - 1/(res_df["Mut_R"] + res_df["NoMut_R"]) + 1/res_df["Mut_S"] - 1/(res_df["Mut_S"] + res_df["NoMut_S"])))
    LRneg_error = np.exp(z * np.sqrt(1/res_df["NoMut_R"] - 1/(res_df["Mut_R"] + res_df["NoMut_R"]) + 1/res_df["NoMut_S"] - 1/(res_df["Mut_S"] + res_df["NoMut_S"])))
    
    res_df["LR+_LB"] = res_df["LR+"] / LRpos_error
    res_df["LR+_UB"] = res_df["LR+"] * LRpos_error
    
    res_df["LR-_LB"] = res_df["LR-"] / LRneg_error
    res_df["LR-_UB"] = res_df["LR-"] * LRneg_error

    return res_df




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
    
    if f"neutral_{col}" in df.columns:
        _, neutral_bh_pvals, _, _ = sm.multipletests(df[f"neutral_{col}"], method='fdr_bh', is_sorted=False, returnsorted=False)
        _, neutral_bonferroni_pvals, _, _ = sm.multipletests(df[f"neutral_{col}"], method='bonferroni', is_sorted=False, returnsorted=False)

        df["neutral_BH_pval"] = neutral_bh_pvals
        df["neutral_Bonferroni_pval"] = neutral_bonferroni_pvals
    
    return df.reset_index(drop=True)