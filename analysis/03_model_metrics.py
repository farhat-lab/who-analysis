import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
import scipy.stats as st
import sys, pickle, sparse, glob, os, yaml
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV

import warnings
warnings.filterwarnings("ignore")
import tracemalloc


# starting the memory monitoring
tracemalloc.start()
    
    
minor_allele_counts = sparse.load_npz("data/minor_allele_counts.npz").todense()

# convert to dataframe
minor_allele_counts = pd.DataFrame(minor_allele_counts)
minor_allele_counts.columns = minor_allele_counts.iloc[0, :]
minor_allele_counts = minor_allele_counts.iloc[1:, :]
minor_allele_counts.rename(columns={0:"sample_id"}, inplace=True)
minor_allele_counts["sample_id"] = minor_allele_counts["sample_id"].astype(int)

# make sample ids the index again
minor_allele_counts = minor_allele_counts.set_index("sample_id")

mean_maf = pd.DataFrame(minor_allele_counts.mean(axis=0))
print(f"Min MAF: {round(mean_maf[0].min(), 2)}, Max MAF: {round(mean_maf[0].max(), 2)}")
del mean_maf

            
def get_threshold_val(y, y_proba, print_thresh=False):

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
    if print_thresh:
        print(f"    Selected threshold: {select_thresh}")

    # return the labels determined using the selected threshold
    return (pred_df["y_proba"] > select_thresh).astype(int).values




def generate_model_output(X, y, model, binary=True, print_thresh=False):
    
    
    # get predicted probabilities. The output is N x k dimensions, where N = number of samples, and k = number of classes
    # the second column is the probability of being in the class 1, so compute the classification threshold using that
    if binary:
        y_proba = model.predict_proba(X)[:, 1]
        y_hat = get_threshold_val(y, y_proba, print_thresh=print_thresh)

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
    else:
        y_hat = model.predict(X)
        mae = np.mean(np.abs((y - y_hat)))
        rmse = np.mean((y - y_hat)**2)
        
        # return a dataframe of the summary stats
        return pd.DataFrame({"MAE": mae,
                             "RMSE": rmse,
                            }, index=[0]
                           )        




def compute_downselected_logReg_model(drug, analysis_dir, pheno_category_lst, binary=True, num_bootstrap=1000):
    '''
    This model computes a logistic regression model using the significant predictors from the first model.
    
    The original model was used to assign coefficients/odds ratios and p-values. Using the significant predictors (p < 0.05 after FDR), this function
    builds another L2-penalized logistic regression to compute sensitivity, specificity, AUC, accuracy, and balanced accuracy. 
    '''
    
    # all features with non-zero coefficients in LR
    res_df = pd.read_csv(os.path.join(analysis_dir, drug, "final_analysis.csv"))
    
    # keep only features that were in the core model and with confidence intervals that lie entirely above or below 0
    # sometimes features may not pass all these filters (i.e. Delamanid), so in that case, only keep significant features, including non-core features
    # if len(res_df.query("Tier==1 & Phenos=='WHO' & unpooled==0 & synonymous==0 & ((coef_LB > 0 & coef_UB > 0) | (coef_LB < 0 & coef_UB < 0))")) == 0:
    #     res_df = res_df.query("(coef_LB > 0 & coef_UB > 0) | (coef_LB < 0 & coef_UB < 0)")
    #     print("Building predictive model with all significant predictors")
    # else:
    #     res_df = res_df.query("Tier==1 & Phenos=='WHO' & unpooled==0 & synonymous==0 & ((coef_LB > 0 & coef_UB > 0) | (coef_LB < 0 & coef_UB < 0))")
    #     print("Building predictive model with all significant core predictors")
    res_df = res_df.query("(coef_LB > 0 & coef_UB > 0) | (coef_LB < 0 & coef_UB < 0)")

    # remove principal components and the pooled variables
    res_df = res_df.loc[~res_df["orig_variant"].str.contains("|".join(["inframe", "lof", "PC"]))]
    
    df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv"), usecols=['sample_id', 'phenotypic_category', 'phenotype']).query("phenotypic_category in @pheno_category_lst")
    df_genos_full = pd.read_csv(os.path.join(analysis_dir, drug, "genos.csv.gz"), compression="gzip", usecols=['sample_id', 'resolved_symbol', 'variant_category', 'predicted_effect', 'variant_allele_frequency', 'variant_binary_status'])
    
#     def pool_mutations(df_genos, effect_lst, pool_col):

#         df = df_genos.copy()
#         df.loc[df["predicted_effect"].isin(effect_lst), ["variant_category", "position"]] = [pool_col, np.nan]

#         # sort descending to keep the largest variant_binary_status and variant_allele_frequency first. In this way, pooled mutations that are actually present are preserved
#         return df.query("variant_category == @pool_col").sort_values(by=["variant_binary_status", "variant_allele_frequency"], ascending=False, na_position="last").drop_duplicates(subset=["sample_id", "resolved_symbol"], keep="first")


#     if len(res_df.loc[res_df["orig_variant"].str.contains("lof")]) > 0:
#         df_pooled_lof = pool_mutations(df_genos, ["frameshift", "start_lost", "stop_gained", "feature_ablation"], "lof")
#         df_genos_full = pd.concat([df_genos, df_pooled_lof], axis=0)
#         del df_pooled_lof

#         if len(res_df.loc[res_df["orig_variant"].str.contains("inframe")]) > 0:
#             df_pooled_inframe = pool_mutations(df_genos, ["inframe_insertion", "inframe_deletion"], "inframe").reset_index(drop=True)
#             df_genos_full = pd.concat([df_genos_full, df_pooled_inframe], axis=0).reset_index(drop=True)
#     else:
#         if len(res_df.loc[res_df["orig_variant"].str.contains("inframe")]) > 0:
#             df_pooled_inframe = pool_mutations(df_genos, ["inframe_insertion", "inframe_deletion"], "inframe").reset_index(drop=True)
#             df_genos_full = pd.concat([df_genos, df_pooled_inframe], axis=0).reset_index(drop=True)
#             del df_pooled_inframe
#         else:
#             df_genos_full = df_genos.copy()

    # del df_genos
    df_genos_full["mutation"] = df_genos_full["resolved_symbol"] + "_" + df_genos_full["variant_category"]

    assert len(set(res_df["orig_variant"]) - set(df_genos_full["mutation"])) == 0
    df_genos_full = df_genos_full.loc[df_genos_full["mutation"].isin(res_df["orig_variant"])]
    
    drop_isolates = df_genos_full.loc[(pd.isnull(df_genos_full["variant_binary_status"])) & (df_genos_full["variant_allele_frequency"] <= 0.75)].sample_id.unique()
    df_genos_full = df_genos_full.query("sample_id not in @drop_isolates")
    model_matrix = df_genos_full.pivot(index="sample_id", columns="mutation", values="variant_binary_status")
    
    # drop any remaining isolates with any missingness (axis = 0)
    model_matrix = model_matrix.dropna(axis=0)
    model_matrix = model_matrix.merge(df_phenos[["sample_id", "phenotype"]], left_index=True, right_on="sample_id").set_index("sample_id")
    print(model_matrix.shape)

    # compute GRM using the minor allele counts of only the samples in the model
    single_drug_samples = model_matrix.index.values
    minor_allele_counts_single_drug = minor_allele_counts.query("sample_id in @single_drug_samples")
    grm = np.cov(minor_allele_counts_single_drug.values)
    
    minor_allele_counts_samples = minor_allele_counts_single_drug.index.values
    model_matrix = model_matrix.loc[minor_allele_counts_samples, :]
    
    scaler = StandardScaler()
    pca = PCA(n_components=5)
    pca.fit(scaler.fit_transform(grm))

    print(f"Explained variance ratios of 5 principal components: {pca.explained_variance_ratio_}")
    eigenvec = pca.components_.T
    eigenvec_df = pd.DataFrame(eigenvec)
    eigenvec_df.index = minor_allele_counts_samples
    
    # combine with eigevectors, then separate the phenotypes
    model_matrix = model_matrix.merge(eigenvec_df, left_index=True, right_index=True)
    if binary:
        y = model_matrix["phenotype"].values
        del model_matrix["phenotype"]
    
    X = scaler.fit_transform(model_matrix.values)
    
    # fit a regression model on the downselected data (only variants with non-zero coefficients and significant p-values after FDR)
    if binary:
        model = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
                                     cv=5,
                                     penalty='l2',
                                     max_iter=10000, 
                                     multi_class='ovr',
                                     scoring='neg_log_loss',
                                     class_weight='balanced'
                                    )
    else:
        model = RidgeCV(alphas=np.logspace(-6, 6, 13),
                        cv=5,
                        max_iter=10000,
                        scoring='neg_root_mean_squared_error'
                       )
    
    # fit and save the baseline model if you want to make more predictions later
    model.fit(X, y)
    if binary:
        print(f"    Regularization parameter: {model.C_[0]}")
        pickle.dump(model, open(os.path.join(analysis_dir, drug, 'core_logReg_model'),'wb'))
    else:
        print(f"    Regularization parameter: {model.alpha_}")
        pickle.dump(model, open(os.path.join(analysis_dir, drug, 'linReg_model'),'wb'))


    # get the summary stats for the overall model
    model_outputs = generate_model_output(X, y, model, binary=True, print_thresh=True)
    model_outputs["BS"] = 0
    
    # next, perform bootstrapping with 1000 replicates
    print(f"Bootstrapping the summary model with {num_bootstrap} replicates")
    for i in range(num_bootstrap):

        # randomly draw sample indices
        sample_idx = np.random.choice(np.arange(0, len(y)), size=len(y), replace=True)

        # get the X and y matrices
        X_bs = X[sample_idx, :]
        y_bs = y[sample_idx]

        if binary:
            bs_model = LogisticRegression(C=model.C_[0], penalty='l2', max_iter=10000, multi_class='ovr', class_weight='balanced')
        else:
            bs_model = Ridge(alpha=model.alpha_, max_iter=10000)
        
        bs_model.fit(X_bs, y_bs)
        summary = generate_model_output(X_bs, y_bs, bs_model, binary=binary, print_thresh=False)
        summary["BS"] = 1
        model_outputs = pd.concat([model_outputs, summary], axis=0)

    return pd.DataFrame(model_outputs)
    

analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'

binary = True
pheno_category_lst = ["WHO"]

#drugs_lst = ['Moxifloxacin', 'Capreomycin', 'Amikacin', 'Kanamycin', 'Ethionamide', 'Levofloxacin', 'Clofazimine', 'Linezolid', 'Bedaquiline', 'Delamanid', 'Streptomycin', 'Pyrazinamide', 'Ethambutol']
drugs_lst = ["Rifampicin"]

for drug in drugs_lst:
    print(f"\nWorking on {drug}")
    summary_df = compute_downselected_logReg_model(drug, analysis_dir, pheno_category_lst, binary=binary, num_bootstrap=1000)

    if binary:
        summary_df.to_csv(os.path.join(analysis_dir, drug, "core_logReg_summary.csv"), index=False)
    else:
        summary_df.to_csv(os.path.join(analysis_dir, drug, "linReg_summary.csv"), index=False)
    
    
# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()

# write peak memory usage in GB
with open("memory_usage.log", "a+") as file:
    file.write(f"{os.path.basename(__file__)}: {script_memory} GB\n")