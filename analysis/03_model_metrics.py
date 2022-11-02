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
from memory_profiler import profile

# open file for writing memory logs to. Append to file, not overwrite
mem_log=open('memory_usage.log','a+')
    
    
@profile(stream=mem_log)
def read_in_matrix_compute_grm(fName, samples):
    minor_allele_counts = sparse.load_npz(fName).todense()

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

    # compute GRM using the mino allele counts of only the samples in the model
    minor_allele_counts = minor_allele_counts.query("sample_id in @samples")
    grm = np.cov(minor_allele_counts.values)

    minor_allele_counts_samples = minor_allele_counts.index.values
    del minor_allele_counts
    return grm, minor_allele_counts_samples



            
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




@profile(stream=mem_log)
def compute_downselected_logReg_model(drug, out_dir, binary=True, num_bootstrap=1000):
    '''
    This model computes a logistic regression model using the significant predictors from the first model.
    
    The original model was used to assign coefficients/odds ratios and p-values. Using the significant predictors (p < 0.05 after FDR), this function
    builds another L2-penalized logistic regression to compute sensitivity, specificity, AUC, accuracy, and balanced accuracy. 
    '''
    
    # final_analysis file with all significant variants for a drug
    res_df = pd.read_csv(os.path.join(out_dir, drug, "final_analysis.csv"))
    
    # # get only significant variants: 0.05 for core features, 0.01 for the rest
    # res_df = res_df.query("(Tier1_only == 1 & WHO_phenos == 1 & poolLOF == 1 & Syn == 0 & BH_pval < 0.05) | (~(Tier1_only == 1 & WHO_phenos == 1 & poolLOF == 1 & Syn == 0) & BH_pval < 0.01)")
    
    # build model using all features with confidence intervals that lie entirely above or below 0
    res_df = res_df.query("(coef_LB > 0 & coef_UB > 0) | (coef_LB < 0 & coef_UB < 0)")

    # read in all genotypes and phenotypes and combine into a single dataframe. 
    # Take the dataframes with the most genotypes and phenotypes represented: tiers=1+2, phenos=ALL
    # if there are significant LOF variants in res_df, then get the corresponding poolLOF matrix and combine matrices 
    df_phenos = pd.read_csv(os.path.join(out_dir, drug, "tiers=1+2/phenos=ALL/dropAF_withSyn", "phenos.csv"))
    
    if len(res_df.loc[res_df["orig_variant"].str.contains("lof")]) > 0:
        model_inputs = pd.read_pickle(os.path.join(out_dir, drug, "tiers=1+2/phenos=ALL/dropAF_withSyn", "filt_matrix.pkl"))
        model_inputs_poolLOF = pd.read_pickle(os.path.join(out_dir, drug, "tiers=1+2/phenos=ALL/dropAF_withSyn_poolLOF", "filt_matrix.pkl"))
        
        # combine dataframes and remove duplicate columns
        model_inputs = pd.concat([model_inputs, model_inputs_poolLOF], axis=1)
        model_inputs = model_inputs.loc[:,~model_inputs.columns.duplicated()]

    else:
        model_inputs = pd.read_pickle(os.path.join(out_dir, drug, "tiers=1+2/phenos=ALL/dropAF_withSyn", "filt_matrix.pkl"))
        
    # combine into a single dataframe and check that there are no principal components left (because there aren't in df_phenos)
    combined = model_inputs.merge(df_phenos[["sample_id", "phenotype"]], on="sample_id", how="inner").set_index("sample_id")
    assert sum(combined.columns.str.contains("PC")) == 0
    
    # compute GRM and get only samples that are represented in the GRM (it should be everything, but this is just to avoid errors)
    # GRM is in the order of minor_allele_counts_samples (N x N)
    grm, minor_allele_counts_samples = read_in_matrix_compute_grm("data/minor_allele_counts.npz", combined.index.values)
    combined = combined.loc[minor_allele_counts_samples, :]
    
    scaler = StandardScaler()
    pca = PCA(n_components=5)
    pca.fit(scaler.fit_transform(grm))

    print(f"Explained variance ratios of 5 principal components: {pca.explained_variance_ratio_}")
    eigenvec = pca.components_.T
    eigenvec_df = pd.DataFrame(eigenvec)
    eigenvec_df.index = minor_allele_counts_samples
    
    # combine with eigevectors, then separate the phenotypes
    combined = combined.merge(eigenvec_df, left_index=True, right_index=True)
    if binary:
        y = combined["phenotype"].values
        del combined["phenotype"]
    
    X = scaler.fit_transform(combined.values)
    
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
        pickle.dump(model, open(os.path.join(out_dir, drug, 'logReg_model'),'wb'))
    else:
        print(f"    Regularization parameter: {model.alpha_}")
        pickle.dump(model, open(os.path.join(out_dir, drug, 'linReg_model'),'wb'))


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
        
        if i % (num_bootstrap / 10) == 0:
            print(i)

    return pd.DataFrame(model_outputs)
    

_, drug = sys.argv

out_dir = '/n/data1/hms/dbmi/farhat/ye12/who/analysis'

# run logistic regression model using only significant predictors saved in the model_analysis.csv file
#if (het_mode != "AF") & (synonymous == True) and (len(tiers_lst) > 1):
binary = True

summary_df = compute_downselected_logReg_model(drug, out_dir, binary=binary, num_bootstrap=1000)

if binary:
    summary_df.to_csv(os.path.join(out_dir, drug, "logReg_summary.csv"), index=False)
else:
    summary_df.to_csv(os.path.join(out_dir, drug, "linReg_summary.csv"), index=False)