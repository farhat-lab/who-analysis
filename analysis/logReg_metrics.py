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




def compute_balanced_accuracy_score_single_variant(model_matrix, model_analysis, y, variant):

    matrix = model_matrix.copy()

    coef = model_analysis.query("orig_variant == @variant")["coef"].values[0]
    if coef > 0:
        matrix.loc[model_matrix[variant] == 1, "assoc"] = 1
        matrix.loc[model_matrix[variant] != 1, "assoc"] = 0
    else:
        matrix.loc[model_matrix[variant] == 1, "assoc"] = 0
        matrix.loc[model_matrix[variant] != 1, "assoc"] = 1

    return sklearn.metrics.accuracy_score(y, matrix["assoc"]), sklearn.metrics.balanced_accuracy_score(y, matrix["assoc"])




def compute_downselected_logReg_model(out_dir, tiers_lst, het_mode, synonymous):
    
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
    

    # update the model_analysis dataframe with accuracy metrics (sens, spec, ppv are already there)
    for i, row in model_analysis.iterrows():

        if "PC" not in row["orig_variant"]:
            model_analysis.loc[i, ["accuracy", "balanced_accuracy"]] = compute_balanced_accuracy_score_single_variant(downselect_matrix, model_analysis, y, row["orig_variant"])

    # save the updated dataframe
    model_analysis.to_csv(os.path.join(out_dir, "model_analysis.csv"), index=False)

    # get predicted probabilities. The output is N x k dimensions, where N = number of samples, and k = number of classes
    # the second column is the probability of being in the class 1, so compute the classification threshold using that
    y_proba = small_model.predict_proba(X)[:, 1]
    y_hat = get_threshold_val(y, y_proba)

    # compute sensitivity, specificity, and accuracy scores (balanced and unbalanced)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y, y_hat).ravel()
    sens = tp / (tp+fn)
    spec = tn / (tn+fp)

    # return the values for the overall model
    return sens, spec, sklearn.metrics.accuracy_score(y, y_hat), sklearn.metrics.balanced_accuracy_score(y, y_hat)



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
if (het_mode != "AF") & (synonymous == True) and (len(tiers_lst) > 1):
    sens, spec, acc, balanced_acc = compute_downselected_logReg_model(out_dir, tiers_lst, het_mode, synonymous)
    pd.DataFrame({"Sens": sens, "Spec": spec, "accuracy": acc, "balanced_accuracy": balanced_acc}, index=[0]).to_csv(os.path.join(out_dir, "logReg_summary.csv"), index=False)