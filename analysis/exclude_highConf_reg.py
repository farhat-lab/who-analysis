import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
import scipy.stats as st
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
import tracemalloc, pickle, warnings
warnings.filterwarnings("ignore")


############# DONE PREVIOUSLY: CREATE A DICTIONARY OF FEATURES AND THE ISOLATES THAT SHOULD BE DROPPED FROM THE ANALYSIS #############

# these are solates that have Tier 1 or 2 rpoB mutations and the mutation of interest, which could confound. 
# We want to fit the model on only isolates that don't have these potentially suspicious tier 2 variants co-occurring with known resistance variants

# # read in genotypes dataframe
# genos = pd.read_csv(f"/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue/{drug}/genos.csv.gz", 
#                     compression="gzip", 
#                     usecols=["sample_id", "resolved_symbol", "variant_category", "variant_binary_status"]
#                    )

# # keep track of total number of samples
# all_samples = genos.sample_id.unique()

# # keep only variants tht are present
# genos = genos.query("variant_binary_status==1")

# # create the mutation column for compatibility with who_variants_combined
# genos["mutation"] = genos["resolved_symbol"] + "_" + genos["variant_category"]
# del genos["resolved_symbol"]
# del genos["variant_category"]

# who_variants_combined = pd.read_csv("analysis/who_confidence_2021.csv")
# who_variants_single_drug = who_variants_combined.query("drug==@drug_WHO_abbr")
# del who_variants_single_drug["drug"]
# del who_variants_combined

# # get high confidence mutations and the samples with it
# highConf_mutations = who_variants_single_drug.query("confidence in ['1) Assoc w R', '2) Assoc w R - Interim']")["mutation"].values
# print(len(highConf_mutations))
# highConf_samples = genos.query("mutation in @highConf_mutations").sample_id.unique()



############# DONE PREVIOUSLY: GET ALL MUTATIONS TO PERFORM THE LIKELIHOOD RATIO TEST FOR #############


# # model_7 = tiers 1+2, phenos=WHO only, no synonymous, unpooled. model_15 = same, but phenos=ALL
# rif_binary_results = pd.read_excel(f"results/BINARY/{drug}.xlsx", sheet_name=["Model_7", "Model_15"])

# model_7 = rif_binary_results["Model_7"]
# model_15 = rif_binary_results["Model_15"]
# print(model_7.shape, model_15.shape)

# # glpK_p.Val192fs has an OR_LB of 1.03
# or_thresh = 1.1
# lrt_mut7 = model_7.query("OR_LB > @or_thresh & BH_pval < 0.01 & confidence not in ['1) Assoc w R', '2) Assoc w R - Interim']")
# lrt_mut15 = model_15.query("OR_LB > @or_thresh & BH_pval < 0.01 & confidence not in ['1) Assoc w R', '2) Assoc w R - Interim']")

# lrt_mut7 = list(set(lrt_mut7["mutation"]).intersection(lrt_mut15["mutation"]).union(lrt_mut7["mutation"]))
# lrt_mut15 = list(set(lrt_mut15["mutation"]) - set(lrt_mut7))
# print(f"Performing LRT for {len(lrt_mut7) + len(lrt_mut15)} mutations with OR lower bounds above {or_thresh}!")


# def get_samples_to_exclude(genos, variant, highConf_mutations):
    
#     variant_samples = genos.query("mutation == @variant").sample_id.unique()
#     exclude_samples = set(highConf_mutations).intersection(variant_samples)
#     return list(exclude_samples)


# exclude_samples_by_variant = {lrt_mut[0]: get_samples_to_exclude(genos, lrt_mut[0], tier1_samples)}

# for i, variant in enumerate(lrt_mut):
#     if i > 0:
#         print(variant)
#         exclude_samples = get_samples_to_exclude(genos, variant, tier1_samples)
#         exclude_samples_by_variant[variant] = exclude_samples
        
# with open('/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue/Rifampicin/BINARY/exclude_rpoB_highConf_samples.pkl', 'wb') as file:
#     pickle.dump(exclude_samples_by_variant, file)
    

############# STEP 0: READ IN PARAMETERS FILE AND PREVIOUSLY GENERATED MATRICES AND GET DIRECTORIES #############

    
# starting the memory monitoring
tracemalloc.start()

_, config_file, drug, drug_WHO_abbr = sys.argv

kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
binary = kwargs["binary"]
atu_analysis = kwargs["atu_analysis"]
assert binary
assert not atu_analysis
analysis_dir = kwargs["output_dir"]

pheno_category_lst = kwargs["pheno_category_lst"]
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
    pheno_category_lst = ["ALL", "WHO"]
else:
    phenos_name = "WHO"
        
model_prefix = kwargs["model_prefix"]
num_PCs = kwargs["num_PCs"]

out_dir = os.path.join(analysis_dir, drug, "BINARY", f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}", model_prefix)
model_matrix = pd.read_pickle(os.path.join(out_dir, "model_matrix.pkl")).reset_index()
    
phenos_file = os.path.join(analysis_dir, drug, "phenos_binary.csv")
df_phenos = pd.read_csv(phenos_file).set_index("sample_id")

genos_file = os.path.join(analysis_dir, drug, "genos.csv.gz")    

# Read in the PC coordinates dataframe, then keep only the desired number of principal components
eigenvec_df = pd.read_csv("data/eigenvec_10PC.csv", index_col=[0])
eigenvec_df = eigenvec_df.iloc[:, :num_PCs]

scaler = StandardScaler()


############# STEP 1: READ IN DICTIONARY OF SAMPLES TO EXCLUDE BY FEATURE, REMOVE THEM, AND COMBINE MODEL MATRIX WITH THE EIGENVECTORS #############


with open(os.path.join(analysis_dir, drug, "BINARY", "exclude_rpoB_highConf_samples.pkl"), 'rb') as file:
    samples_to_exclude_dict = pickle.load(file)
    
variant = "rpoC_p.Glu1092Asp"
samples_to_exclude = samples_to_exclude_dict[variant]
prev_size = len(model_matrix)
model_matrix = model_matrix.query("sample_id not in @samples_to_exclude").set_index("sample_id")
print(f"Removed {len(model_matrix)-prev_size} isolates that don't have {variant} occurring without a resistance-associated mutation")

# keep only eigenvec coordinates for samples in the matrix dataframe
eigenvec_df = eigenvec_df.loc[model_matrix.index]
df_phenos = df_phenos.loc[model_matrix.index]

assert sum(model_matrix.merge(eigenvec_df, left_index=True, right_index=True).index != df_phenos.index.values) == 0

# concatenate the eigenvectors to the matrix and check the index ordering against the phenotypes matrix
model_matrix = model_matrix.merge(eigenvec_df, left_index=True, right_index=True)


############# STEP 2: FIT L2-PENALIZED REGRESSION FOR EACH VARIANT's MODEL #############


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
    print(f"Binarization threshold: {select_thresh}")

    # return the predicted class labels
    return (pred_df["y_prob"] > select_thresh).astype(int).values


def run_regression(model_matrix, y):

    model = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
                                  cv=5,
                                  penalty='l2',
                                  max_iter=10000, 
                                  multi_class='ovr',
                                  scoring='neg_log_loss',
                                  class_weight='balanced'
                                 )

    
    X = scaler.fit_transform(model_matrix.values)
    model.fit(X, y)
    
    # get positive class probabilities and predicted classes after determining the binarization threshold
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = get_threshold_val_and_classes(y_prob, y)

    return results_df


# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")