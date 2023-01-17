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


############# STEP 0: READ IN PARAMETERS FILE AND GET DIRECTORIES #############

    
# starting the memory monitoring
tracemalloc.start()

_, config_file, drug, _ = sys.argv

kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
binary = kwargs["binary"]
atu_analysis = kwargs["atu_analysis"]
analysis_dir = kwargs["output_dir"]

# double check. If running CC vs. CC-ATU analysis, they are binary phenotypes
if atu_analysis:
    binary = True

pheno_category_lst = kwargs["pheno_category_lst"]
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
    pheno_category_lst = ["ALL", "WHO"]
else:
    phenos_name = "WHO"
        
model_prefix = kwargs["model_prefix"]
num_PCs = kwargs["num_PCs"]
num_bootstrap = kwargs["num_bootstrap"]

if binary:
    if atu_analysis:
        out_dir = os.path.join(analysis_dir, drug, "ATU", f"tiers={'+'.join(tiers_lst)}", model_prefix)
        
        # the CC and CC-ATU models are in the same folder, but the output files (i.e. regression_coef.csv have different suffixes to distinguish)
        model_suffix = kwargs["atu_analysis_type"]
        assert model_suffix == "CC" or model_suffix == "CC-ATU"
    else:
        out_dir = os.path.join(analysis_dir, drug, "BINARY", f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}", model_prefix)
else:
    out_dir = os.path.join(analysis_dir, drug, "MIC", f"tiers={'+'.join(tiers_lst)}", model_prefix)

scaler = StandardScaler()
    

############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############


if binary:
    if atu_analysis:
        phenos_file = os.path.join(analysis_dir, drug, "phenos_atu.csv")
    else:
        phenos_file = os.path.join(analysis_dir, drug, "phenos_binary.csv")
else:
    phenos_file = os.path.join(analysis_dir, drug, "phenos_mic.csv")

df_phenos = pd.read_csv(phenos_file)

if atu_analysis:
    df_phenos = df_phenos.query("phenotypic_category == @model_suffix")
    print(f"Running model on {model_suffix} phenotypes")
    
    
############# STEP 1.1: MAKE SURE THAT EVERY SAMPLE ONLY HAS A SINGLE MIC #############
    
    
# keep only unique MICs. Many samples have MICs tested in different media, so prioritize them according to the model hierarchy and
if not binary:
    # general hierarchy: solid > liquid > plates
    # MABA, Frozen Broth Microdilution Plate (PMID31969421), UKMYC5, UKMYC6, and REMA are plates
    # 7H9 is a liquid media
    media_lst = ["7H10", "LJ", "7H11", "MGIT", "MODS", "BACTEC", "7H9", "Frozen Broth Microdilution Plate (PMID31969421)", "UKMYC6", "UKMYC5", 
                 "REMA", "MYCOTB", "MABA", "MABA24", "MABA48", "non-colourmetric", "M24 BMD"]

    media_hierarchy = dict(zip(media_lst, np.arange(len(media_lst))+1))
    
    # check that no media are missing from either
    if len(set(df_phenos.medium.values) - set(media_hierarchy.keys())) > 0:
        raise ValueError(f"{set(df_phenos.medium.values).symmetric_difference(set(media_hierarchy.keys()))} media are different between df_phenos and media_hierarchy")
    # add media hierarchy to dataframe, sort so that the highest (1) positions come first, then drop duplicates so that every sample has a single MIC
    else:
        df_phenos["media_hierarchy_pos"] = df_phenos["medium"].map(media_hierarchy)
        df_phenos = df_phenos.sort_values("media_hierarchy_pos", ascending=True).drop_duplicates(["sample_id", "mic_value"], keep="first").reset_index(drop=True)
        del df_phenos["media_hierarchy_pos"]
        assert len(df_phenos) == len(df_phenos["sample_id"].unique())
    

############# STEP 2: COMPUTE THE GENETIC RELATEDNESS MATRIX, REMOVING RESISTANCE LOCI #############


model_1_name = "all_model_matrix.pkl"
model_2_name = "model_matrix.pkl"
eigenvec_1_name = "all_model_eigenvecs.pkl"
eigenvec_2_name = "model_eigenvecs.pkl"

model_matrix_1 = pd.read_pickle(os.path.join(out_dir, model_1_name))
model_matrix_2 = pd.read_pickle(os.path.join(out_dir, model_2_name))

eigenvec_df_1 = pd.read_pickle(os.path.join(out_dir, eigenvec_1_name))
eigenvec_df_2 = pd.read_pickle(os.path.join(out_dir, eigenvec_2_name))

model_matrix_1 = model_matrix_1.query("sample_id in @eigenvec_df_1.index.values").sort_values("sample_id", ascending=True)
model_matrix_2 = model_matrix_2.query("sample_id in @eigenvec_df_2.index.values").sort_values("sample_id", ascending=True)

df_phenos = df_phenos.query("sample_id in @eigenvec_df_1.index.values").sort_values("sample_id", ascending=True).reset_index(drop=True)
assert sum(model_matrix_1.merge(eigenvec_df_1, left_index=True, right_index=True).index != df_phenos["sample_id"]) == 0
assert sum(model_matrix_2.merge(eigenvec_df_2, left_index=True, right_index=True).index != df_phenos["sample_id"]) == 0

# concatenate the eigenvectors to the matrix and check the index ordering against the phenotypes matrix
X1 = model_matrix_1.merge(eigenvec_df_1, left_index=True, right_index=True).values
X1 = scaler.fit_transform(X1)
    
# concatenate the eigenvectors to the matrix and check the index ordering against the phenotypes matrix
X2 = model_matrix_2.merge(eigenvec_df_2, left_index=True, right_index=True).values
X2 = scaler.fit_transform(X2)

if binary:
    y = df_phenos["phenotype"].values
else:
    y = np.log(df_phenos["mic_value"].values)

print(f"    {X1.shape[0]} samples and {X1.shape[1]} variables in model 1")
print(f"    {X2.shape[0]} samples and {X2.shape[1]} variables in model 2")


############# STEP 5: FIT L2-PENALIZED REGRESSION #############


if binary:
    model_1 = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
                                 cv=5,
                                 penalty='l2',
                                 max_iter=10000, 
                                 multi_class='ovr',
                                 scoring='neg_log_loss',
                                 class_weight='balanced'
                                )
    model_2 = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
                                 cv=5,
                                 penalty='l2',
                                 max_iter=10000, 
                                 multi_class='ovr',
                                 scoring='neg_log_loss',
                                 class_weight='balanced'
                                )
else:
    model_1 = RidgeCV(alphas=np.logspace(-6, 6, 13),
                    cv=5,
                    scoring='neg_root_mean_squared_error'
                   )
    model_2 = RidgeCV(alphas=np.logspace(-6, 6, 13),
                    cv=5,
                    scoring='neg_root_mean_squared_error'
                   )

model_1.fit(X1, y)
model_2.fit(X2, y)
pickle.dump(model_1, open(os.path.join(out_dir, "model_1.sav"), "wb"))
pickle.dump(model_2, open(os.path.join(out_dir, "model_2.sav"), "wb"))

if binary:
    print(f"    Regularization parameters: {model_1.C_[0]}, {model_2.C_[0]}")
    print(f"    Log Losses: {-model_1.score(X1, y)}, {-model_2.score(X2, y)}")
else:
    print(f"    Regularization parameters: {model_2.alpha_}, {model_2.alpha_}")

    
loss1 = -sklearn.metrics.log_loss(y, model_1.predict_proba(X1), normalize=False)
loss2 = -sklearn.metrics.log_loss(y, model_2.predict_proba(X2), normalize=False)
print(loss1, loss2)
chi_stat = 2 * (loss1 - loss2)
print(f"Chi2 Test Statistic: {chi_stat}")
print(f"p-value: {1 - st.chi2.cdf(x=chi_stat, df=1)}")

# ############# STEP 6: BOOTSTRAP COEFFICIENTS #############

# # use the regularization parameter determined above
# def bootstrap_compute_log_loss(model, X, y):
    
#     logLosses = []
    
#     for i in range(num_bootstrap):

#         # randomly draw sample indices
#         sample_idx = np.random.choice(np.arange(0, len(y)), size=len(y), replace=True)

#         # get the X and y matrices
#         X_bs = X[sample_idx, :]
#         y_bs = y[sample_idx]

#         bs_model = LogisticRegression(C=model.C_[0], penalty='l2', max_iter=10000, multi_class='ovr', class_weight='balanced')
#         bs_model.fit(X_bs, y_bs)
#         logLosses.append(-bs_model.score(X, y))

#     return logLosses


# # save bootstrapped coefficients and principal components
# logLoss_1 = bootstrap_compute_log_loss(model_1, X1, y)
# logLoss_2 = bootstrap_compute_log_loss(model_2, X2, y)

# np.save(os.path.join(out_dir, "model1_logLoss"), logLoss_1)
# np.save(os.path.join(out_dir, "model2_logLoss"), logLoss_2)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")