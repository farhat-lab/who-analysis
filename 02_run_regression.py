import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
import tracemalloc, pickle
who_variants_combined = pd.read_csv("analysis/who_confidence_2021.csv")

# utils files
sys.path.append("utils_files")
from data_utils import *
from stats_utils import *


############# CODE TO MAKE THE COMBINE WHO 2021 VARIANTS + CONFIDENCES FILE #############
# who_variants = pd.read_csv("analysis/who_resistance_variants_all.csv")
# variant_mapping = pd.read_csv("data/v1_to_v2_variants_mapping.csv", usecols=["gene_name", "variant", "raw_variant_mapping_data.variant_category"])
# variant_mapping.columns = ["gene", "V1", "V2"]
# variant_mapping["mutation"] = variant_mapping["gene"] + "_" + variant_mapping["V2"]

# # combine with the new names to get a dataframe with the confidence leve,s and variant mappings between 2021 and 2022
# who_variants_combined = who_variants.merge(variant_mapping[["V1", "mutation"]], left_on="variant", right_on="V1", how="inner")
# del who_variants_combined["variant"]

# # check that they have all the same variants
# assert len(set(who_variants_combined["V1"]).symmetric_difference(set(who_variants["variant"]))) == 0

# del who_variants_combined["genome_index"]
# del who_variants_combined["gene"]
# del who_variants_combined["V1"]

# # some V1 mutations were combined into a single V2 mutation, so they may have multiple confidences listed. Keep the highest confidence instance
# who_variants_combined = who_variants_combined.dropna().sort_values("confidence", ascending=True).drop_duplicates(subset=["drug", "mutation"], keep="first")
# who_variants_combined.to_csv("analysis/who_confidence_2021.csv", index=False)



########################## STEP 0: READ IN PARAMETERS FILE AND GET DIRECTORIES ##########################

    
# starting the memory monitoring
tracemalloc.start()

_, config_file, drug, drug_WHO_abbr = sys.argv

kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
binary = kwargs["binary"]
atu_analysis = kwargs["atu_analysis"]
analysis_dir = kwargs["output_dir"]
alpha = kwargs["alpha"]

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
num_bootstrap = kwargs["num_bootstrap"]

if binary:
    if atu_analysis:
        out_dir = os.path.join(analysis_dir, drug, "ATU", f"tiers={'+'.join(tiers_lst)}", model_prefix)
        
        # the CC and CC-ATU models are in the same folder, but the output files (i.e. regression_coef.csv have different suffixes to distinguish)
        model_suffix = kwargs["atu_analysis_type"]
        assert model_suffix == "CC" or model_suffix == "CC-ATU"
    else:
        out_dir = os.path.join(analysis_dir, drug, "BINARY", f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}", model_prefix)
        model_suffix = ""
else:
    out_dir = os.path.join(analysis_dir, drug, "MIC", f"tiers={'+'.join(tiers_lst)}", model_prefix)
    model_suffix = ""

    
# # model_analysis file was already made, so skip
# if os.path.isfile(os.path.join(out_dir, f"model_analysis{model_suffix}.csv")):
#     print("Regression was already fit. Skipping this analysis")
#     exit()
    

########################## STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES ##########################


# no model (basically just for Pretomanid because there are no WHO phenotypes, so some models don't exist)
if not os.path.isfile(os.path.join(out_dir, "model_matrix.pkl")):
    exit()
else:
    matrix = pd.read_pickle(os.path.join(out_dir, "model_matrix.pkl"))

# matrix = pd.read_pickle(os.path.join(out_dir, "model_matrix_L2.2.1.pkl"))
# model_suffix = "_L2.2.1"
print(matrix.shape)
    
if binary:
    if atu_analysis:
        phenos_file = os.path.join(analysis_dir, drug, "phenos_atu.csv")
    else:
        phenos_file = os.path.join(analysis_dir, drug, "phenos_binary.csv")
else:
    phenos_file = os.path.join(analysis_dir, drug, "phenos_mic.csv")

df_phenos = pd.read_csv(phenos_file)

# replace - with _ for file naming later
if atu_analysis:
    df_phenos = df_phenos.query("phenotypic_category == @model_suffix")
    print(f"Running model on {model_suffix} phenotypes")
    model_suffix = "_" + model_suffix.replace('-', '_')
    
# keep only unique MICs. Many samples have MICs tested in different media, so prioritize them according to the model hierarchy and
if not binary:
    df_phenos = process_multiple_MICs(df_phenos)
    

############# STEP 2: GET THE MATRIX ON WHICH TO FIT THE DATA +/- EIGENVECTOR COORDINATES, DEPENDING ON THE PARAM #############


# read in eigenvectors files, which was previously computed, and keep only the desired number of PCs
eigenvec_df = pd.read_csv("data/eigenvec_100PC.csv", index_col=[0])
keep_PCs = select_PCs_for_model(analysis_dir, drug, pheno_category_lst, eigenvec_df, thresh=0.01)
eigenvec_df = eigenvec_df[keep_PCs]

# keep only samples (rows) that are in matrix and use loc with indices to ensure they are in the same order
matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True, how="inner")
df_phenos = df_phenos.set_index("sample_id")
df_phenos = df_phenos.loc[matrix.index]

# check that the sample ordering is the same in the genotype and phenotype matrices
assert sum(matrix.index != df_phenos.index) == 0
X = matrix.values
    
# scale inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)

# binary vs. quant (MIC) phenotypes
if binary:
    y = df_phenos["phenotype"].values
    assert len(np.unique(y)) == 2
else:
    y = np.log(df_phenos["mic_value"].values)

if len(y) != X.shape[0]:
    raise ValueError(f"Shapes of model inputs {X.shape} and outputs {len(y)} are incompatible")
print(f"{X.shape[0]} samples and {X.shape[1]} variables in the model")


######################### STEP 3: FIT L2-PENALIZED REGRESSION ##########################


if not os.path.isfile(os.path.join(out_dir, f"model{model_suffix}.sav")):
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
                        scoring='neg_root_mean_squared_error'
                       )
    model.fit(X, y)

    # save model because the regularization parameter will be used subsequently
    if not atu_analysis:
        pickle.dump(model, open(os.path.join(out_dir, f"model{model_suffix}.sav"), "wb"))
        
    # save coefficients
    coef_df = pd.DataFrame({"mutation": matrix.columns, "coef": np.squeeze(model.coef_)})
    coef_df.to_csv(os.path.join(out_dir, f"regression_coef{model_suffix}.csv"), index=False)
else:
    model = pickle.load(open(os.path.join(out_dir, f"model{model_suffix}.sav"), "rb"))
    coef_df = pd.read_csv(os.path.join(out_dir, f"regression_coef{model_suffix}.csv"))

if binary:
    print(f"Regularization parameter: {model.C_[0]}")
else:
    print(f"Regularization parameter: {model.alpha_}")
    
    
########################## STEP 4: PERFORM PERMUTATION TEST TO GET SIGNIFICANCE AND BOOTSTRAPPING TO GET ODDS RATIO CONFIDENCE INTERVALS ##########################


if not os.path.isfile(os.path.join(out_dir, f"coef_permutation{model_suffix}.csv")):
    print(f"Peforming permutation test with {num_bootstrap} replicates")
    permute_df = perform_permutation_test(model, X, y, num_bootstrap, binary=binary)
    permute_df.columns = matrix.columns
    permute_df.to_csv(os.path.join(out_dir, f"coef_permutation{model_suffix}.csv"), index=False)
else:
    permute_df = pd.read_csv(os.path.join(out_dir, f"coef_permutation{model_suffix}.csv"))
    

if not os.path.isfile(os.path.join(out_dir, f"coef_bootstrapping{model_suffix}.csv")):
    print(f"Peforming bootstrapping with {num_bootstrap} replicates")
    bootstrap_df = perform_bootstrapping(model, X, y, num_bootstrap, binary=binary, save_summary_stats=False)
    bootstrap_df.columns = matrix.columns
    bootstrap_df.to_csv(os.path.join(out_dir, f"coef_bootstrapping{model_suffix}.csv"), index=False)
else:
    bootstrap_df = pd.read_csv(os.path.join(out_dir, f"coef_bootstrapping{model_suffix}.csv"))

    
########################## STEP 4: ADD PERMUTATION TEST P-VALUES TO THE MAIN COEF DATAFRAME ##########################
    
    
final_df = get_coef_and_confidence_intervals(alpha, binary, who_variants_combined, drug_WHO_abbr, coef_df, permute_df, bootstrap_df)
final_df.sort_values("coef", ascending=False).to_csv(os.path.join(out_dir, f"model_analysis{model_suffix}.csv"), index=False) 

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")