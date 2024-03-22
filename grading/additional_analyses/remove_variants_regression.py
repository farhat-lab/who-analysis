import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys, shutil
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
import tracemalloc, pickle
who_variants_combined = pd.read_csv("analysis/who_confidence_2021.csv")
lineages_combined = pd.read_csv("data/combined_lineages_samples.csv", low_memory=False)

# utils files are in a separate folder
sys.path.append("utils")
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
pool_type = kwargs["pool_type"]
analysis_dir = kwargs["output_dir"]
alpha = kwargs["alpha"]
num_PCs = kwargs["num_PCs"]

# read in the eigenvector dataframe and keep only the PCs for the model
eigenvec_df = pd.read_csv("PCA/eigenvec_100PC.csv", usecols=["sample_id"] + [f"PC{num+1}" for num in np.arange(num_PCs)]).set_index("sample_id")

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


########################## STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES ##########################

    
# no model (basically just for Pretomanid because there are no WHO phenotypes, so some models don't exist)
if not os.path.isfile(os.path.join(out_dir, f"model_matrix{model_suffix}.pkl")):
    exit()
else:
    matrix = pd.read_pickle(os.path.join(out_dir, f"model_matrix{model_suffix}.pkl"))
    
    
if pool_type != "unpooled":
    
    model_unpooled = pd.read_pickle(os.path.join(out_dir.replace(pool_type, "unpooled"), f"model_matrix{model_suffix}.pkl"))

    if len(set(model_unpooled.columns).symmetric_difference(matrix.columns)) == 0:
        print("This pooled model is the same as the corresponding unpooled model. Exiting...")

        # # then copy every file in the corresponding unpooled model to this folder
        # for fName in os.listdir(out_dir.replace(pool_type, "unpooled")):

        #     # ignore the dropped features folder because that is already there from script 01
        #     if os.path.isfile(os.path.join(out_dir.replace(pool_type, "unpooled"), fName)):
        #         shutil.copy(os.path.join(out_dir.replace(pool_type, "unpooled"), fName), os.path.join(out_dir, fName))
        exit()
    
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
    
# keep only unique MICs. Many samples have MICs tested in different media, so prioritize them according to the model hierarchy
if not binary:

    # there are no critical concentrations for Pretomanid, so keep only the most common medium and use un-normalized values because they're all the same medium
    if drug == "Pretomanid":
        pheno_col = "mic_value"
    else:
        pheno_col = "norm_MIC"
    
    cc_df = pd.read_csv("data/drug_CC.csv")

    # first apply the media hierarchy to decide which of the measured MICs to keep for each isolate (for isolates with multiple MICs measured in different media)
    df_phenos = process_multiple_MICs_different_media(df_phenos)

    # then normalize the MICs to the most common medium. This creates a new column: norm_MIC that should be used if not Pretomanid
    df_phenos, most_common_medium = normalize_MICs_return_dataframe(drug, df_phenos, cc_df)
    print(f"    Min MIC: {np.min(df_phenos[pheno_col].values)}, Max MIC: {np.max(df_phenos[pheno_col].values)} in {most_common_medium}")

model_suffix = '_no_inhA_promoter_variant'

if os.path.isfile(os.path.join(out_dir, f"model_analysis{model_suffix}.csv")):
    print("Regression was already run for this model")
    exit()
else:
    print(f"Saving reuslts to {out_dir}")
    

############# STEP 2: GET THE MATRIX ON WHICH TO FIT THE DATA +/- EIGENVECTOR COORDINATES, DEPENDING ON THE PARAM #############


print(f"{matrix.shape[0]} samples and {matrix.shape[1]} genotypic features in the model")
matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True, how="inner")
print(f"Data shape before removing any variants: {matrix.shape}")
assert 'inhA_c.-770T>C' in matrix.columns

# remove anything that is 0 everywhere
matrix = matrix.drop('inhA_c.-770T>C', axis=1)
print(f"Data shape after removing variants: {matrix.shape}")
assert 'inhA_c.-770T>C' not in matrix.columns

# keep only samples (rows) that are in matrix and use loc with indices to ensure they are in the same order
df_phenos = df_phenos.set_index("sample_id")
df_phenos = df_phenos.loc[matrix.index]

# check that the sample ordering is the same in the genotype and phenotype matrices
assert sum(matrix.index != df_phenos.index) == 0
scaler = StandardScaler()

# scale values because input matrix and PCA matrix are on slightly different scales
X = scaler.fit_transform(matrix.values)

# binary vs. quant (MIC) phenotypes
if binary:
    y = df_phenos["phenotype"].values
    assert len(np.unique(y)) == 2
else:
    y = np.log2(df_phenos[pheno_col].values)

if len(y) != X.shape[0]:
    raise ValueError(f"Shapes of model inputs {X.shape} and outputs {len(y)} are incompatible")

print(f"{X.shape[0]} samples and {X.shape[1]} variables in the model")


######################### STEP 3: FIT L2-PENALIZED REGRESSION ##########################


if not os.path.isfile(os.path.join(out_dir, "model_no_inhA_promoter_variant.sav")):
    if binary:
        model = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
                                     cv=5,
                                     penalty='l2',
                                     max_iter=100000, 
                                     multi_class='ovr',
                                     scoring='neg_log_loss',
                                     class_weight='balanced',
                                     n_jobs=-1
                                    )


    else:
        model = RidgeCV(alphas=np.logspace(-6, 6, 13),
                        cv=5,
                        scoring='neg_root_mean_squared_error',
                       )
    model.fit(X, y)
    pickle.dump(model, open(os.path.join(out_dir, "model_no_inhA_promoter_variant.sav"), "wb"))
else:
    model = pickle.load(open(os.path.join(out_dir, "model_no_inhA_promoter_variant.sav"), "rb"))

# save coefficients
if not os.path.isfile(os.path.join(out_dir, f"regression_coef{model_suffix}.csv")):
    coef_df = pd.DataFrame({"mutation": matrix.columns, "coef": np.squeeze(model.coef_)})
    coef_df.to_csv(os.path.join(out_dir, f"regression_coef{model_suffix}.csv"), index=False)
else:
    coef_df = pd.read_csv(os.path.join(out_dir, f"regression_coef{model_suffix}.csv"))
    
if binary:
    print(f"Regularization parameter: {model.C_[0]}")
else:
    print(f"Regularization parameter: {model.alpha_}")
    
    
########################## STEP 4: PERFORM PERMUTATION TEST TO GET SIGNIFICANCE AND BOOTSTRAPPING TO GET ODDS RATIO CONFIDENCE INTERVALS ##########################


if not os.path.isfile(os.path.join(out_dir, f"coef_permutation{model_suffix}.csv")):
    print(f"Peforming permutation test with {num_bootstrap} replicates")
    permute_df = perform_permutation_test(model, X, y, num_bootstrap, binary=binary, fit_type='OLS', progress_bar=False)
    permute_df.columns = matrix.columns
    permute_df.to_csv(os.path.join(out_dir, f"coef_permutation{model_suffix}.csv"), index=False)
else:
    permute_df = pd.read_csv(os.path.join(out_dir, f"coef_permutation{model_suffix}.csv"))

# if not os.path.isfile(os.path.join(out_dir, f"coef_bootstrapping{model_suffix}.csv")):
#     print(f"Peforming bootstrapping with {num_bootstrap} replicates")
#     bootstrap_df = perform_bootstrapping(model, X, y, num_bootstrap, binary=binary)
#     bootstrap_df.columns = matrix.columns
#     bootstrap_df.to_csv(os.path.join(out_dir, f"coef_bootstrapping{model_suffix}.csv"), index=False)
# else:
#     bootstrap_df = pd.read_csv(os.path.join(out_dir, f"coef_bootstrapping{model_suffix}.csv"))

    
########################## STEP 4: ADD PERMUTATION TEST P-VALUES TO THE MAIN COEF DATAFRAME ##########################
    
    
final_df = get_coef_and_confidence_intervals(alpha, binary, who_variants_combined, drug_WHO_abbr, coef_df, permute_df, bootstrap_df=None)
final_df.sort_values("coef", ascending=False).to_csv(os.path.join(out_dir, f"model_analysis{model_suffix}.csv"), index=False) 

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")