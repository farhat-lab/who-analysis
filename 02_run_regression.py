import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
import tracemalloc, pickle
who_variants_combined = pd.read_csv("analysis/who_confidence_2021.csv")

# analysis utils is in the analysis folder
sys.path.append(os.path.join(os.getcwd(), "analysis"))
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
        model_suffix = ""
else:
    out_dir = os.path.join(analysis_dir, drug, "MIC", f"tiers={'+'.join(tiers_lst)}", model_prefix)
    model_suffix = ""
    

########################## STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES ##########################


# no model (basically just for Pretomanid because there are no WHO phenotypes, so some models don't exist)
if not os.path.isfile(os.path.join(out_dir, "model_matrix.pkl")):
    exit()
else:
    model_inputs = pd.read_pickle(os.path.join(out_dir, "model_matrix.pkl"))
    
if binary:
    if atu_analysis:
        phenos_file = os.path.join(analysis_dir, drug, "phenos_atu.csv")
    else:
        phenos_file = os.path.join(analysis_dir, drug, "phenos_binary.csv")
else:
    phenos_file = os.path.join(analysis_dir, drug, "phenos_mic.csv")

df_phenos = pd.read_csv(phenos_file).set_index("sample_id")

# replace - with _ for file naming later
if atu_analysis:
    df_phenos = df_phenos.query("phenotypic_category == @model_suffix")
    print(f"Running model on {model_suffix} phenotypes")
    model_suffix = "_" + model_suffix.replace('-', '_')
    
    
########################## STEP 1.1: MAKE SURE THAT EVERY SAMPLE ONLY HAS A SINGLE MIC ##########################
    
    
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
    

############# STEP 2: GET THE MATRIX ON WHICH TO FIT THE DATA +/- EIGENVECTOR COORDINATES, DEPENDING ON THE PARAM #############


if num_PCs > 0:
    # read in eigenvectors files, which was previously computed, and keep only the desired number of PCs
    eigenvec_df = pd.read_csv("data/eigenvec_10PC.csv", index_col=[0]).iloc[:, :num_PCs]
    
    # keep only the samples that are in this model, then concatenate the eigenvectors to the matrix
    eigenvec_df = eigenvec_df.loc[model_inputs.index]
    model_inputs = model_inputs.merge(eigenvec_df, left_index=True, right_index=True)

# keep only samples (rows) that are in model_inputs
df_phenos = df_phenos.loc[model_inputs.index]

# check that the sample ordering is the same in the genotype and phenotype matrices
assert sum(model_inputs.index != df_phenos.index) == 0
X = model_inputs.values
    
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


########################## STEP 3: FIT L2-PENALIZED REGRESSION ##########################


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

if binary:
    print(f"Regularization parameter: {model.C_[0]}")
else:
    print(f"Regularization parameter: {model.alpha_}")
    
# save coefficients
coef_df = pd.DataFrame({"mutation": model_inputs.columns, "coef": np.squeeze(model.coef_)})
coef_df.to_csv(os.path.join(out_dir, f"regression_coef{model_suffix}.csv"), index=False)
   
    
########################## STEP 4: PERFORM PERMUTATION TEST TO GET SIGNIFICANCE AND BOOTSTRAPPING TO GET ODDS RATIO CONFIDENCE INTERVALS ##########################


print(f"Peforming permutation test with {num_bootstrap} replicates")
permute_df = perform_permutation_test(model, X, y, num_bootstrap, binary=binary)
permute_df.columns = model_inputs.columns
permute_df.to_csv(os.path.join(out_dir, f"coef_permutation{model_suffix}.csv"), index=False)

print(f"Peforming bootstrapping with {num_bootstrap} replicates")
bootstrap_df = perform_bootstrapping(model, X, y, num_bootstrap, binary=binary, save_summary_stats=False)
bootstrap_df.columns = model_inputs.columns
bootstrap_df.to_csv(os.path.join(out_dir, f"coef_bootstrapping{model_suffix}.csv"), index=False)

    
########################## STEP 4: ADD PERMUTATION TEST P-VALUES TO THE MAIN COEF DATAFRAME ##########################
    

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
    if row["coef"] > 0:
        coef_df.loc[i, "pval"] = np.mean(permute_df[row["mutation"]] >= row["coef"])
    else:
        coef_df.loc[i, "pval"] = np.mean(permute_df[row["mutation"]] <= row["coef"])
        
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
final_df.sort_values("coef", ascending=False).to_csv(os.path.join(out_dir, f"model_analysis{model_suffix}.csv"), index=False)        
        
# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")