import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
import tracemalloc, pickle
who_variants_combined = pd.read_csv("analysis/who_confidence_2021.csv")
lineages = pd.read_csv("data/lineages.csv", low_memory=False)

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

_, config_file, drug, drug_WHO_abbr, lineage = sys.argv

kwargs = yaml.safe_load(open(config_file))
binary = kwargs["binary"]
tiers_lst = kwargs["tiers_lst"]
synonymous = kwargs["synonymous"]
alpha = kwargs["alpha"]
model_prefix = kwargs["model_prefix"]
pheno_category_lst = kwargs["pheno_category_lst"]
atu_analysis = kwargs["atu_analysis"]
atu_analysis_type = kwargs["atu_analysis_type"]
analysis_dir = kwargs["output_dir"]
num_PCs = kwargs["num_PCs"]
num_bootstrap = kwargs["num_bootstrap"]

if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
else:
    phenos_name = "WHO"

scaler = StandardScaler()
    
out_dir = os.path.join(analysis_dir, drug, f"BINARY/lineage_models/L{lineage}")

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)


########################## STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES ##########################


# no model (basically just for Pretomanid because there are no WHO phenotypes, so some models don't exist)
matrix = pd.read_pickle(os.path.join(analysis_dir, drug, "BINARY", f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}", model_prefix, "model_matrix.pkl"))
df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv")).set_index("sample_id")

# read in eigenvectors files, which was previously computed, and keep only the desired number of PCs
eigenvec_df = pd.read_csv("data/eigenvec_10PC.csv", index_col=[0]).iloc[:, :num_PCs]

# keep only the samples that are in this model, then concatenate the eigenvectors to the matrix
eigenvec_df = eigenvec_df.loc[matrix.index]
matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True, how="inner")
print(matrix.shape)

# keep only the specified lineage
if len(lineage) == 1:
    lineage_samples = lineages.query("Primary_Lineage == @lineage")["Sample_ID"].unique()
else:
    lineage_samples = []

    for i, row in lineages.iterrows():
        # if lineage in row["Lineage"].split(","):
        if "2.2.1" in row["Lineage"].split(",") or "2.2.1.1" in row["Lineage"].split(",") or "2.2.1.2" in row["Lineage"].split(","):
            lineage_samples.append(row["Sample_ID"])
    lineage_samples = np.unique(lineage_samples)
        
del lineages
matrix = matrix.loc[matrix.index.isin(lineage_samples)]
matrix = matrix[matrix.columns[~((matrix == 0).all())]]

# keep only samples (rows) that are in matrix
df_phenos = df_phenos.loc[matrix.index]

# check that the sample ordering is the same in the genotype and phenotype matrices
assert sum(matrix.index != df_phenos.index) == 0
X = scaler.fit_transform(matrix.values)
y = df_phenos["phenotype"].values
assert len(np.unique(y)) == 2
print(f"{X.shape[0]} samples and {X.shape[1]} variables in the L{lineage} model")

model = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
                             cv=5,
                             penalty='l2',
                             max_iter=10000, 
                             multi_class='ovr',
                             scoring='neg_log_loss',
                             class_weight='balanced'
                            )

model.fit(X, y)
print(f"Regularization parameter: {model.C_[0]}")

# save coefficients
coef_df = pd.DataFrame({"mutation": matrix.columns, "coef": np.squeeze(model.coef_)})
coef_df.to_csv(os.path.join(out_dir, "regression_coef.csv"), index=False)
    
print(f"Peforming permutation test with {num_bootstrap} replicates")
permute_df = perform_permutation_test(model, X, y, num_bootstrap, binary=binary)
permute_df.columns = matrix.columns
permute_df.to_csv(os.path.join(out_dir, "coef_permutation.csv"), index=False)

print(f"Peforming bootstrapping with {num_bootstrap} replicates")
bootstrap_df = perform_bootstrapping(model, X, y, num_bootstrap, binary=binary, save_summary_stats=False)
bootstrap_df.columns = matrix.columns
bootstrap_df.to_csv(os.path.join(out_dir, "coef_bootstrapping.csv"), index=False)

    
########################## STEP 4: ADD PERMUTATION TEST P-VALUES TO THE MAIN COEF DATAFRAME ##########################
    
    
final_df = get_coef_and_confidence_intervals(alpha, binary, who_variants_combined, drug_WHO_abbr, coef_df, permute_df, bootstrap_df)
final_df.sort_values("coef", ascending=False).to_csv(os.path.join(out_dir, "model_analysis.csv"), index=False) 

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")