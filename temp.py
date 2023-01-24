import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import binomtest
import sys, glob, os, yaml, tracemalloc

analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'

# analysis utils is in the analysis folder
sys.path.append(os.path.join(os.getcwd(), "analysis"))
from stats_utils import *

# starting the memory monitoring
tracemalloc.start()


_, config_file, drug = sys.argv

kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
analysis_dir = kwargs["output_dir"]

model_prefix = kwargs["model_prefix"]
num_PCs = kwargs["num_PCs"]

pheno_category_lst = kwargs["pheno_category_lst"]
# make sure that both phenotypes are included
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
    pheno_category_lst = ["ALL", "WHO"]
else:
    phenos_name = "WHO"

scaler = StandardScaler()

if not os.path.isdir(os.path.join(analysis_dir, drug, "BINARY/LRT")):
    os.makedirs(os.path.join(analysis_dir, drug, "BINARY/LRT"))
    

############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############


phenos_file = os.path.join(analysis_dir, drug, "phenos_binary.csv")
df_phenos = pd.read_csv(phenos_file).set_index("sample_id")

# different matrices, depending on the phenotypes. Get all mutation 
matrix = pd.read_pickle(os.path.join(analysis_dir, drug, "BINARY", f"tiers=1+2/phenos={phenos_name}", model_prefix, "model_matrix.pkl"))
mutations_lst = matrix.columns

# Read in the PC coordinates dataframe, then keep only the desired number of principal components
eigenvec_df = pd.read_csv("data/eigenvec_10PC.csv", index_col=[0]).iloc[:, :num_PCs]

# concatenate the eigenvectors to the matrix and check the index ordering against the phenotypes matrix
matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True)
print(f"{matrix.shape[0]} samples and {matrix.shape[1]} variables in the largest {phenos_name} model")

df_phenos = df_phenos.loc[matrix.index]
assert sum(matrix.index != df_phenos.index.values) == 0
y = df_phenos["phenotype"].values


############# STEP 2: READ IN THE ORIGINAL DATA: MODEL_MATRIX PICKLE FILE FOR A GIVEN MODEL #############

    
def remove_single_mut(matrix, mutation):
    
    if mutation not in matrix.columns:
        raise ValueError(f"{mutation} is not in the argument matrix!")
    
    small_matrix = matrix.loc[:, matrix.columns != mutation]
    assert small_matrix.shape[1] + 1 == matrix.shape[1]
    return small_matrix


X = remove_single_mut(matrix, "rpoB_p.Ser450Leu").values
X = scaler.fit_transform(X)
print(X.shape)

model = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
                              cv=5,
                              penalty='l2',
                              max_iter=10000, 
                              multi_class='ovr',
                              scoring='neg_log_loss',
                              class_weight='balanced'
                             )

model.fit(X, y)
print(model.C_[0])

# def get_annotated_genos(analysis_dir, drug):
#     '''
#     This function gets the phenotype and genotype dataframes for a given drug. These are the same across models of the same type (i.e. all BINARY with WHO phenotypes, or all ATU models). 
#     '''
 
#     genos_files = glob.glob(os.path.join(analysis_dir, drug, "genos*.csv.gz"))
#     print(f"{len(genos_files)} genotypes files")
#     # genos_files = [os.path.join(analysis_dir, drug, "genos_1.csv.gz")]
#     df_genos = pd.concat([pd.read_csv(fName, compression="gzip", low_memory=False, 
#                                       usecols=["resolved_symbol", "variant_category", "predicted_effect", "position"]
#                                      ) for fName in genos_files]).drop_duplicates()
    
#     # get annotations for mutations to combine later. Exclude lof and inframe, these will be manually replaced later
#     df_genos["mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
#     annotated_genos = df_genos.drop_duplicates(["mutation", "predicted_effect", "position"])[["mutation", "predicted_effect", "position"]]
#     del df_genos
    
#     return annotated_genos



# def compute_statistics_single_model(res_df, df_phenos, matrix, annotated_genos, alpha=0.05):
    
#     # pivot to matrix and add sample IDs and phenotypes to the matrix
#     matrix = matrix.merge(df_phenos[["sample_id", "phenotype"]], left_index=True, right_on="sample_id").reset_index(drop=True)
#     print(matrix.shape)
    
#     # coefficient dictionary to keep track of which variants have positive and negative coefficients
#     variant_coef_dict = dict(zip(res_df["mutation"], res_df["coef"]))

#     # get dataframe of the univariate stats add them to the results dataframe
#     full_predict_values = compute_univariate_stats(matrix[matrix.columns[~matrix.columns.str.contains("PC")]], variant_coef_dict)
#     res_df = res_df.merge(full_predict_values, on="mutation", how="outer").drop_duplicates("mutation", keep="first")
        
#     # add confidence intervals for all stats except the likelihood ratios
#     try:
#         res_df = compute_exact_confidence_intervals(res_df, alpha)
#     except:
#         print("Confidence intervals failed")

#     # add confidence intervals for the likelihood ratios
#     try:
#         res_df = compute_likelihood_ratio_confidence_intervals(res_df, alpha)
#     except:
#         print("LR confidence intervals failed")
        
#     # get effect annotations and merge them with the results dataframe
#     res_df = res_df.merge(annotated_genos, on="mutation", how="outer")
#     res_df = res_df.loc[~pd.isnull(res_df["coef"])]
        
#     # check that every mutation is present in at least 1 isolate
#     if res_df.Num_Isolates.min() < 1:
#         print("Some total numbers of isolates are less than 1")
    
#     return res_df
    
    
    
# drug = "Rifampicin"
# phenos_name = "WHO"
# # make sure that both phenotypes are included
# if phenos_name == "ALL":
#     pheno_category_lst = ["ALL", "WHO"]
# else:
#     pheno_category_lst = ["WHO"]

# # get all Tier 2 mutations that are significant in the first round of analysis
# tier2_mutations_of_interest = get_tier2_mutations_of_interest(analysis_dir, drug, phenos_name)

# # get all samples with high confidence resistance-associated mutations
# samples_highConf_tier1 = np.load(os.path.join(analysis_dir, f"{drug}/samples_highConf_tier1.npy"))

# # read in the dataframe of samples that contain Tier 2 mutations (contains all columns as normal)
# positive_tier2_genos = pd.read_csv(os.path.join(analysis_dir, drug, "positive_tier2_genos.csv"))
# positive_tier2_genos["mutation"] = positive_tier2_genos["resolved_symbol"] + "_" + positive_tier2_genos["variant_category"]

# # get all samples that have high confidence Tier 1 mutations and the mutation of interest
# exclude_samples = set(samples_highConf_tier1).intersection(positive_tier2_genos.query("mutation in @tier2_mutations_of_interest").sample_id)


# ###############################################################


# res_df = pd.read_csv(os.path.join(analysis_dir, drug, f"BINARY/exclude_comutation/{phenos_name}phenos_coef.csv"))
# bootstrap_df = pd.read_csv(os.path.join(analysis_dir, drug, f"BINARY/exclude_comutation/{phenos_name}phenos_bootstrap.csv"))

# # read in model matrix
# matrix = pd.read_pickle(os.path.join(analysis_dir, drug, f"BINARY/exclude_comutation/model_matrix.pkl"))
# num_samples = len(matrix)

# res_df = get_pvalues_add_ci(res_df, bootstrap_df, "mutation", num_samples, alpha=0.05)
# del bootstrap_df
# res_df = BH_FDR_correction(res_df)
# res_df["Bonferroni_pval"] = np.min([res_df["pval"] * len(res_df["pval"]), np.ones(len(res_df["pval"]))], axis=0)


# ###############################################################


# # read in the phenotypes dataframe and keep only desired phenotypes
# df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv"))
# df_phenos = df_phenos.query("phenotypic_category in @pheno_category_lst & sample_id not in @exclude_samples")
    
# # then get the corresponding inputs, drop any more features with no signal, after only isolates of the desired phenotype have been kept
# matrix = matrix.reset_index().query("sample_id in @df_phenos.sample_id").set_index("sample_id")
# matrix = matrix[matrix.columns[~((matrix == 0).all())]]
# print(matrix.shape)


# ###############################################################


# annotated_genos = get_annotated_genos(analysis_dir, drug)
# print("Finished getting annotated genos file")

# res_df = compute_statistics_single_model(res_df, df_phenos, matrix, annotated_genos, alpha=0.05)
# res_df.to_csv(os.path.join(analysis_dir, drug, f"BINARY/exclude_comutation/{phenos_name}phenos_univariate_stats.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")