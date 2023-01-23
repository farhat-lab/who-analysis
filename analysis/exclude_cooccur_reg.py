import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
import scipy.stats as st
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
import tracemalloc, pickle

# analysis utils is in the analysis folder
sys.path.append(os.path.join(os.getcwd(), "analysis"))
from stats_utils import *

who_variants_combined = pd.read_csv("analysis/who_confidence_2021.csv")

_, config_file, drug, drug_WHO_abbr = sys.argv

kwargs = yaml.safe_load(open(config_file))

analysis_dir = kwargs["output_dir"]
synonymous = kwargs["synonymous"]
tiers_lst = ["1", "2"]
num_PCs = kwargs["num_PCs"]
# num_bootstrap = kwargs["num_bootstrap"]
num_bootstrap = 100

pheno_category_lst = kwargs["pheno_category_lst"]
# make sure that both phenotypes are included
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
    pheno_category_lst = ["ALL", "WHO"]
else:
    phenos_name = "WHO"

scaler = StandardScaler()

if not os.path.isdir(os.path.join(analysis_dir, drug, "BINARY/exclude_comutation")):
    os.makedirs(os.path.join(analysis_dir, drug, "BINARY/exclude_comutation"))


######################## STEP 1: GET LIST OF SAMPLES WITH RESISTANCE-ASSOCIATED MUTATIONS ########################


# RESISTANCE-ASSOCIATED = CATEGORY 1 OR 2 MUTATION FROM THE 2021 CATALOG 
# get all Tier 2 mutations that are significant in the first round of analysis
tier2_mutations_of_interest = get_tier2_mutations_of_interest(analysis_dir, drug, phenos_name)

# get all samples with high confidence resistance-associated mutations
if not os.path.isfile(os.path.join(analysis_dir, f"{drug}/samples_highConf_tier1.npy")):
    
    # get dataframe of samples that have tier 1 mutations
    genos_1 = pd.read_csv(os.path.join(analysis_dir, drug, "genos_1.csv.gz"), compression="gzip",
                          usecols=["sample_id", "resolved_symbol", "variant_category", "variant_binary_status"]
                         ).query("variant_binary_status == 1")
    
    del genos_1["variant_binary_status"]
    genos_1["mutation"] = genos_1["resolved_symbol"] + "_" + genos_1["variant_category"]
    del genos_1["resolved_symbol"]
    del genos_1["variant_category"]
    
    # get the number of high confidence resistance-associated 
    highConf_mut = who_variants_combined.loc[(who_variants_combined["drug"]==drug_WHO_abbr) & 
                                             (who_variants_combined["confidence"].str.contains("|".join(["1", "2"])))
                                            ].mutation.unique()
    
    samples_highConf_tier1 = genos_1.query("mutation in @highConf_mut")["sample_id"].unique()
    np.save(os.path.join(analysis_dir, f"{drug}/samples_highConf_tier1"), samples_highConf_tier1)
    
else:
    samples_highConf_tier1 = np.load(os.path.join(analysis_dir, f"{drug}/samples_highConf_tier1.npy"))

    
######################## STEP 2: GET DATAFRAME OF SAMPLES THAT HAVE ANY TIER 2 MUTATIONS ########################


### TODO: WRITE THIS IN SUBPROCESS. RIGHT NOW, THE FOLLOWING LINE NEEDS TO BE RUN ON THE COMMAND LINE BEFORE RUNNING THIS SCRIPT
# command = f'gunzip -c /n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue/{drug}/genos_2.csv.gz | awk -F "," ' + 'NR==1 || $6 == 1 {print > /n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue/Rifampicin/positive_tier2_genos.csv"}' 

# subprocess.run(command, shell=True)

# read in the dataframe of samples that contain Tier 2 mutations (contains all columns as normal)
positive_tier2_genos = pd.read_csv(os.path.join(analysis_dir, drug, "positive_tier2_genos.csv"))
positive_tier2_genos["mutation"] = positive_tier2_genos["resolved_symbol"] + "_" + positive_tier2_genos["variant_category"]

# get all samples that have high confidence Tier 1 mutations and the mutation of interest
exclude_samples = set(samples_highConf_tier1).intersection(positive_tier2_genos.query("mutation in @tier2_mutations_of_interest").sample_id)
print(f"{len(exclude_samples)} total samples to exclude")


######################## STEP 3: GET MODEL MATRIX ########################


matrix_file = os.path.join(analysis_dir, drug, f"BINARY/exclude_comutation/model_matrix.pkl")

if not os.path.isfile(matrix_file):
    
    print("Creating model matrix...")

    # read in only the genotypes files for the tiers for this model
    df_model = pd.concat([pd.read_csv(os.path.join(analysis_dir, drug, f"genos_{num}.csv.gz"), compression="gzip", low_memory=False) for num in tiers_lst])

    print(f"{len(df_model.sample_id.unique())} samples before exclusion")
    df_model = df_model.query("sample_id not in @exclude_samples")
    print(f"{len(df_model.sample_id.unique())} samples after exclusion")

    # drop synonymous variants, unless the boolean is True
    if not synonymous:
        df_model = df_model.query("predicted_effect not in ['synonymous_variant', 'stop_retained_variant', 'initiator_codon_variant']").reset_index(drop=True)

    # drop any samples with ambiguous allele fractions
    drop_isolates = df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] <= 0.75)].sample_id.unique()
    df_model = df_model.query("sample_id not in @drop_isolates")    

    # 1 = alternative allele, 0 = reference allele, NaN = missing
    df_model["mutation"] = df_model["resolved_symbol"] + "_" + df_model["variant_category"]

    # drop more duplicates, but I think this might be because we have multiple data pulls at a time
    # NaN is larger than any number, so sort ascending and keep first
    df_model = df_model.sort_values("variant_binary_status", ascending=True).drop_duplicates(["sample_id", "mutation"], keep="first")
    matrix = df_model.pivot(index="sample_id", columns="mutation", values="variant_binary_status")
    del df_model

    # drop any isolate with missingness, then remove features with no signal
    matrix = matrix.dropna(axis=0, how="any")
    matrix = matrix[matrix.columns[~((matrix == 0).all())]]


    ############ STEP 4: PREPARE MODEL INPUTS ############


    # Read in the PC coordinates dataframe, then keep only the desired number of principal components
    eigenvec_df = pd.read_csv("data/eigenvec_10PC.csv", index_col=[0]).iloc[:, :num_PCs]
    eigenvec_df = eigenvec_df.loc[matrix.index]

    # merge the eigenvectors to the matrix and check the index ordering against the phenotypes matrix
    matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True)
    matrix.to_pickle(matrix_file)
    
else:
    matrix = pd.read_pickle(matrix_file)


# read in the phenotypes dataframe and keep only desired phenotypes
df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv"))
df_phenos = df_phenos.query("phenotypic_category in @pheno_category_lst & sample_id not in @exclude_samples")
    
# then get the corresponding inputs
matrix = matrix.reset_index().query("sample_id in @df_phenos.sample_id").set_index("sample_id")
df_phenos = df_phenos.set_index("sample_id").loc[matrix.index]
assert sum(matrix.index != df_phenos.index.values) == 0
print(matrix.shape, df_phenos.shape)

# drop any more features with no signal, after only isolates of the desired phenotype have been kept
matrix = matrix[matrix.columns[~((matrix == 0).all())]]
print(matrix.shape)


############# STEP 5: FIT L2-PENALIZED REGRESSION #############


def run_regression(X, y):

    model = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
                                  cv=5,
                                  penalty='l2',
                                  max_iter=10000, 
                                  multi_class='ovr',
                                  scoring='neg_log_loss',
                                  class_weight='balanced'
                                 )

    
    model.fit(X, y)
    
    # get positive class probabilities and predicted classes after determining the binarization threshold
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = get_threshold_val_and_classes(y_prob, y)
    log_like = -sklearn.metrics.log_loss(y_true=y, y_pred=y_prob, normalize=False)
        
    print([model.C_[0], 
            log_like,
            sklearn.metrics.roc_auc_score(y_true=y, y_score=y_prob),
            sklearn.metrics.accuracy_score(y_true=y, y_pred=y_pred),
            sklearn.metrics.balanced_accuracy_score(y_true=y, y_pred=y_pred),
           ])
    
    return model


X = scaler.fit_transform(matrix.values)
y = df_phenos["phenotype"].values
model = run_regression(X, y)

res_df = pd.DataFrame({"mutation": matrix.columns, 'coef': np.squeeze(model.coef_)})
res_df.to_csv(os.path.join(analysis_dir, drug, f"BINARY/exclude_comutation/{phenos_name}phenos_coef.csv"), index=False)


############# STEP 6: BOOTSTRAP THE COEFFICIENTS #############


# use the regularization parameter determined above
print(f"Bootstrapping with {num_bootstrap} replicates")
coef_df = bootstrap_coef(model, X, y, num_bootstrap, binary=True)
coef_df.columns = matrix.columns
coef_df.to_csv(os.path.join(analysis_dir, drug, f"BINARY/exclude_comutation/{phenos_name}phenos_bootstrap.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB")