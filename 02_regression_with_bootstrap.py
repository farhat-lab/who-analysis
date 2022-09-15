import numpy as np
import pandas as pd
import glob, os, yaml, subprocess, sparse, sys
from Bio import SeqIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import warnings
warnings.filterwarnings("ignore")


############# STEP 0: READ IN PARAMETERS FILE AND GET DIRECTORIES #############


_, config_file = sys.argv

kwargs = yaml.safe_load(open(config_file))

drug = kwargs["drug"]
out_dir = kwargs["out_dir"]
model_prefix = kwargs["model_prefix"].split(".")[0]
num_PCs = kwargs["num_PCs"]
MAF = kwargs["MAF"]
num_bootstrap = kwargs["num_bootstrap"]
impute = kwargs["impute"]

# list of directories
tidy_dir = "/n/data1/hms/dbmi/farhat/ye12/who/narrow_format"
phenos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/phenotypes'
phenos_dir = os.path.join(phenos_dir, f"drug_name={drug}")


############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############


model_inputs = pd.read_pickle(os.path.join(out_dir, drug, model_prefix, "filt_matrix.pkl"))
print(f"Model matrix shape: {model_inputs.shape}")

# reset index so that sample_id is now a column, which makes slicing easier
model_inputs = model_inputs.reset_index()

df_phenos = pd.read_csv(os.path.join(out_dir, drug, model_prefix, "phenos.csv"))


############# STEP 2: COMPUTE THE GENETIC RELATEDNESS MATRIX #############


if num_PCs > 0:
    print(f"Fitting regression with population structure correction with {num_PCs} principal components")

    # read in all dataframes in the narrow directory
    print("Reading in dataframes for the genotypes matrix...")
    tidy_dfs_lst = [pd.read_csv(os.path.join(tidy_dir, fName)) for fName in os.listdir(tidy_dir)]

    # concatenate into a single dataframe
    tidy_combined = pd.concat(tidy_dfs_lst, axis=0)

    # H37Rv full genome genbank file
    genome = SeqIO.read("/n/data1/hms/dbmi/farhat/Sanjana/GCF_000195955.2_ASM19595v2_genomic.gbff", "genbank")
    print(len(genome.seq))

    # make dataframe mapping positions to reference nucleotides. This is needed to get the reference allele at every site and compare to the data
    pos_unique = tidy_combined.position.unique()
    ref_dict = dict(zip(pos_unique, [genome.seq[int(pos_unique[0]-1)] for pos in pos_unique]))

    tidy_combined["ref"] = tidy_combined["position"].map(ref_dict)
    assert sum(pd.isnull(tidy_combined.ref)) == 0

    # keep only samples that will be used for the model
    tidy_combined = tidy_combined.query("sample_id in @model_inputs.sample_id.values")

    # boolean of whether the reference and actual alleles are different
    tidy_combined["diff"] = (tidy_combined["ref"] != tidy_combined["nucleotide"]).astype(int)
    tidy_matrix = tidy_combined.pivot(index="sample_id", columns="position", values="diff").fillna(0)

    # should only be 1s and 0s
    assert len(np.unique(tidy_matrix.values)) == 2
    assert 1 in np.unique(tidy_matrix.values)
    assert 0 in np.unique(tidy_matrix.values)

    # check again that positions and isolates are unique
    assert len(np.unique(tidy_matrix.index.values)) == len(tidy_matrix.index)
    assert len(np.unique(tidy_matrix.columns)) == len(tidy_matrix.columns)

    # MAF filtering and save GRM
    tidy_matrix = tidy_matrix[tidy_matrix.columns[tidy_matrix.mean(axis=0) >= MAF]]

    grm = np.cov(tidy_matrix.values)
    grm_df = pd.DataFrame(grm)
    grm_df.columns = tidy_matrix.index.values
    grm_df.index = tidy_matrix.index.values
    grm_df.to_pickle(os.path.join(out_dir, drug, model_prefix, "GRM.pkl"))
    
    
############# STEP 3: RUN PCA ON THE GRM #############


    pca = PCA(n_components=num_PCs)
    pca.fit(grm)

    print(f"Explained variance of {num_PCs} principal components: {pca.explained_variance_}")
    print("Saving eigenvectors...")
    eigenvec = pca.components_.T
    eigenvec_df = pd.DataFrame(eigenvec)
    eigenvec_df["sample_id"] = tidy_matrix.index.values
    eigenvec_df.to_csv(os.path.join(out_dir, drug, model_prefix, f"PC_{num_PCs}.csv"), index=False)

    
############# STEP 4: PREPARE INPUTS TO THE MODEL #############


    # drop any samples from the phenotypes and genotypes dataframes that are not in the eigenvector dataframe (some samples may not have genotypes)
    # reorder the phenotypes, genotypes, and eigevector dataframes so that they are all in the same sample order
    df_phenos = df_phenos.query("sample_id in @eigenvec_df.sample_id.values").sort_values("sample_id", ascending=True).reset_index(drop=True)
    model_inputs = model_inputs.query("sample_id in @eigenvec_df.sample_id.values").sort_values("sample_id", ascending=True).reset_index(drop=True)
    eigenvec_df = eigenvec_df.sort_values("sample_id", ascending=True).reset_index(drop=True)

    assert len(df_phenos) == len(eigenvec_df) == len(model_inputs)

    # set index for these 2 dataframes so that later only the values can be extracted
    model_inputs = model_inputs.set_index("sample_id")
    eigenvec_df = eigenvec_df.set_index("sample_id")

    # save model_inputs to use later. This is the actual matrix used for model fitting, after all filtering steps
    model_inputs.to_pickle(os.path.join(out_dir, drug, model_prefix, "model_matrix.pkl"))

    print(f"phenotype file shape: {df_phenos.shape}")
    print(f"genotypes matrix shape: {model_inputs.shape}")
    print(f"eigenvectors shape: {eigenvec_df.shape}")

    # concatenate the eigenvectors to the matrix
    X = np.concatenate([model_inputs.values, eigenvec_df.values], axis=1)
    
else:
    print("Fitting regression without population structure correction")
    # sort by sample_id so that everything is in the same order
    model_inputs = model_inputs.sort_values("sample_id", ascending=True).reset_index(drop=True)
    df_phenos = df_phenos.sort_values("sample_id", ascending=True).reset_index(drop=True)    
    assert len(df_phenos) == len(model_inputs)

    # set index so that later only the values can be extracted and save it. This is the actual matrix used for model fitting, after all filtering steps
    model_inputs = model_inputs.set_index("sample_id")
    model_inputs.to_pickle(os.path.join(out_dir, drug, model_prefix, "model_matrix.pkl"))

    print(f"phenotype file shape: {df_phenos.shape}")
    print(f"genotypes matrix shape: {model_inputs.shape}")
    X = model_inputs.values

    
# scale inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df_phenos.phenotype.values

assert len(y) == X.shape[0]


############# STEP 5: FIT L2-PENALIZED REGRESSION #############


print(f"Performing cross-validation to get regularization parameter...")
model = LogisticRegressionCV(Cs=np.logspace(-4, 4, 9), 
                             cv=5,
                             penalty='l2', 
                             max_iter=10000, 
                             multi_class='ovr',
                             scoring='neg_log_loss'
                            )
model.fit(X, y)
print(f"Regularization parameter: {model.C_[0]}")

# save coefficients
res_df = pd.DataFrame({"variant": np.concatenate([model_inputs.columns, [f"PC{num}" for num in np.arange(num_PCs)]]), 'coef': np.squeeze(model.coef_)})
res_df.to_csv(os.path.join(out_dir, drug, model_prefix, "regression_coef.csv"), index=False)


############# STEP 6: BOOTSTRAP COEFFICIENTS #############

# use the regularization parameter determined above
print(f"Performing bootstrapping for coefficient confidence intervals...")
coefs = []
for i in range(num_bootstrap):
   
    # randomly draw sample indices
    sample_idx = np.random.choice(np.arange(0, len(y)), size=len(y), replace=True)

    # get the X and y matrices
    X_bs = X[sample_idx, :]
    y_bs = y[sample_idx]

    bs_model = LogisticRegression(C=model.C_[0], penalty='l2', max_iter=10000, multi_class='ovr')
    bs_model.fit(X_bs, y_bs)
    coefs.append(np.squeeze(bs_model.coef_))
    
    # print progress
    if i % (num_bootstrap / 10) == 0:
        print(i)

    
# save bootstrapped coefficients and principal components
coef_df = pd.DataFrame(coefs)
coef_df.columns = np.concatenate([model_inputs.columns, [f"PC{num}" for num in np.arange(num_PCs)]])
coef_df.to_csv(os.path.join(out_dir, drug, model_prefix, "coef_bootstrap.csv"), index=False)