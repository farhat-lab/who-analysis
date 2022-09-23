import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
from Bio import SeqIO, Seq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import itertools
import warnings
warnings.filterwarnings("ignore")


############# STEP 0: READ IN PARAMETERS FILE AND GET DIRECTORIES #############


_, config_file = sys.argv

kwargs = yaml.safe_load(open(config_file))

drug = kwargs["drug"]
out_dir = kwargs["out_dir"]
model_prefix = kwargs["model_prefix"].split(".")[0]
num_PCs = kwargs["num_PCs"]
num_bootstrap = kwargs["num_bootstrap"]
impute = kwargs["impute"]

# list of directories
matrix_dir = "/n/data1/hms/dbmi/farhat/ye12/who/matrix"
phenos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/phenotypes'
phenos_dir = os.path.join(phenos_dir, f"drug_name={drug}")


############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############


model_inputs = pd.read_pickle(os.path.join(out_dir, drug, model_prefix, "filt_matrix.pkl"))

# reset index so that sample_id is now a column, which makes slicing easier
model_inputs = model_inputs.reset_index()

df_phenos = pd.read_csv(os.path.join(out_dir, drug, model_prefix, "phenos.csv"))


############# STEP 2: COMPUTE THE GENETIC RELATEDNESS MATRIX, REMOVING RESISTANCE LOCI #############


if num_PCs > 0:
    
    print(f"Fitting regression with population structure correction with {num_PCs} principal components")

    ############# GENERATE THE SNP MATRIX IF IT DOESN'T EXIST #############    
    if not os.path.isfile("minor_allele_counts.npz"):
        print("Creating matrix of minor allele counts")
        # read in dataframe of loci associated with drug resistance
        drugs_loci = pd.read_csv("drugs_loci.csv")

        # add 1 to the start because it's 0-indexed
        drugs_loci["Start"] += 1
        assert sum(drugs_loci["End"] <= drugs_loci["Start"]) == 0

        # get all positions in resistance loci
        remove_pos = [list(range(int(row["Start"]), int(row["End"])+1)) for _, row in drugs_loci.iterrows()]
        remove_pos = list(itertools.chain.from_iterable(remove_pos))
        print(f"{len(remove_pos)} positions in resistance-determining regions will be removed")

        matrices = [pd.read_csv(os.path.join(matrix_dir, fName)) for fName in os.listdir(matrix_dir)]
        matrices_combined = pd.concat(matrices, axis=0).set_index("sample_id")

        # convert column names to integers because remove_pos are integers
        matrices_combined.columns = matrices_combined.columns.astype(int)

        # remove positions in resistance-determining genes
        matrices_combined = matrices_combined[matrices_combined.columns[~matrices_combined.columns.isin(remove_pos)]]

        assert np.nan not in matrices_combined.values

        # get the major alleles. Then compare --> set 1 for minor alleles, 0 for major
        major_alleles = matrices_combined.mode(axis=0)

        # put into dataframe to compare with the SNP dataframe
        major_alleles_df = pd.concat([major_alleles]*len(matrices_combined), ignore_index=True)
        major_alleles_df.index = matrices_combined.index.values

        assert matrices_combined.shape == major_alleles_df.shape
        minor_allele_counts = (matrices_combined != major_alleles_df).astype(int)

        # to save in sparse format, need to put the column names and indices into the dataframe, everything must be numerical
        save_matrix = minor_allele_counts.copy()
        save_matrix.loc[0, :] = save_matrix.columns

        # sort -- the first value is 0, which is a placeholder for the sample_id
        save_matrix = save_matrix.sort_values("sample_id", ascending=True)

        # put the sample_ids into the main body of the matrix and convert everything to integers
        save_matrix = save_matrix.reset_index().astype(int)

        # check that numbers of columns and rows have each increased by 1 and save
        assert sum(np.array(save_matrix.shape) - np.array(minor_allele_counts.shape) == np.ones(2)) == 2
        sparse.save_npz("minor_allele_counts", sparse.COO(save_matrix.values))

    else:
        minor_allele_counts = sparse.load_npz("minor_allele_counts.npz").todense()
        
        # convert to dataframe
        minor_allele_counts = pd.DataFrame(minor_allele_counts)
        minor_allele_counts.columns = minor_allele_counts.iloc[0, :]
        minor_allele_counts = minor_allele_counts.iloc[1:, :]
        minor_allele_counts.rename(columns={0:"sample_id"}, inplace=True)
        minor_allele_counts["sample_id"] = minor_allele_counts["sample_id"].astype(int)

        # make sample ids the index again
        minor_allele_counts = minor_allele_counts.set_index("sample_id")
        
    mean_maf = pd.DataFrame(minor_allele_counts.mean(axis=0))
    print(f"Min MAF: {round(mean_maf[0].min(), 2)}, Max MAF: {round(mean_maf[0].max(), 2)}")

    # compute GRM using the mino allele counts of only the samples in the model
    print("Computing genetic relatedness matrix")
    minor_allele_counts = minor_allele_counts.query("sample_id in @model_inputs.sample_id.values")
    grm = np.cov(minor_allele_counts.values)
    
    
############# STEP 3: RUN PCA ON THE GRM #############


    pca = PCA(n_components=num_PCs)
    pca.fit(grm)

    print(f"Explained variance of {num_PCs} principal components: {pca.explained_variance_}")
    eigenvec = pca.components_.T
    eigenvec_df = pd.DataFrame(eigenvec)
    eigenvec_df["sample_id"] = minor_allele_counts.index.values

    
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

    # concatenate the eigenvectors to the matrix
    X = np.concatenate([model_inputs.values, eigenvec_df.values], axis=1)
    
else:
    print("    Fitting regression without population structure correction")
    # sort by sample_id so that everything is in the same order
    model_inputs = model_inputs.sort_values("sample_id", ascending=True).reset_index(drop=True)
    df_phenos = df_phenos.sort_values("sample_id", ascending=True).reset_index(drop=True)    
    assert len(df_phenos) == len(model_inputs)

    # set index so that later only the values can be extracted and save it. This is the actual matrix used for model fitting, after all filtering steps
    model_inputs = model_inputs.set_index("sample_id")
    model_inputs.to_pickle(os.path.join(out_dir, drug, model_prefix, "model_matrix.pkl"))
    X = model_inputs.values

    
# scale inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df_phenos.phenotype.values

assert len(y) == X.shape[0]
print(f"    {X.shape[0]} isolates and {X.shape[1]} features in the model")


############# STEP 5: FIT L2-PENALIZED REGRESSION #############


model = LogisticRegressionCV(Cs=np.logspace(-4, 4, 9), 
                             cv=5,
                             penalty='l2', 
                             max_iter=10000, 
                             multi_class='ovr',
                             scoring='neg_log_loss'
                            )
model.fit(X, y)
print(f"    Regularization parameter: {model.C_[0]}")

# save coefficients
res_df = pd.DataFrame({"variant": np.concatenate([model_inputs.columns, [f"PC{num}" for num in np.arange(num_PCs)]]), 'coef': np.squeeze(model.coef_)})
res_df.to_csv(os.path.join(out_dir, drug, model_prefix, "regression_coef.csv"), index=False)


############# STEP 6: BOOTSTRAP COEFFICIENTS #############

# use the regularization parameter determined above
print(f"Bootstrapping coefficient confidence intervals with {num_bootstrap} replicates")
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
    
    # # print progress
    # if i % (num_bootstrap / 10) == 0:
    #     print("   ", i)

    
# save bootstrapped coefficients and principal components
coef_df = pd.DataFrame(coefs)
coef_df.columns = np.concatenate([model_inputs.columns, [f"PC{num}" for num in np.arange(num_PCs)]])
coef_df.to_csv(os.path.join(out_dir, drug, model_prefix, "coef_bootstrap.csv"), index=False)