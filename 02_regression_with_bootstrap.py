import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys, pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import warnings
warnings.filterwarnings("ignore")


############# STEP 0: READ IN PARAMETERS FILE AND GET DIRECTORIES #############


_, config_file, drug = sys.argv

kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
pheno_category_lst = kwargs["pheno_category_lst"]
model_prefix = kwargs["model_prefix"]
num_PCs = kwargs["num_PCs"]
num_bootstrap = kwargs["num_bootstrap"]

out_dir = '/n/data1/hms/dbmi/farhat/ye12/who/analysis'
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
else:
    phenos_name = "WHO"

out_dir = os.path.join(out_dir, drug, f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}", model_prefix)

if not os.path.isdir(out_dir):
    print("No model for this analysis")
    exit()

genos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/full_genotypes'
phenos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/phenotypes'
phenos_dir = os.path.join(phenos_dir, f"drug_name={drug}")


############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############


model_inputs = pd.read_pickle(os.path.join(out_dir, "filt_matrix.pkl"))

# reset index so that sample_id is now a column, which makes slicing easier
model_inputs = model_inputs.reset_index()

df_phenos = pd.read_csv(os.path.join(out_dir, "phenos.csv"))


############# STEP 2: COMPUTE THE GENETIC RELATEDNESS MATRIX, REMOVING RESISTANCE LOCI #############


if num_PCs > 0:
    
    print(f"Fitting regression with population structure correction with {num_PCs} principal components")

    if not os.path.isfile("data/minor_allele_counts.npz"):
        raise ValueError("Minor allele counts dataframe does not exist. Please run sample_numbers.py")
    else:
        minor_allele_counts = sparse.load_npz("data/minor_allele_counts.npz").todense()
        
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
    model_inputs.to_pickle(os.path.join(out_dir, "model_matrix.pkl"))
    eigenvec_df.to_pickle(os.path.join(out_dir, "model_eigenvecs.pkl"))

    # concatenate the eigenvectors to the matrix and check the index ordering against the phenotypes matrix
    X = model_inputs.merge(eigenvec_df, left_index=True, right_index=True).values
    assert sum(model_inputs.merge(eigenvec_df, left_index=True, right_index=True).index != df_phenos["sample_id"]) == 0
    
else:
    print("Fitting regression without population structure correction")
    # sort by sample_id so that everything is in the same order
    model_inputs = model_inputs.sort_values("sample_id", ascending=True).reset_index(drop=True)
    df_phenos = df_phenos.sort_values("sample_id", ascending=True).reset_index(drop=True)    
    assert len(df_phenos) == len(model_inputs)

    # set index so that later only the values can be extracted and save it. This is the actual matrix used for model fitting, after all filtering steps
    model_inputs = model_inputs.set_index("sample_id")
    model_inputs.to_pickle(os.path.join(out_dir, "model_matrix.pkl"))
    X = model_inputs.values
    assert sum(model_inputs.index != df_phenos["sample_id"]) == 0
    
    
# scale inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df_phenos.phenotype.values

assert len(y) == X.shape[0]
print(f"    {X.shape[0]} isolates and {X.shape[1]} features in the model")


############# STEP 5: FIT L2-PENALIZED REGRESSION #############


model = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
                             cv=5,
                             penalty='l2',
                             max_iter=10000, 
                             multi_class='ovr',
                             #scoring='neg_log_loss',
                             scoring='balanced_accuracy',
                             class_weight='balanced'
                            )
model.fit(X, y)
print(f"    Regularization parameter: {model.C_[0]}")
pickle.dump(model, open(os.path.join(out_dir, 'logReg_model'),'wb'))

# save coefficients
res_df = pd.DataFrame({"variant": np.concatenate([model_inputs.columns, [f"PC{num}" for num in np.arange(num_PCs)]]), 'coef': np.squeeze(model.coef_)})
res_df.to_csv(os.path.join(out_dir, "regression_coef.csv"), index=False)


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

    bs_model = LogisticRegression(C=model.C_[0], penalty='l2', max_iter=10000, multi_class='ovr', class_weight='balanced')
    bs_model.fit(X_bs, y_bs)
    coefs.append(np.squeeze(bs_model.coef_))

# save bootstrapped coefficients and principal components
coef_df = pd.DataFrame(coefs)
coef_df.columns = np.concatenate([model_inputs.columns, [f"PC{num}" for num in np.arange(num_PCs)]])
coef_df.to_csv(os.path.join(out_dir, "coef_bootstrap.csv"), index=False)