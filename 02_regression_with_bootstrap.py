import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
import warnings
warnings.filterwarnings("ignore")
import tracemalloc
analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'


############# STEP 0: READ IN PARAMETERS FILE AND GET DIRECTORIES #############


# starting the memory monitoring
tracemalloc.start()

_, config_file, drug, _ = sys.argv

kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
binary = kwargs["binary"]

if binary:
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
    out_dir = os.path.join(analysis_dir, drug, f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}", model_prefix)
# no phenotype categories for the MIC model
else:
    out_dir = os.path.join(analysis_dir, drug, "MIC", f"tiers={'+'.join(tiers_lst)}", model_prefix)

scaler = StandardScaler()


############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############


if binary:
    phenos_file = os.path.join(analysis_dir, drug, "phenos_binary.csv")
else:
    phenos_file = os.path.join(analysis_dir, drug, "phenos_mic.csv")

df_phenos = pd.read_csv(phenos_file)


############# STEP 2: COMPUTE THE GENETIC RELATEDNESS MATRIX, REMOVING RESISTANCE LOCI #############


if not os.path.isfile(os.path.join(out_dir, "model_matrix.pkl")):
    model_inputs = pd.read_pickle(os.path.join(out_dir, "filt_matrix.pkl"))

    # reset index so that sample_id is now a column, which makes slicing easier
    model_inputs = model_inputs.reset_index()

    if num_PCs > 0:
        if not os.path.isfile("data/minor_allele_counts.npz"):
            raise ValueError("Minor allele counts dataframe does not exist. Please run compute_samples_summary.py")

        def read_in_matrix_compute_grm(fName, model_inputs):
            minor_allele_counts = sparse.load_npz(fName).todense()

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
            minor_allele_counts = minor_allele_counts.query("sample_id in @model_inputs.sample_id.values")
            grm = np.cov(minor_allele_counts.values)

            minor_allele_counts_samples = minor_allele_counts.index.values
            del minor_allele_counts
            return grm, minor_allele_counts_samples


        # compute GRM
        grm, minor_allele_counts_samples = read_in_matrix_compute_grm("data/minor_allele_counts.npz", model_inputs)


    ############# STEP 3: RUN PCA ON THE GRM #############


        pca = PCA(n_components=num_PCs)
        pca.fit(scaler.fit_transform(grm))

        print(f"Explained variance ratios of {num_PCs} principal components: {pca.explained_variance_ratio_}")
        eigenvec = pca.components_.T
        eigenvec_df = pd.DataFrame(eigenvec)
        eigenvec_df["sample_id"] = minor_allele_counts_samples
        del grm


    ############# STEP 4: PREPARE INPUTS TO THE MODEL #############


        # drop any samples from the phenotypes and genotypes dataframes that are not in the eigenvector dataframe (some samples may not have genotypes)
        model_inputs = model_inputs.query("sample_id in @eigenvec_df.sample_id.values").sort_values("sample_id", ascending=True).reset_index(drop=True)
        eigenvec_df = eigenvec_df.sort_values("sample_id", ascending=True).reset_index(drop=True)

        assert len(df_phenos) == len(eigenvec_df) == len(model_inputs)

        # set index for these 2 dataframes so that later only the values can be extracted
        model_inputs = model_inputs.set_index("sample_id")
        eigenvec_df = eigenvec_df.set_index("sample_id")

        # save model_inputs to use later. This is the actual matrix used for model fitting, after all filtering steps
        model_inputs.to_pickle(os.path.join(out_dir, "model_matrix.pkl"))
        eigenvec_df.to_pickle(os.path.join(out_dir, "model_eigenvecs.pkl"))

    else:
        # sort by sample_id so that everything is in the same order
        model_inputs = model_inputs.sort_values("sample_id", ascending=True).reset_index(drop=True)

        # set index so that later only the values can be extracted and save it. This is the actual matrix used for model fitting, after all filtering steps
        model_inputs = model_inputs.set_index("sample_id")
        model_inputs.to_pickle(os.path.join(out_dir, "model_matrix.pkl"))
        
    # to save space, delete this file. Don't need it anymore
    os.remove(os.path.join(out_dir, "filt_matrix.pkl"))
    
else:
    model_inputs = pd.read_pickle(os.path.join(out_dir, "model_matrix.pkl"))
    eigenvec_df = pd.read_pickle(os.path.join(out_dir, "model_eigenvecs.pkl"))
    
    model_inputs = model_inputs.query("sample_id in @eigenvec_df.index.values").sort_values("sample_id", ascending=True)
    

if num_PCs > 0:
    df_phenos = df_phenos.query("sample_id in @eigenvec_df.index.values").sort_values("sample_id", ascending=True).reset_index(drop=True)
    assert sum(model_inputs.merge(eigenvec_df, left_index=True, right_index=True).index != df_phenos["sample_id"]) == 0
    
    # concatenate the eigenvectors to the matrix and check the index ordering against the phenotypes matrix
    X = model_inputs.merge(eigenvec_df, left_index=True, right_index=True).values
else:
    df_phenos = df_phenos.query("sample_id in @model_inputs.index.values").sort_values("sample_id", ascending=True).reset_index(drop=True)
    assert sum(model_inputs.index != df_phenos["sample_id"]) == 0
    X = model_inputs.values
    
        
# scale inputs
X = scaler.fit_transform(X)

# binary vs. quant (MIC) phenotypes
if binary:
    y = df_phenos["phenotype"].values
    assert len(np.unique(y)) == 2
else:
    y = np.log(df_phenos["mic_value"].values)
    print(f"Min MIC: {np.min(df_phenos['mic_value'].values)}, Max MIC: {np.max(df_phenos['mic_value'].values)}")

if len(y) != X.shape[0]:
    raise ValueError(f"Shapes of model inputs {X.shape} and outputs {len(y)} are incompatible")
print(f"    {X.shape[0]} samples and {X.shape[1]} variables in the model")


############# STEP 5: FIT L2-PENALIZED REGRESSION #############


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
    print(f"    Regularization parameter: {model.C_[0]}")
else:
    print(f"    Regularization parameter: {model.alpha_}")

# save coefficients
res_df = pd.DataFrame({"variant": np.concatenate([model_inputs.columns, [f"PC{num}" for num in np.arange(num_PCs)]]), 'coef': np.squeeze(model.coef_)})
res_df.to_csv(os.path.join(out_dir, "regression_coef.csv"), index=False)


############# STEP 6: BOOTSTRAP COEFFICIENTS #############

# use the regularization parameter determined above
def bootstrap_coef():
    coefs = []
    for i in range(num_bootstrap):

        # randomly draw sample indices
        sample_idx = np.random.choice(np.arange(0, len(y)), size=len(y), replace=True)

        # get the X and y matrices
        X_bs = X[sample_idx, :]
        y_bs = y[sample_idx]

        if binary:
            bs_model = LogisticRegression(C=model.C_[0], penalty='l2', max_iter=10000, multi_class='ovr', class_weight='balanced')
        else:
            bs_model = Ridge(alpha=model.alpha_, max_iter=10000)
        bs_model.fit(X_bs, y_bs)
        coefs.append(np.squeeze(bs_model.coef_))

    return pd.DataFrame(coefs)


# save bootstrapped coefficients and principal components
coef_df = bootstrap_coef()
coef_df.columns = np.concatenate([model_inputs.columns, [f"PC{num}" for num in np.arange(num_PCs)]])
coef_df.to_csv(os.path.join(out_dir, "coef_bootstrap.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")