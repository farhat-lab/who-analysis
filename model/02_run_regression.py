import numpy as np
import pandas as pd
import glob, os, yaml, sys, argparse, shutil
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
import tracemalloc, pickle
lineages_combined = pd.read_csv("lineages/combined_lineages_samples.csv", low_memory=False)

# utils files are in a separate folder
sys.path.append("utils")
from data_utils import *
from stats_utils import *

drug_abbr_dict = {"Delamanid": "DLM",
                  "Bedaquiline": "BDQ",
                  "Clofazimine": "CFZ",
                  "Ethionamide": "ETO",
                  "Linezolid": "LZD",
                  "Moxifloxacin": "MXF",
                  "Capreomycin": "CAP",
                  "Amikacin": "AMK",
                  "Pretomanid": "PMD",
                  "Pyrazinamide": "PZA",
                  "Kanamycin": "KAN",
                  "Levofloxacin": "LFX",
                  "Streptomycin": "STM",
                  "Ethambutol": "EMB",
                  "Isoniazid": "INH",
                  "Rifampicin": "RIF"
                 }


########################## STEP 0: READ IN PARAMETERS FILE AND GET DIRECTORIES ##########################

    
# starting the memory monitoring
tracemalloc.start()

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", dest='config_file', default='config.ini', type=str, required=True)
parser.add_argument('-drug', "--d", dest='drug', type=str, required=True)
parser.add_argument('--MIC-single-medium', dest='keep_single_medium', action='store_true', help='If specified, keep only the most common media for the MIC models')

cmd_line_args = parser.parse_args()
config_file = cmd_line_args.config_file
drug = cmd_line_args.drug
keep_single_medium = cmd_line_args.keep_single_medium

drug_WHO_abbr = drug_abbr_dict[drug]
kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
binary = kwargs["binary"]
atu_analysis = kwargs["atu_analysis"]
pool_type = kwargs["pool_type"]
analysis_dir = kwargs["output_dir"]
alpha = kwargs["alpha"]
num_PCs = kwargs["num_PCs"]

# read in the eigenvector dataframe and keep only the PCs for the model
eigenvec_df = pd.read_csv("PCA/eigenvec_50PC.csv", usecols=["sample_id"] + [f"PC{num+1}" for num in np.arange(num_PCs)]).set_index("sample_id")

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

    
# no model (some models don't exist: i.e. Pretomanid ALL models and any pooled models that aren't different from the corresponding unpooled models)
if not os.path.isfile(os.path.join(out_dir, f"model_matrix{model_suffix}.pkl")):
    exit()
else:
    matrix = pd.read_pickle(os.path.join(out_dir, f"model_matrix{model_suffix}.pkl"))
    
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

if os.path.isfile(os.path.join(out_dir, f"model_analysis{model_suffix}.csv")):
    print("Regression was already run for this model")
    exit()
else:
    print(f"Saving model results to {out_dir}")
    
# keep only unique MICs. Many samples have MICs tested in different media, so prioritize them according to the model hierarchy
if not binary:

    cc_df = pd.read_csv("data/drug_CC.csv")

    if keep_single_medium or drug == 'Pretomanid':

        # no normalized value for Pretomanid because there are no WHO-approved critical concentrations, so we just use the most common one
        # or only keep the most common medium
        df_phenos, most_common_medium = normalize_MICs_return_dataframe(drug, df_phenos, cc_df, keep_single_medium=keep_single_medium)

        # non-normalized column name
        mic_col = 'mic_value'

    else:
        # first apply the media hierarchy to decide which of the measured MICs to keep for each isolate (for isolates with multiple MICs measured in different media)
        df_phenos = process_multiple_MICs_different_media(df_phenos)
        
        # then, drop any media that can't be normalized and normalize to the scale of the most common medium
        df_phenos, most_common_medium = normalize_MICs_return_dataframe(drug, df_phenos, cc_df, keep_single_medium=keep_single_medium)

        # normalized column name
        mic_col = 'norm_MIC'
            
    print(f"    Min MIC: {np.min(df_phenos[mic_col].values)}, Max MIC: {np.max(df_phenos[mic_col].values)} in {most_common_medium}")
    

############# STEP 2: GET THE MATRIX ON WHICH TO FIT THE DATA +/- EIGENVECTOR COORDINATES, DEPENDING ON THE PARAM #############


print(f"{matrix.shape[0]} samples and {matrix.shape[1]} genotypic features in the model")
matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True, how="inner")
    
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
    y = np.log2(df_phenos[mic_col].values)

if len(y) != X.shape[0]:
    raise ValueError(f"Shapes of model inputs {X.shape} and outputs {len(y)} are incompatible")

print(f"{X.shape[0]} samples and {X.shape[1]} variables in the model")


######################### STEP 3: FIT L2-PENALIZED REGRESSION ##########################


if not os.path.isfile(os.path.join(out_dir, "model.sav")):
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
    pickle.dump(model, open(os.path.join(out_dir, "model.sav"), "wb"))
else:
    model = pickle.load(open(os.path.join(out_dir, "model.sav"), "rb"))

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
    
    
########################## STEP 4: PERFORM PERMUTATION TEST TO GET SIGNIFICANCE ##########################


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
    
    
final_df = get_coef_and_confidence_intervals(alpha, binary, drug_WHO_abbr, coef_df, permute_df, bootstrap_df=None)
final_df.sort_values("coef", ascending=False).to_csv(os.path.join(out_dir, f"model_analysis{model_suffix}.csv"), index=False) 

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")