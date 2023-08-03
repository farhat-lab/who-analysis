import numpy as np
import pandas as pd
import glob, os, yaml, sys, subprocess
import warnings
warnings.filterwarnings("ignore")
import tracemalloc
drug_gene_mapping = pd.read_csv("data/drug_gene_mapping.csv")
sys.path.append("utils")
from data_utils import *


######################### STEP 0: READ IN PARAMETERS FILE AND MAKE OUTPUT DIRECTORIES #########################


# starting the memory monitoring
tracemalloc.start()

_, config_file, drug, drug_WHO_abbr = sys.argv

kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
binary = kwargs["binary"]
atu_analysis = kwargs["atu_analysis"]
input_data_dir = kwargs["input_dir"]
analysis_dir = kwargs["output_dir"]

# double check. If running CC vs. CC-ATU analysis, they are binary phenotypes
if atu_analysis:
    binary = True
    
pheno_category_lst = kwargs["pheno_category_lst"]
# make sure that both phenotypes are included
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
    pheno_category_lst = ["ALL", "WHO"]
else:
    phenos_name = "WHO"

missing_isolate_thresh = kwargs["missing_isolate_thresh"]
missing_feature_thresh = kwargs["missing_feature_thresh"]
amb_mode = kwargs["amb_mode"]
AF_thresh = kwargs["AF_thresh"]
impute = kwargs["impute"]
synonymous = kwargs["synonymous"]
pool_type = kwargs["pool_type"]

if amb_mode == "DROP":
    model_prefix = "dropAF"
elif amb_mode == "AF":
    model_prefix = "encodeAF"
elif amb_mode == "BINARY":
    model_prefix = "binarizeAF"
else:
    raise ValueError(f"{amb_mode} is not a valid mode for handling intermediate AFs")

if synonymous:
    model_prefix += "_withSyn"
else:
    model_prefix += "_noSyn"
    
model_prefix += f"_{pool_type}"
    
# add to config file for use in the second and third scripts
kwargs["model_prefix"] = model_prefix

with open(config_file, "w") as file:
    yaml.dump(kwargs, file, default_flow_style=False, sort_keys=False)
  
if binary:
    if atu_analysis:
        out_dir = os.path.join(analysis_dir, drug, "ATU", f"tiers={'+'.join(tiers_lst)}", model_prefix)
    else:
        out_dir = os.path.join(analysis_dir, drug, "BINARY", f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}", model_prefix)
else:
    out_dir = os.path.join(analysis_dir, drug, "MIC", f"tiers={'+'.join(tiers_lst)}", model_prefix)

# create all directories down to dropped_features, which will contain text files of the features dropped during data processing
if not os.path.isdir(os.path.join(out_dir, "dropped_features")):
    os.makedirs(os.path.join(out_dir, "dropped_features"))

# # open the logging file
# log_file = open(os.path.join(out_dir, 'log.txt'), 'a')

# # Redirect print statements to the file
# sys.stdout = log_file

print(f"\nSaving model results to {out_dir}")

if binary:
    phenos_dir = os.path.join(input_data_dir, "phenotypes", f"drug_name={drug}")
    pheno_col = "phenotype"
    if atu_analysis:
        phenos_file = os.path.join(analysis_dir, drug, "phenos_atu.csv")
    else:
        phenos_file = os.path.join(analysis_dir, drug, "phenos_binary.csv")
else:
    phenos_dir = os.path.join(input_data_dir, "mic", f"drug_name={drug}")
    phenos_file = os.path.join(analysis_dir, drug, "phenos_mic.csv")
    pheno_col = "mic_value"

    
# this is mainly for the CC vs. CC-ATU analysis, which use the same genotype dataframes. Only the phenotypes are different
if os.path.isfile(os.path.join(out_dir, "model_matrix.pkl")):
    print("Model matrix already exists. Proceeding with modeling")
    exit()


######################### STEP 1: GET ALL AVAILABLE PHENOTYPES, PROCESS THEM, AND SAVE TO A GENERAL PHENOTYPES FILE FOR EACH DRUG #########################        

    
if not os.path.isfile(phenos_file):

    print(f"Creating phenotypes dataframe: {phenos_file}")
    
    # read them all in, concatenate, and get the number of samples
    df_phenos = pd.concat([pd.read_csv(os.path.join(phenos_dir, fName)) for fName in os.listdir(phenos_dir) if "run" in fName], axis=0)
    
    # sometimes the data has duplicates
    df_phenos = df_phenos.drop_duplicates(keep="first").reset_index(drop=True)
    
    if binary:
        if atu_analysis:
            df_phenos = df_phenos.loc[df_phenos["phenotypic_category"].str.contains("CC")]
        else:
            df_phenos = df_phenos.loc[~df_phenos["phenotypic_category"].str.contains("CC")]
        
        print(f"    Phenotypic categories: {df_phenos.phenotypic_category.unique()}")
        if len(df_phenos) == 0:
            print("There are no phenotypes for this analysis. Quitting this model")
            exit()

        # Drop samples with multiple recorded binary phenotypes
        drop_samples = df_phenos.groupby("sample_id").nunique().query(f"{pheno_col} > 1").reset_index()["sample_id"].values
         
        # the ATU dataframe has duplicates -- each sample has a phenotype for CC and one for CC-ATU
        if not atu_analysis:
            if len(drop_samples) > 0:
                print(f"    Dropped {len(drop_samples)}/{len(df_phenos['sample_id'].unique())} isolates with multiple recorded phenotypes")
                df_phenos = df_phenos.query("sample_id not in @drop_samples")
        else:
            # check that all samples are present twice in the ATU analysis dataframe
            assert sum(df_phenos.groupby(["sample_id"]).count()[pheno_col].unique() != np.array([2])) == 0
            
            if len(drop_samples) == 0:
                print("Phenotypes for all samples are the same for CC and CC-ATU designations. Quitting this model")
                exit()
            else:
                print(f"    {len(drop_samples)}/{len(df_phenos['sample_id'].unique())} isolates have different phenotypes using different CCs")

        assert sum(np.unique(df_phenos["phenotype"]) != np.array(['R', 'S'])) == 0
        df_phenos["phenotype"] = df_phenos["phenotype"].map({'S': 0, 'R': 1})
        
    else:
        # standardize media names so that later when prioritizing media and normalizing MICs, this is correct 
        df_phenos["medium"] = df_phenos["medium"].replace("Middlebrook7H10", "7H10")

        # separate the string value for the MIC into separate lower and upper bounds and compute the midpoint, which is stored in pheno_col ("mic_value")
        df_phenos = get_mic_midpoints(df_phenos, pheno_col)

        # Drop samples with different MICs recorded in the same media
        drop_samples = df_phenos.groupby(["sample_id", "medium"]).nunique().query(f"{pheno_col} > 1").reset_index()["sample_id"].values

        if len(drop_samples) > 0:
            print(f"    Dropped {len(drop_samples)}/{len(df_phenos['sample_id'].unique())} isolates with multiple recorded MICs in the same media")
            df_phenos = df_phenos.query("sample_id not in @drop_samples")
                    
    # column not needed, so remove to save space
    if "box" in df_phenos.columns:
        del df_phenos["box"]

    # check that there is resistance data for all samples
    assert sum(pd.isnull(df_phenos[pheno_col])) == 0
    
    # this is the phenotypes file for all models for the drug. 
    df_phenos.to_csv(phenos_file, index=False)

# phenotypes CSV files exist
else:
    print(f"{phenos_file} already exists")