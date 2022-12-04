import numpy as np
import pandas as pd
import glob, os, sparse
import subprocess


import numpy as np
import pandas as pd
import glob, os, yaml, sys
import warnings
warnings.filterwarnings("ignore")
import tracemalloc
analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'
drug_gene_mapping = pd.read_csv("data/drug_gene_mapping.csv")



_, drug = sys.argv        
  
phenos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/mic'
phenos_dir = os.path.join(phenos_dir, f"drug_name={drug}")
pheno_col = "mic_value"

############# STEP 1: GET ALL AVAILABLE PHENOTYPES, PROCESS THEM, AND SAVE TO A GENERAL PHENOTYPES FILE FOR EACH MODEL TYPE #############


def get_mic_midpoints(mic_df, pheno_col):
    '''
    This function processes the MIC data from string ranges to float midpoints.  
    '''
    mic_sep = mic_df[pheno_col].str.split(",", expand=True)
    mic_sep.columns = ["MIC_lower", "MIC_upper"]

    mic_sep["Lower_bracket"] = mic_sep["MIC_lower"].str[0] #.map(bracket_mapping)
    mic_sep["Upper_bracket"] = mic_sep["MIC_upper"].str[-1] #.map(bracket_mapping)

    mic_sep["MIC_lower"] = mic_sep["MIC_lower"].str[1:]
    mic_sep["MIC_upper"] = mic_sep["MIC_upper"].str[:-1]
    mic_sep = mic_sep.replace("", np.nan)

    mic_sep[["MIC_lower", "MIC_upper"]] = mic_sep[["MIC_lower", "MIC_upper"]].astype(float)
    mic_sep = pd.concat([mic_df[["sample_id", "medium"]], mic_sep], axis=1)

    # exclude isolates with unknown lower concentrations, indicated by square bracket in the lower bound
    mic_sep = mic_sep.query("Lower_bracket != '['")
    
    # some mislabeling, where the upper bracket is a parentheses. Can't be possible because the upper bound had to have been tested
    mic_sep.loc[(mic_sep["MIC_lower"] == 0), "Upper_bracket"] = "]"
    
    # upper bracket parentheses should be [max_MIC, NaN), so check this
    assert len(mic_sep.loc[(mic_sep["Upper_bracket"] == ")") &
                           (~pd.isnull(mic_sep["MIC_upper"]))
                          ]) == 0
    
    # if the upper bound is NaN, then the MIC (midpoint) should be the lower bound, which is the maximum concentration tested
    mic_sep.loc[pd.isnull(mic_sep["MIC_upper"]), pheno_col] = mic_sep.loc[pd.isnull(mic_sep["MIC_upper"])]["MIC_lower"]
    
    # otherwise, take the average
    mic_sep.loc[~pd.isnull(mic_sep["MIC_upper"]), pheno_col] = np.mean([mic_sep.loc[~pd.isnull(mic_sep["MIC_upper"])]["MIC_lower"], mic_sep.loc[~pd.isnull(mic_sep["MIC_upper"])]["MIC_upper"]], axis=0)
    
    # check that there are no NaNs in the MIC column
    assert sum(mic_sep[pheno_col].isna()) == 0
    return mic_sep.drop_duplicates()

    
# read them all in, concatenate, and get the number of samples
df_phenos = pd.concat([pd.read_csv(os.path.join(phenos_dir, fName)) for fName in os.listdir(phenos_dir) if "run" in fName], axis=0)
    
# sometimes the data has duplicates
df_phenos = df_phenos.drop_duplicates(keep="first").reset_index(drop=True)

# Drop samples with multiple recorded phenotypes
drop_samples = df_phenos.groupby(["sample_id"]).nunique().query(f"{pheno_col} > 1").index.values

if len(drop_samples) > 0:
    print(f"    Dropping {len(drop_samples)} of {len(df_phenos['sample_id'].unique())} isolates with multiple recorded phenotypes for {drug}")
    df_phenos = df_phenos.query("sample_id not in @drop_samples")

# check that there is resistance data for all samples
assert sum(pd.isnull(df_phenos[pheno_col])) == 0
df_phenos = get_mic_midpoints(df_phenos, pheno_col)