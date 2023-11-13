import numpy as np
import pandas as pd
import glob, os, yaml, sys, subprocess
import scipy.stats as st
import sklearn.metrics
from stats_utils import *



def create_master_genos_files(drug, genos_dir, analysis_dir, include_tier2=False):

    tier_paths = glob.glob(os.path.join(genos_dir, f"drug_name={drug}", "*"))

    if include_tier2:
        tiers_lst = ['1', '2']
    else:
        tiers_lst = ['1']
    
    for dir_name in tier_paths:
        
        if dir_name[-1] in tiers_lst:
            
            tier = dir_name[-1]
            num_files = len(os.listdir(dir_name))

            # concatenate all files in the directory and save to a gzipped csv file with the tier number as the suffix
            # 5th column is the neutral column, but it's all NaN, so remove to save space
            command = f"awk '(NR == 1) || (FNR > 1)' {dir_name}/* | cut --complement -d ',' -f 5 | gzip > {analysis_dir}/{drug}/genos_{tier}.csv.gz"
            subprocess.run(command, shell=True)
            
            print(f"Created {analysis_dir}/{drug}/genos_{tier}.csv.gz from {num_files} files")



def pool_mutations(df, effect_lst, pool_col):
        
    df.loc[df["predicted_effect"].isin(effect_lst), ["variant_category", "position"]] = [pool_col, np.nan]

    # sort descending to keep the largest variant_binary_status and variant_allele_frequency first. In this way, pooled mutations that are actually present are preserved
    df_pooled = df.query("variant_category == @pool_col").sort_values(by=["variant_binary_status", "variant_allele_frequency"], ascending=False, na_position="last").drop_duplicates(subset=["sample_id", "resolved_symbol"], keep="first")

    # remake the mutation column so that it's gene + inframe/LoF
    df_pooled["mutation"] = df_pooled["resolved_symbol"] + "_" + df_pooled["variant_category"]

    # combine with the unpooled variants and the other variants and return
    return pd.concat([df_pooled, df.query("variant_category != @pool_col")], axis=0)
    
    

def process_multiple_MICs_different_media(df_phenos):
    
    # general hierarchy: solid > liquid > plates
    # MABA, Frozen Broth Microdilution Plate (PMID31969421), UKMYC5, UKMYC6, and REMA are plates
    # 7H9 is a liquid media
    media_lst = ["7H10", "LJ", "7H11", "MGIT", "MODS", "BACTEC", "7H9", "Frozen Broth Microdilution Plate (PMID31969421)", "UKMYC6", "UKMYC5", 
                 "REMA", "MYCOTB", "MABA", "MABA24", "MABA48", "non-colourmetric", "M24 BMD"]

    media_hierarchy = dict(zip(media_lst, np.arange(len(media_lst))+1))
    
    # check that no media are missing from either
    if len(set(df_phenos.medium.values) - set(media_hierarchy.keys())) > 0:
        raise ValueError(f"{set(df_phenos.medium.values).symmetric_difference(set(media_hierarchy.keys()))} media are different between df_phenos and media_hierarchy")
    # add media hierarchy to dataframe, sort so that the highest (1) positions come first, then drop duplicates so that every sample has a single MIC
    else:
        df_phenos["media_hierarchy_pos"] = df_phenos["medium"].map(media_hierarchy)
        df_phenos = df_phenos.sort_values("media_hierarchy_pos", ascending=True).drop_duplicates("sample_id", keep="first").reset_index(drop=True)
        del df_phenos["media_hierarchy_pos"]
        assert len(df_phenos) == len(df_phenos["sample_id"].unique())
        
    return df_phenos



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




def normalize_MICs_return_dataframe(drug, df, cc_df):
    '''
    Normalize all MICs to the most commonly tested medium for a given drug
    '''

    most_common_medium = df.medium.value_counts(sort=True, ascending=False).index.values[0]
    old_samples_lst = set(df.sample_id)
    
    if drug in cc_df["Drug"].values:

        single_drug_CC = cc_df.query("Drug==@drug").reset_index(drop=True)
        most_common_medium_cc = single_drug_CC.query("Medium==@most_common_medium")["Value"].values[0]
        
        # if a medium can not be normalized, remove it 
        remove_media_lst = []
        
        for medium in df.medium.unique():
            if medium not in single_drug_CC["Medium"].values:
                remove_media_lst.append(medium)

        df = df.query("medium not in @remove_media_lst").reset_index(drop=True)
        
        print(f"    Dropped {len(set(old_samples_lst)-set(df.sample_id))}/{len(old_samples_lst)} isolates in {remove_media_lst} without critical concentrations")

        critical_conc_dict = dict(zip(cc_df["Medium"], cc_df["Value"]))
        df["medium_CC"] = df["medium"].map(critical_conc_dict)
        
        # no NaNs because the unnormalizable media were removed already
        assert len(df.loc[pd.isnull(df["medium_CC"])]) == 0
        
        # normalize MICs: multiply the MIC by the ratio of the critical concentration in 7h10 to the critical concentration in the experimental media
        df["norm_medium"] = most_common_medium
        df["norm_MIC"] = df["mic_value"] * most_common_medium_cc / df["medium_CC"]
    else:
        df = df.query("medium == @most_common_medium")
        print(f"    Dropped {len(set(old_samples_lst)-set(df.sample_id))}/{len(old_samples_lst)} isolates that are not in {most_common_medium} because {drug} is not in the critical concentrations dataframe")
    
    return df, most_common_medium

        

def remove_features_save_list(matrix, fName, dropNA=False):
    
    if dropNA:
        init_samples = matrix.index.values
        matrix = matrix.dropna(axis=0)
        next_samples = matrix.index.values

    # features that are the same everywhere, so drop them because there is no signal
    drop_features = matrix.loc[:, matrix.nunique() == 1].columns

    if len(drop_features) > 0:
        with open(fName, "w+") as file:
            for feature in drop_features:
                file.write(feature + "\n")
                
    if dropNA:
        print(f"    Dropped {len(set(init_samples)-set(next_samples))}/{len(init_samples)} isolates with missingness and {len(drop_features)}/{matrix.shape[1]} associated features")
    else:
        print(f"    Dropped {len(drop_features)}/{matrix.shape[1]} features with no signal")

    # return only the features with signal
    return matrix.loc[:, matrix.nunique() > 1]