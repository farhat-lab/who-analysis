import numpy as np
import pandas as pd
import glob, os, yaml, sys
import scipy.stats as st
import sklearn.metrics
import statsmodels.stats.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV, LinearRegression
import tracemalloc, pickle

from stats_utils import *
scaler = StandardScaler()



def process_multiple_MICs(df_phenos):
    
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
        df_phenos = df_phenos.sort_values("media_hierarchy_pos", ascending=True).drop_duplicates(["sample_id", "mic_value"], keep="first").reset_index(drop=True)
        del df_phenos["media_hierarchy_pos"]
        assert len(df_phenos) == len(df_phenos["sample_id"].unique())
        
    return df_phenos




def select_PCs_for_model(analysis_dir, drug, pheno_category_lst, eigenvec_df, thresh=0.01):
    '''
    This function returns an array of the principal components that are significantly associated with a given drug's phenotype, so they should be included in the L2 model.
    '''
    
    df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv")).set_index("sample_id").query("phenotypic_category in @pheno_category_lst")
    eigenvec_single_model = eigenvec_df.loc[df_phenos.index]
    
    X = scaler.fit_transform(eigenvec_single_model.values)
    y = df_phenos["phenotype"].values
    
    log_reg = LogisticRegression(class_weight='balanced', max_iter=10000, penalty=None)
    log_reg.fit(X, y)
    coef_df = pd.DataFrame({"PC": eigenvec_single_model.columns, "coef": np.squeeze(log_reg.coef_)})

    num_reps = 1000
    print(f"Performing permutation test with {num_reps} replicates")
    permute_df = perform_permutation_test(log_reg, X, y, num_reps, binary=True, penalty_type=None)
    permute_df.columns = coef_df["PC"].values

    # assess significance using the results of the permutation test
    for i, row in coef_df.iterrows():
        # p-value is the proportion of permutation coefficients that are AT LEAST AS EXTREME as the test statistic
        # ONE-SIDED because we are interested in the sign of the coefficient
        if row["coef"] > 0:
            coef_df.loc[i, "pval"] = np.mean(permute_df[row["PC"]] >= row["coef"])
        else:
            coef_df.loc[i, "pval"] = np.mean(permute_df[row["PC"]] <= row["coef"])
            
    # add FDR correction because we are testing associations with 50 PCs
    _, bh_pvals, _, _ = sm.multipletests(coef_df["pval"], method='fdr_bh', is_sorted=False, returnsorted=False)
    coef_df["BH_pval"] = bh_pvals
    
    significant = coef_df.query("BH_pval < @thresh")
    print(f"Kept {len(significant)}/{len(coef_df)} PCs for model fitting")
    return coef_df.query("BH_pval < @thresh")["PC"].values


    
    
def get_model_inputs_exclude_cooccur(variant, exclude_variants_dict, samples_highConf_tier1, df_model, df_phenos, eigenvec_df):
    '''
    Use this function to get the input matrix (dataframe) and phenotypes for a model, dropping all isolates that contain both the variant of interest 
    
    
    sample_id must be the index of BOTH df_phenos and eigenvec_df
    '''
    
    samples_with_variant = exclude_variants_dict[variant]
    samples_to_exclude = set(samples_with_variant).intersection(samples_highConf_tier1)
    print(f"{len(samples_to_exclude)} samples will be excluded")
    df_model = df_model.query("sample_id not in @samples_to_exclude")

    # drop more duplicates, but I think this might be because we have multiple data pulls at a time
    # NaN is larger than any number, so sort ascending and keep first
    
    matrix = df_model.pivot(index="sample_id", columns="mutation", values="variant_binary_status")

    # drop any isolate with missingness (because the RIF genes are well-sequenced areas), and any features that are 0 everywhere
    matrix = matrix.dropna(axis=0, how="any")
    matrix = matrix[matrix.columns[~((matrix == 0).all())]]
    
    if variant not in matrix.columns:
        print(f"{variant} was dropped from in the matrix")
        return None, None
    else:
        # combine with eigenvectors
        eigenvec_df = eigenvec_df.loc[matrix.index]
        matrix = matrix.merge(eigenvec_df, left_index=True, right_index=True)
        assert sum(matrix.index != df_phenos.index.values) == 0

        y = df_phenos.loc[matrix.index]["phenotype"].values
    
    return matrix, y





def create_lineages_file_single_model(df, lineage_col):
        
    if "Sample_ID" in df.columns:
        df = df.rename(columns={"Sample_ID": "sample_id"})
    elif "Sample ID" in df.columns:
        df = df.rename(columns={"Sample ID": "sample_id"})
        
    # first separate multiple lineages for the same sample
    add_multiLineage_df = pd.DataFrame(columns=["sample_id", lineage_col])

    for i, row in df.iterrows():

        if "," in row[lineage_col]:

            split_multi_lineages = row[lineage_col].split(",")

            for single_lineage in split_multi_lineages:
                add_multiLineage_df = pd.concat([add_multiLineage_df, pd.DataFrame({"sample_id": row["sample_id"],
                                                                                    lineage_col: single_lineage
                                                                                   }, index=[0])], axis=0)

    df = pd.concat([df.loc[~df[lineage_col].str.contains(",")], add_multiLineage_df], axis=0)[["sample_id", lineage_col]].drop_duplicates()
    
    # then add higher levels for sublineage 
    add_lineage_df = pd.DataFrame(columns=["sample_id", lineage_col])

    for i, row in df.iterrows():

        lineage = row[lineage_col]

        # means it's a sublineage
        if "." in lineage:
            
            split_lineages = lineage.split(".")

            for i, _ in enumerate(split_lineages):
                add_lineage_df = pd.concat([add_lineage_df, pd.DataFrame({"sample_id": row["sample_id"],
                                                                          lineage_col: ".".join(split_lineages[:i+1])
                                                                         }, index=[0])], axis=0)
                    
    df = pd.concat([df, add_lineage_df], axis=0)[["sample_id", lineage_col]].drop_duplicates().reset_index(drop=True)

    # just need a dummy column for values
    df["Count"] = 1
    
    return df.pivot(index="sample_id", columns=lineage_col, values="Count").fillna(0).astype(int)