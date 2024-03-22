import numpy as np
import pandas as pd
import scipy.stats as st
import glob, os, sys, yaml, subprocess, itertools, sparse, warnings
from functools import reduce
warnings.filterwarnings(action='ignore')

drug_gene_mapping = pd.read_csv("../data/drug_gene_mapping.csv")
samples_summary = pd.read_csv("../data/samples_summary.csv")

# utils files are in a separate folder
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "utils"))
from stats_utils import *
from data_utils import *

# CHANGE ANALYSIS DIR BEFORE RUNNING THE NOTEBOOK!
analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'

drug_abbr_dict = {"Delamanid": "DLM",
                  "Bedaquiline": "BDQ",
                  "Clofazimine": "CFZ",
                  "Ethionamide": "ETA",
                  "Linezolid": "LZD",
                  "Moxifloxacin": "MXF",
                  "Capreomycin": "CAP",
                  "Amikacin": "AMI",
                  "Pretomanid": "PTM",
                  "Pyrazinamide": "PZA",
                  "Kanamycin": "KAN",
                  "Levofloxacin": "LEV",
                  "Streptomycin": "STM",
                  "Ethambutol": "EMB",
                  "Isoniazid": "INH",
                  "Rifampicin": "RIF"
                 }

silent_lst = ['synonymous_variant', 'stop_retained_variant', 'initiator_codon_variant']


################################# Write Final Dataframes for the Binary Analysis to an Excel File #################################


# Each drug will have a separate Excel file. Each file will have one sheet per model

def get_single_model_results(drug, tiers_lst, folder, model_prefix):
    '''
    Combines the statistical results from the permutation test and the LRT
    '''
    
    ################## 1. READ IN RIDGE REGRESSION RESULTS ##################
    model_permute = pd.read_csv(os.path.join(analysis_dir, drug, folder, model_prefix, "model_analysis.csv")).query("~mutation.str.contains('PC')")
    
    ################## 2. READ IN LRT RESULTS ##################
    LRTresults = pd.read_csv(os.path.join(analysis_dir, drug, folder, model_prefix, "LRT_results.csv"))
    LRTresults["mutation"] = LRTresults.mutation.str.replace("lof", "LoF")

    # because the p-values are NaN for the FULL model row, they will be removed, so then the dataframes can be merged using inner
    LRTresults = add_pval_corrections(LRTresults.iloc[1:, ])

    # check that all mutations are represented in both the LRT results and regression model results
    assert len(set(model_permute["mutation"].values).symmetric_difference(LRTresults["mutation"].values)) == 0
    
    # combine results into a single dataframe for easy searching. REMOVE BONFERRONI AND COEFS
    combined_results = model_permute[model_permute.columns[~model_permute.columns.str.contains("|".join(["Bonferroni", "coef"]))]].merge(LRTresults[["mutation", "LRT_pval", "BH_LRT_pval", "LRT_neutral_pval", "BH_LRT_neutral_pval"]]
                                                                                                                  , on="mutation", how="inner")

    combined_results["Tier"] = tiers_lst[-1]

    # columns to return, in the desired order
    keep_cols = ['mutation', 'Tier', 'predicted_effect', 'position', 'Odds_Ratio',
                           #'OR_LB', 'OR_UB', 
                 'pval', 'BH_pval', 'neutral_pval', 'BH_neutral_pval', 'LRT_pval', 'BH_LRT_pval', 'LRT_neutral_pval', 'BH_LRT_neutral_pval']

    keep_cols += ['Num_Isolates', "Mut_R", "Mut_S", "NoMut_S", "NoMut_R", 
                  'R_PPV', 'S_PPV', 'Sens', 'Spec', 'LR+', 'LR-',
                   'R_PPV_LB', 'R_PPV_UB', 'S_PPV_LB', 'S_PPV_UB', 'Sens_LB', 'Sens_UB', 'Spec_LB',
                   'Spec_UB', 'LR+_LB', 'LR+_UB', 'LR-_LB', 'LR-_UB'
                   ]

    return combined_results[keep_cols]



def add_significance_category(df, drug, model_path):
    '''
    Add significance category annotations. Add the ones with the fewest requirements first, and then progressively add
    '''
    
    col_name = "regression_confidence"

    if col_name in df.columns:
        del df[col_name]
        
    df = df.reset_index(drop=True)
    df[["Tier", "silent"]] = df[["Tier", "silent"]].astype(int)
    
    # lower significance threshold for tier 2 genes and silent variants
    if len(df["Tier"].unique()) == 2 or "withSyn" in model_path:
        thresh = 0.01
    else:
        thresh = 0.05

    ################################################################# relaxed thresholds for pncA #################################################################
    # If a variant is significant in permutation test and LRT, make Interim.
    df.loc[(df['mutation'].str.contains('pncA')) & (df["Odds_Ratio"] > 1) & (df["BH_pval"] <= thresh) & (df['Mut_R'] >= 2) & ((df["BH_LRT_pval"] <= thresh) | (df['R_PPV'] >= 0.5)), col_name] = "Assoc w R - Interim"
    df.loc[(df['mutation'].str.contains('pncA')) & (df["Odds_Ratio"] < 1) & (df["BH_pval"] <= thresh) & (df['Mut_S'] >= 2) & ((df["BH_LRT_pval"] <= thresh) | (df['S_PPV'] >= 0.5)), col_name] = "Assoc w S - Interim"

    # if they meet both LRT and PPV criteria, upgrade to Group 1/5
    df.loc[(df['mutation'].str.contains('pncA')) & (df[col_name]=='Assoc R - Interim') & (df["BH_LRT_pval"] <= thresh) & (df['R_PPV'] >= 0.5), col_name] = "Assoc w R"
    df.loc[(df['mutation'].str.contains('pncA')) & (df[col_name]=='Assoc S - Interim') & (df["BH_LRT_pval"] <= thresh) & (df['S_PPV'] >= 0.5), col_name] = "Assoc w S"

    ################################################################ regular thresholds for other genes ################################################################
    # If a variant is significant in permutation test and LRT, make Interim.
    df.loc[(~df['mutation'].str.contains('pncA')) & (df["Odds_Ratio"] > 1) & (df["BH_pval"] <= thresh) & (df['Num_Isolates'] >= 5) & ((df["BH_LRT_pval"] <= thresh) | (df['R_PPV_LB'] >= 0.25)), col_name] = "Assoc w R - Interim"
    df.loc[(~df['mutation'].str.contains('pncA')) & (df["Odds_Ratio"] < 1) & (df["BH_pval"] <= thresh) & (df['Num_Isolates'] >= 5) & ((df["BH_LRT_pval"] <= thresh) | (df['S_PPV_LB'] >= 0.25)) , col_name] = "Assoc w S - Interim"

    # if they meet both LRT and PPV criteria, upgrade to Group 1/5
    df.loc[(~df['mutation'].str.contains('pncA')) & (df[col_name]=="Assoc w R - Interim") & (df["BH_LRT_pval"] <= thresh) & (df['R_PPV_LB'] >= 0.25), col_name] = "Assoc w R"
    df.loc[(~df['mutation'].str.contains('pncA')) & (df[col_name]=="Assoc w S - Interim") & (df["BH_LRT_pval"] <= thresh) & (df['S_PPV_LB'] >= 0.25), col_name] = "Assoc w S"

    ################################################################ Neutral criteria ################################################################
    # neutral mutations: not significant in regression AND significant in the neutral LRT test or the permutation neutral test AND present at high enough frequency
    df.loc[(~df["predicted_effect"].isin(silent_lst)) & (df["BH_pval"] > thresh) & ((df["BH_neutral_pval"] <= thresh) | (df["BH_LRT_neutral_pval"] <= thresh)) & (df['Num_Isolates'] >= 5) , col_name] = "Neutral"
    
    # for silent mutations, use raw p-values to determine neutrality.
    df.loc[(df["predicted_effect"].isin(silent_lst)) & (df["pval"] > thresh) & ((df["neutral_pval"] <= thresh) | (df["LRT_neutral_pval"] <= thresh)) & (df['Num_Isolates'] >= 5) , col_name] = "Neutral"

    # fill in any variants without a grading with Uncertain
    df[col_name] = df[col_name].replace('nan', np.nan)
    df.loc[pd.isnull(df[col_name]), col_name] = "Uncertain"
    
    df[col_name] = df[col_name].astype(str)
    assert sum(pd.isnull(df[col_name])) == 0

    # naming standardization with SOLO results
    df["mutation"] = df.mutation.str.replace("lof", "LoF")
    df["predicted_effect"] = df.predicted_effect.str.replace("lof", "LoF")

    return df




def export_binary_analyses(drugs_lst, read_folder, write_folder, analyses_lst, pooled_model_variants=False):
    '''
    pooled_model_variants boolean indicates whether to get the statistics for the non-lof, non-inframe mutations from the unpooled models or the pooled models
    '''
    
    if not os.path.isdir(f"../results/{write_folder}"):
        os.mkdir(f"../results/{write_folder}")
    
    for drug in np.sort(drugs_lst):
        
        all_analyses = {}

        for i, model_path in enumerate(analyses_lst):
            
            # some may not be there. Usually this is Pretomanid because there are no tier 2 genes or WHO phenotypes
            if os.path.isfile(os.path.join(analysis_dir, drug, read_folder, model_path, "model_analysis.csv")):
                            
                tiers_lst = [["1", "2"] if "1+2" in model_path else ["1"]][0]
                phenos_name = ["ALL" if "phenos=ALL" in model_path else "WHO"][0]
                
                # if "dropAF_withSyn_unpooled" in model_path:
                phenos_name = ["ALL" if "ALL" in model_path else "WHO"][0]
                add_analysis = get_single_model_results(drug, tiers_lst, read_folder, model_path)
                
                add_analysis["pool_type"] = model_path.split("_")[-1]
                add_analysis["silent"] = int("withSyn" in model_path)
                
                add_analysis = add_analysis[add_analysis.columns[~add_analysis.columns.str.contains("|".join(["coef", "Bonferroni"]))]]
                add_analysis = add_significance_category(add_analysis, drug, model_path)

                # exclude mutations that are already covered in earlier models
                exclude_mutations = []
                
                # for models with synonymous mutations, keep only the data for the synonymous ones
                # the data for nonsyn mutations will come from the noSyn models
                if "withSyn" in model_path:
                    try:
                        exclude_mutations += list(pd.read_pickle(os.path.join(analysis_dir, drug, read_folder, model_path.replace("withSyn", "noSyn"), "model_matrix.pkl")).columns)
                    except:
                        pass
                        
                # select which model results to keep for mutations (non-LoF, non-inframe) tested in both the pooled and unpooled models
                # if pooled_model_variants = True, keep the stats from the pooled model. else, keep the stats from the unpooled model
                if pooled_model_variants:

                    # no pooled + synonymous models
                    if "unpooled" in model_path and "noSyn" in model_path:
                        try:
                            exclude_mutations += list(pd.read_pickle(os.path.join(analysis_dir, drug, read_folder, model_path.replace("unpooled", "poolSeparate"), "model_matrix.pkl")).columns)
                        except:
                            pass
                            
                else:
                    # exclude mutations in the pooled model so that we keep the values estimated in the unpooled model
                    if ("poolSeparate" in model_path or "poolLoF" in model_path) and "noSyn" in model_path:
                        try:
                            exclude_mutations += list(pd.read_pickle(os.path.join(analysis_dir, drug, read_folder, model_path.replace("poolSeparate", "unpooled").replace("poolLoF", "unpooled"), "model_matrix.pkl")).columns)
                        except:
                            pass
                            
                add_analysis = add_analysis.query("mutation not in @exclude_mutations")

                # the phenotype category is only relevant for the binary analysis
                if read_folder == "BINARY":
                    add_analysis["Phenos"] = ["ALL" if "phenos=ALL" in model_path else "WHO"][0]

                add_analysis.rename(columns={"Num_Isolates": "Present_SR",
                                             "Mut_R": "Present_R",
                                             "NoMut_R": "Absent_R",
                                             "Mut_S": "Present_S",
                                             "NoMut_S": "Absent_S"
                                            }, inplace=True)

                if len(add_analysis) > 0:
                    all_analyses[model_path.replace("phenos=", "").replace("/", ",").replace("tiers=", "T").replace("dropAF_", "").replace("encodeAF_", "").replace("binarizeAF", "")] = add_analysis
    
        with pd.ExcelWriter(f"../results/{write_folder}/{drug}.xlsx") as file:
            for key, val in all_analyses.items():
                val.to_excel(file, sheet_name=key, index=False)
                    
        print(f"Finished {len(all_analyses)} analyses for {drug}")


# write Excel files of full results from pooled models and unpooled models
# these are saved to separate files, then they will be merged in the next script
binary_analyses_lst = [
                        ########### Tier 1, WHO phenos ###########
                        "tiers=1/phenos=WHO/dropAF_noSyn_unpooled",
                        "tiers=1/phenos=WHO/dropAF_noSyn_poolLoF",
                        # "tiers=1/phenos=WHO/dropAF_noSyn_poolSeparate",
                        "tiers=1/phenos=WHO/dropAF_withSyn_unpooled",
                        ########### Tier 1, ALL phenos ###########
                        "tiers=1/phenos=ALL/dropAF_noSyn_unpooled",
                        "tiers=1/phenos=ALL/dropAF_noSyn_poolLoF",
                        # "tiers=1/phenos=ALL/dropAF_noSyn_poolSeparate",
                        "tiers=1/phenos=ALL/dropAF_withSyn_unpooled",
                      ]

drugs_lst = list(drug_abbr_dict.keys())
# export_binary_analyses(drugs_lst, "BINARY", "BINARY_POOL", binary_analyses_lst, pooled_model_variants=True)
export_binary_analyses(drugs_lst, "BINARY", "BINARY", binary_analyses_lst, pooled_model_variants=False)