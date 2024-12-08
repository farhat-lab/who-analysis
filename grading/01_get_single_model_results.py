import numpy as np
import pandas as pd
import scipy.stats as st
import glob, os, sys, yaml, subprocess, itertools, sparse, warnings, argparse
from functools import reduce
warnings.filterwarnings(action='ignore')

drug_gene_mapping = pd.read_csv("./data/drug_gene_mapping.csv")
samples_summary = pd.read_csv("./data/samples_summary.csv")

# utils files are in a separate folder
sys.path.append("utils")
from stats_utils import *
from data_utils import *

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
    LRT_results = pd.read_csv(os.path.join(analysis_dir, drug, folder, model_prefix, "LRT_results.csv"))

    # because the p-values are NaN for the FULL model row, they will be removed, so then the dataframes can be merged using inner
    # compute p-values first while PCs are still in the dataframe because they all need to be considered, then merge inner later will drop them
    LRT_results = add_pval_corrections(LRT_results.iloc[1:, ])

    # check that all mutations are represented in both the LRT results and regression model results
    if len(set(model_permute["mutation"].values).symmetric_difference(LRT_results["mutation"].values)) != 0:
        print(drug, folder, model_prefix)
        print(set(model_permute["mutation"].values) - set(LRT_results["mutation"].values))
        print(set(LRT_results["mutation"].values) - set(model_permute["mutation"].values))
        exit()
    
    # combine results into a single dataframe for easy searching. REMOVE BONFERRONI AND COEFS
    combined_results = model_permute[model_permute.columns[~model_permute.columns.str.contains("|".join(["Bonferroni", "coef"]))]].merge(LRT_results[["mutation", "LRT_pval", "BH_LRT_pval", "LRT_neutral_pval", "BH_LRT_neutral_pval"]]
                                                                                                                  , on="mutation", how="inner")

    combined_results["Tier"] = tiers_lst[-1]

    # columns to return, in the desired order
    keep_cols = ['mutation', 'Tier', 'predicted_effect', 'position', 'Odds_Ratio',
                           #'OR_LB', 'OR_UB', 
                 'pval', 'BH_pval', 'neutral_pval', 'BH_neutral_pval', 'LRT_pval', 'BH_LRT_pval']

    keep_cols += ['Num_Isolates', "Mut_R", "Mut_S", "NoMut_S", "NoMut_R", 
                  'R_PPV', 'S_PPV', 'NPV', 'Sens', 'Spec', 'LR+', 'LR-',
                   'R_PPV_LB', 'R_PPV_UB', 'S_PPV_LB', 'S_PPV_UB', 'NPV_LB', 'NPV_UB', 'Sens_LB', 'Sens_UB', 'Spec_LB',
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
    
    ################################################################ regular thresholds ################################################################
    df.loc[(df["Odds_Ratio"] > 1) & (df["BH_pval"] <= thresh) & (df['Num_Isolates'] >= 5) & (df['R_PPV_LB'] >= 0.25) & (df["BH_LRT_pval"] <= thresh), col_name] = "Assoc w R"
    df.loc[(df["Odds_Ratio"] < 1) & (df["BH_pval"] <= thresh) & (df['Num_Isolates'] >= 5) & (df['S_PPV_LB'] >= 0.25), col_name] = "Not assoc w R"

    ################################################################# relaxed thresholds for pncA #################################################################
    df.loc[(df['mutation'].str.contains('pncA')) & (df["Odds_Ratio"] > 1) & (df["BH_pval"] <= thresh) & (df['Mut_R'] >= 2) & (df['R_PPV'] >= 0.5) & (df["BH_LRT_pval"] <= thresh), col_name] = "Assoc w R"
    df.loc[(df['mutation'].str.contains('pncA')) & (df["Odds_Ratio"] < 1) & (df["BH_pval"] <= thresh) & (df['Mut_S'] >= 2) & (df['S_PPV'] >= 0.5), col_name] = "Not assoc w R"
    
    ################################################################ Neutral criteria ################################################################
    # Use threshold of 0.05 for all neutral tests for consistency with SOLO. For silent variants, the threshold of 0.01 is used in the above steps for non-Neutral
    df.loc[(~df["predicted_effect"].isin(silent_lst)) & (df["BH_pval"] > 0.05) & (df["BH_neutral_pval"] <= 0.05) & (df['Num_Isolates'] >= 5), col_name] = "Neutral"
    
    # for silent mutations, use raw p-values to determine neutrality. Use FDR-corrected p-value from permutation test though (not permutation neutral)
    df.loc[(df["predicted_effect"].isin(silent_lst)) & (df["BH_pval"] > 0.05) & (df["neutral_pval"] <= 0.05) & (df['Num_Isolates'] >= 5), col_name] = "Neutral"  
    
    # fill in any variants without a grading with Uncertain
    df[col_name] = df[col_name].replace('nan', np.nan)
    df.loc[pd.isnull(df[col_name]), col_name] = "Uncertain"
    
    df[col_name] = df[col_name].astype(str)
    assert sum(pd.isnull(df[col_name])) == 0

    return df




def export_binary_analyses(drugs_lst, read_folder, write_folder, analyses_lst):

    if not os.path.isdir(f"./results/{write_folder}"):
        os.mkdir(f"./results/{write_folder}")
    
    for drug in np.sort(drugs_lst):
        
        all_analyses = {}

        for i, model_path in enumerate(analyses_lst):
            
            # some may not be there. Usually this is Pretomanid because there are no tier 2 genes or WHO phenotypes
            if os.path.isfile(os.path.join(analysis_dir, drug, read_folder, model_path, "model_analysis.csv")):
                            
                tiers_lst = [["1", "2"] if "1+2" in model_path else ["1"]][0]
                phenos_name = ["ALL" if "phenos=ALL" in model_path else "WHO"][0]
                
                add_analysis = get_single_model_results(drug, tiers_lst, read_folder, model_path)
                
                add_analysis["pool_type"] = model_path.split("_")[-1]
                add_analysis["silent"] = int("withSyn" in model_path)
                add_analysis['Phenos'] = phenos_name
                add_analysis["Tier"] = int(tiers_lst[-1])
                
                add_analysis = add_analysis[add_analysis.columns[~add_analysis.columns.str.contains("|".join(["coef", "Bonferroni"]))]]
                add_analysis = add_significance_category(add_analysis, drug, model_path)

                # exclude mutations that are already covered in earlier models
                exclude_mutations = []
                
                # for models with synonymous mutations, keep only the results for the synonymous ones
                # the data for nonsyn mutations will come from the noSyn models
                if "withSyn" in model_path:
                    try:
                        exclude_mutations += list(pd.read_pickle(os.path.join(analysis_dir, drug, read_folder, model_path.replace("withSyn", "noSyn"), "model_matrix.pkl")).columns)
                    except:
                        pass

                # for models with LoF pooling, keep only the results for the pooled mutations
                # the data for unpooled mutations will come from the unpooled models
                if "poolLoF" in model_path:
                    try:
                        exclude_mutations += list(pd.read_pickle(os.path.join(analysis_dir, drug, read_folder, model_path.replace("poolLoF", "unpooled"), "model_matrix.pkl")).columns)
                    except:
                        pass
                            
                add_analysis = add_analysis.query("mutation not in @exclude_mutations")

                add_analysis.rename(columns={"Num_Isolates": "Present_SR",
                                             "Mut_R": "Present_R",
                                             "NoMut_R": "Absent_R",
                                             "Mut_S": "Present_S",
                                             "NoMut_S": "Absent_S"
                                            }, inplace=True)

                if len(add_analysis) > 0:
                    all_analyses[model_path.replace("phenos=", "").replace("/", ",").replace("tiers=", "T").replace("dropAF_", "").replace("encodeAF_", "").replace("binarizeAF", "")] = add_analysis

        if len(all_analyses) > 0:
            
            with pd.ExcelWriter(f"./results/{write_folder}/{drug}.xlsx") as file:
                for key, val in all_analyses.items():
                    val.to_excel(file, sheet_name=key, index=False)
                    
        print(f"Finished {len(all_analyses)} analyses for {drug}")



def export_MIC_analyses(drugs_lst, read_folder, write_folder, analyses_lst):
    '''
    This function just reads the MIC analyses and combines them into a single Excel file
    '''
    
    if not os.path.isdir(f"./results/{write_folder}"):
        os.mkdir(f"./results/{write_folder}")
    
    for drug in np.sort(drugs_lst):
        
        all_analyses = {}

        for i, model_path in enumerate(analyses_lst):
            
            # some may not be there. Usually this is Pretomanid because there are no tier 2 genes or WHO phenotypes
            if os.path.isfile(os.path.join(analysis_dir, drug, read_folder, model_path, "model_analysis.csv")):
                            
                tiers_lst = [["1", "2"] if "1+2" in model_path else ["1"]][0]

                # read in the model analysis file and remove the principal components
                add_analysis = pd.read_csv(os.path.join(analysis_dir, drug, read_folder, model_path, "model_analysis.csv")).query("~mutation.str.startswith('PC')")
                add_analysis = add_analysis[add_analysis.columns[~add_analysis.columns.str.contains("|".join(["Bonferroni"]))]]

                add_analysis["pool_type"] = model_path.split("_")[-1]
                add_analysis["silent"] = int("withSyn" in model_path)                
                add_analysis["Tier"] = int(tiers_lst[-1])
            
                # columns to return, in the desired order
                keep_cols = ['mutation', 'Tier', 'coef', 'pval', 'BH_pval', 'neutral_pval', 'BH_neutral_pval']

                add_analysis = add_analysis[keep_cols]

                # exclude mutations that are already covered in earlier models
                exclude_mutations = []
                
                # for models with synonymous mutations, keep only the results for the synonymous ones
                # the data for nonsyn mutations will come from the noSyn models
                if "withSyn" in model_path:
                    try:
                        exclude_mutations += list(pd.read_pickle(os.path.join(analysis_dir, drug, read_folder, model_path.replace("withSyn", "noSyn"), "model_matrix.pkl")).columns)
                    except:
                        pass

                # for models with LoF pooling, keep only the results for the pooled mutations
                # the data for unpooled mutations will come from the unpooled models
                if "poolLoF" in model_path:
                    try:
                        exclude_mutations += list(pd.read_pickle(os.path.join(analysis_dir, drug, read_folder, model_path.replace("poolLoF", "unpooled"), "model_matrix.pkl")).columns)
                    except:
                        pass
                            
                add_analysis = add_analysis.query("mutation not in @exclude_mutations")

                if len(add_analysis) > 0:
                    all_analyses[model_path.replace("phenos=", "").replace("/", ",").replace("tiers=", "T").replace("dropAF_", "").replace("encodeAF_", "").replace("binarizeAF", "")] = add_analysis

        if len(all_analyses) > 0:
            
            with pd.ExcelWriter(f"./results/{write_folder}/{drug}.xlsx") as file:
                for key, val in all_analyses.items():
                    val.to_excel(file, sheet_name=key, index=False)
                    
        print(f"Finished {len(all_analyses)} analyses for {drug}")
            


parser = argparse.ArgumentParser()

# need the config file just to get the kwargs, so doesn't matter which one you pass in
parser.add_argument("-c", "--config", dest='config_file', default='config.ini', type=str, required=True)

cmd_line_args = parser.parse_args()
config_file = cmd_line_args.config_file

kwargs = yaml.safe_load(open(config_file))
analysis_dir = kwargs["output_dir"]

# write Excel files of full results from pooled models and unpooled models
# these are saved to separate files, then they will be merged in the next script
binary_analyses_lst = [
                        ########### Tier 1, WHO phenos ###########
                        "tiers=1/phenos=WHO/dropAF_noSyn_unpooled",
                        "tiers=1/phenos=WHO/dropAF_noSyn_poolLoF",
                        "tiers=1/phenos=WHO/dropAF_withSyn_unpooled",
                        ########### Tier 1, ALL phenos ###########
                        "tiers=1/phenos=ALL/dropAF_noSyn_unpooled",
                        "tiers=1/phenos=ALL/dropAF_noSyn_poolLoF",
                        "tiers=1/phenos=ALL/dropAF_withSyn_unpooled",
                      ]

mic_analyses_lst = [
                    ########### Tier 1, MICs ###########
                    "tiers=1/dropAF_noSyn_unpooled",
                    "tiers=1/dropAF_noSyn_poolLoF",
                    "tiers=1/dropAF_withSyn_unpooled",
                    ]

drugs_lst = list(set(os.listdir(analysis_dir)) - set(['Pretomanid']))
print(f"Grading mutations for {len(drugs_lst)} drugs: {','.join(drugs_lst)}")
export_binary_analyses(drugs_lst, "BINARY", "BINARY", binary_analyses_lst)

# export the MIC models as well to a separate folder
export_MIC_analyses(drugs_lst, "MIC", "MIC", mic_analyses_lst)