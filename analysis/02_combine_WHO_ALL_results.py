import numpy as np
import pandas as pd
import scipy.stats as st
import glob, os, sys, yaml, subprocess, itertools, sparse, warnings
from functools import reduce
warnings.filterwarnings(action='ignore')

drug_gene_mapping = pd.read_csv("../data/drug_gene_mapping.csv")
samples_summary = pd.read_csv("../data/samples_summary.csv")

# final results from the V2 paper
who_variants = pd.read_csv("../results/WHO-catalog-V2.csv", header=[2]).query("tier==1").reset_index(drop=True)

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

# all columns to keep
cols_lst = ['Odds_Ratio', 'pval', 'BH_pval', 'neutral_pval', 'BH_neutral_pval', 'LRT_pval', 'BH_LRT_pval', 'LRT_neutral_pval', 'BH_LRT_neutral_pval', 'Present_SR', 'Present_R', 'Present_S', 'Absent_S', 'Absent_R', 'R_PPV', 'S_PPV', 'Sens', 'Spec', 'R_PPV_LB', 'R_PPV_UB', 'S_PPV_LB', 'S_PPV_UB', 'Sens_LB', 'Sens_UB', 'Spec_LB', 'Spec_UB', 'regression_confidence']

def get_results_single_pheno_group(drug, excel_dir):

    full_results_excel = pd.read_excel(f"../results/{excel_dir}/{drug}.xlsx", sheet_name=None)
    
    # don't need these columns
    del_cols = ["Phenos", "pool_type", "synonymous", "confidence_V1"]
    
    WHO_combined = []
    ALL_combined = []

    # read all the results dataframes into two lists for the WHO and ALL phenotypic groups
    for name, df in full_results_excel.items():

        df = df[df.columns[~df.columns.isin(del_cols)]]
        
        if "WHO" in name:
            WHO_combined.append(df)
        elif "ALL" in name:
            ALL_combined.append(df)
        else:
            print(name)

    # no ALL phenotypes for Pretomanid
    if drug != 'Pretomanid':
        ALL_combined = pd.concat(ALL_combined)
        
    WHO_combined = pd.concat(WHO_combined)

    return WHO_combined, ALL_combined




def clean_WHO_results_write_to_csv(drug, in_folder, out_folder, tiers_lst=[1]):
    '''
    For Pretomanid only: this function organizes the results columns and renames them with the WHO prefix. There is no merging of results from different models because there were only WHO phenotypes for Pretomanid.
    '''
    
    WHO_results_single_drug, ALL_results_single_drug = get_results_single_pheno_group(drug, in_folder)

    WHO_results_single_drug = pd.concat([WHO_results_single_drug[["mutation", "predicted_effect"]], 
                                         WHO_results_single_drug[cols_lst], 
                                       ], axis=1)

    # any mutations that were not in any regression model are added back in here as Uncertain with additional info in the Reason column
    missing_mut_df = who_variants.query("drug==@drug & tier in @tiers_lst & variant not in @WHO_results_single_drug.mutation.values")[['variant', 'effect']].rename(columns={'variant': 'mutation', 'effect': 'predicted_effect'})
    missing_mut_df['regression_confidence'] = 'Uncertain'
    missing_mut_df['Reason'] = 'Not Graded'

    save_df = pd.concat([WHO_results_single_drug, missing_mut_df], axis=0).rename(columns=dict(zip(cols_lst, [f"WHO_{col}" for col in cols_lst]))).sort_values("WHO_Odds_Ratio", ascending=False)

    # make this column so that it can be used just like for the other drugs, for which WHO and ALL results were combined. It's just a copy of WHO_regression_confidence because there is no ALL dataset
    save_df["FINAL CONFIDENCE GRADING"] = save_df["WHO_regression_confidence"]

    # rename columns for consistency with SOLO, then save
    save_df.rename(columns={'WHO_regression_confidence': 'Initial confidence grading WHO dataset'}).to_csv(f"../results/{out_folder}/{drug}.csv", index=False)



def combine_WHO_ALL_results_write_to_csv(drug, in_folder, out_folder, tiers_lst=[1]):
    '''
    Function to merge WHO and ALL gradings into a single grading for each drug. Not run for Pretomanid because there are only WHO phenotypes.
    '''
    
    WHO_results_single_drug, ALL_results_single_drug = get_results_single_pheno_group(drug, in_folder)

    # full list of mutations tested in both models
    all_mutations = list(set(WHO_results_single_drug.mutation).union(set(ALL_results_single_drug.mutation)))

    R_interim_lst = []
    R_lst = []
    S_interim_lst = []
    S_lst = []
    uncertain_lst = []
    discrepant_or_lst = []
    neutral_lst = []
    
    for mutation in all_mutations:

        # if a mutation was not tested in a model, assign it to Ungraded for the purposes of combining results. It will still be NaN in the other columns of the dataframe though
        if len(ALL_results_single_drug.query("mutation==@mutation")) == 0:
            ALL_conf = "Ungraded"
        else:
            ALL_conf = ALL_results_single_drug.query("mutation==@mutation")["regression_confidence"].values[0]

        if len(WHO_results_single_drug.query("mutation==@mutation")) == 0:
            WHO_conf = "Ungraded"
        else:
            WHO_conf = WHO_results_single_drug.query("mutation==@mutation")["regression_confidence"].values[0]        

        # regardless of what the WHO grading is, if ALL is uncertain, keep uncertain because ALL is a bigger, more representative dataset
        if ALL_conf == "Uncertain":
            uncertain_lst.append(mutation)

        # if the two phenotypic categories disagree in the sign of the OR (and have significant ORs), make uncertain
        # this includs both the Probable and strict categories
        if "Assoc w R" in WHO_conf and "Assoc w S" in ALL_conf:
            uncertain_lst.append(mutation)
            discrepant_or_lst.append(mutation)

        if "Assoc w R" in ALL_conf and "Assoc w S" in WHO_conf:
            uncertain_lst.append(mutation)
            discrepant_or_lst.append(mutation)
        
        # because ALL is a bigger, more representative dataset, make interim if WHO = uncertain/neutral and ALL = assoc
        # if ALL is in the Probable category, however, downgrade to uncertain
        if WHO_conf in ["Uncertain"]:
            if ALL_conf in ['Assoc w R']:
                R_interim_lst.append(mutation)
            elif ALL_conf in ['Assoc w S']:
                S_interim_lst.append(mutation)
            elif "Probable" in ALL_conf:
                uncertain_lst.append(mutation)
            elif ALL_conf == "Neutral":
                neutral_lst.append(mutation)

        if WHO_conf in ["Neutral", "Ungraded"]:
            if ALL_conf == 'Assoc w R':
                R_interim_lst.append(mutation)
            elif ALL_conf == 'Assoc w S':
                S_interim_lst.append(mutation)
            elif "Probable" in ALL_conf:
                uncertain_lst.append(mutation)
            elif ALL_conf == "Neutral":
                neutral_lst.append(mutation)

        # if ALL is Neutral and WHO has any association, assign to Uncertain
        if ALL_conf == "Neutral" and "Assoc" in WHO_conf:
            uncertain_lst.append(mutation)

        # upgrade mutations that are Probable in WHO and assoc w R/S in ALL to assoc
        if WHO_conf == "Probable Assoc w R":
            if ALL_conf == "Assoc w R":
                R_lst.append(mutation)

        if WHO_conf == "Probable Assoc w S":
            if ALL_conf == "Assoc w S":
                S_lst.append(mutation)

        # downgrade mutations that are assoc w R/S in WHO but Probable in ALL to interim
        if ALL_conf == "Probable Assoc w R":
            if WHO_conf == "Assoc w R":
                R_interim_lst.append(mutation)

        if ALL_conf == "Probable Assoc w S":
            if WHO_conf == "Assoc w S":
                S_interim_lst.append(mutation)

        if "Probable" in WHO_conf and "Probable" in ALL_conf:
            uncertain_lst.append(mutation)

    # check that the 4 up/downgrade lists are mutually exclusive (otherwise would indicate a bug)
    assert len(set(R_interim_lst).intersection(S_interim_lst)) == 0
    assert len(set(R_interim_lst).intersection(uncertain_lst)) == 0
    assert len(set(S_interim_lst).intersection(uncertain_lst)) == 0
    assert len(set(R_interim_lst).intersection(neutral_lst)) == 0
    assert len(set(S_interim_lst).intersection(neutral_lst)) == 0
    assert len(set(uncertain_lst).intersection(neutral_lst)) == 0
    assert len(set(R_lst).intersection(S_lst)) == 0
    assert len(set(R_interim_lst).intersection(R_lst)) == 0
    assert len(set(S_interim_lst).intersection(S_lst)) == 0

    WHO_final = WHO_results_single_drug
    WHO_final = pd.concat([WHO_final[["mutation", "predicted_effect"]], 
                           WHO_final[cols_lst], 
                          ], axis=1)
    WHO_final.rename(columns=dict(zip(cols_lst, [f"WHO_{col}" for col in cols_lst])), inplace=True)

    ALL_final = ALL_results_single_drug
    ALL_final = pd.concat([ALL_final[["mutation", "predicted_effect"]], 
                           ALL_final[cols_lst],
                          ], axis=1)
    ALL_final.rename(columns=dict(zip(cols_lst, [f"ALL_{col}" for col in cols_lst])), inplace=True)

    final_df = WHO_final.merge(ALL_final, on=["mutation", "predicted_effect"], how="outer").drop_duplicates().reset_index(drop=True)
    
    # start with WHO confidences first, then make up- or downgrades depending on the ALL results
    final_df["regression_confidence"] = final_df["WHO_regression_confidence"].fillna(final_df["ALL_regression_confidence"])

    # upgrades to interim
    final_df.loc[final_df["mutation"].isin(R_interim_lst), "regression_confidence"] = "Assoc w R - Interim"
    final_df.loc[final_df["mutation"].isin(R_lst), "regression_confidence"] = "Assoc w R"

    final_df.loc[final_df["mutation"].isin(S_interim_lst), "regression_confidence"] = "Assoc w S - Interim"
    final_df.loc[final_df["mutation"].isin(S_lst), "regression_confidence"] = "Assoc w S"
    
    final_df.loc[final_df["mutation"].isin(neutral_lst), "regression_confidence"] = "Neutral"

    # downgrades to uncertain and all remaining probables downgraded to uncertain
    final_df.loc[(final_df["mutation"].isin(uncertain_lst)) | (final_df["regression_confidence"].str.contains("Probable")), "regression_confidence"] = "Uncertain"

    # rename columns for consistency with SOLO
    final_df.rename(columns={'WHO_regression_confidence': 'Initial confidence grading WHO dataset',
                             'ALL_regression_confidence': 'Initial confidence grading ALL dataset',
                             'regression_confidence': 'FINAL CONFIDENCE GRADING'
                            }, inplace=True)

    final_df.loc[final_df["mutation"].isin(discrepant_or_lst), "Reason"] = "Discrepant ORs"
    
    # check that no mutations have been duplicated
    assert len(final_df.mutation.unique()) == len(final_df)

    # fix LoF naming
    final_df["mutation"] = final_df.mutation.str.replace("lof", "LoF")
    final_df["predicted_effect"] = final_df.predicted_effect.str.replace("lof", "LoF")

    # reorder columns so that the MIC columns are at the end
    final_df = final_df[np.concatenate([final_df.columns[~final_df.columns.str.contains('MIC')],  final_df.columns[final_df.columns.str.contains('MIC')]])]

    # any mutations that were not in any regression model are added back in here as Uncertain with additional info in the Reason column
    missing_mut_df = who_variants.query("drug==@drug & tier in @tiers_lst & variant not in @final_df.mutation.values")[['variant', 'effect']].rename(columns={'variant': 'mutation', 'effect': 'predicted_effect'})
    missing_mut_df['FINAL CONFIDENCE GRADING'] = 'Uncertain'
    missing_mut_df['Reason'] = 'Not Graded'
    
    pd.concat([final_df, missing_mut_df], axis=0).sort_values("WHO_Odds_Ratio", ascending=False).to_csv(f"../results/{out_folder}/{drug}.csv", index=False)



def write_results_for_all_drugs(drugs_lst, in_folder, out_folder, tiers_lst=[1]):
    '''
    Function to clean / combine all results for all drugs
    '''
    if not os.path.isdir(f"../results/{out_folder}"):
        os.mkdir(f"../results/{out_folder}")
    
    for drug in np.sort(drugs_lst):
    
        if drug == "Pretomanid":
            clean_WHO_results_write_to_csv(drug, in_folder, out_folder, tiers_lst=[1])            
        else:
            combine_WHO_ALL_results_write_to_csv(drug, in_folder, out_folder, tiers_lst=[1])
    
        print(drug)


# need to do all these steps for both the unpooled and pooled models. They will be combined in the next script
drugs_lst = list(drug_abbr_dict.keys())
write_results_for_all_drugs(drugs_lst, "BINARY", "UNPOOLED", tiers_lst=[1])
write_results_for_all_drugs(drugs_lst, "BINARY_POOL", "POOLED", tiers_lst=[1])