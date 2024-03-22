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
del who_variants['mutation']
who_variants.rename(columns={'drug': 'Drug', 'variant': 'mutation', 'effect': 'predicted_effect', 'INITIAL CONFIDENCE GRADING': 'SOLO INITIAL CONFIDENCE GRADING', 'FINAL CONFIDENCE GRADING': 'SOLO FINAL CONFIDENCE GRADING'}, inplace=True)

# utils files are in a separate folder
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "utils"))
from stats_utils import *
from data_utils import *

# CHANGE ANALYSIS DIR BEFORE RUNNING THE NOTEBOOK!
analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'

_, out_fName = sys.argv

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
    del_cols = ["Phenos", "pool_type", "silent"]
    
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

    # keep the first instance of each mutation because the models are ordered unpooled, poolLoF, silent/unpooled, so preferentially keep the earliest instances
    # no ALL phenotypes for Pretomanid
    if drug != 'Pretomanid':
        ALL_combined = pd.concat(ALL_combined).drop_duplicates("mutation", keep='first')
        
    WHO_combined = pd.concat(WHO_combined).drop_duplicates("mutation", keep='first')

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
    missing_mut_df = who_variants.query("Drug==@drug & tier in @tiers_lst & mutation not in @WHO_results_single_drug.mutation.values")[['mutation', 'predicted_effect']]
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

    df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv"))
    WHO_ALL_R_difference = np.abs(df_phenos.query("phenotypic_category=='WHO'").phenotype.mean() - df_phenos.phenotype.mean())
    
    WHO_results_single_drug, ALL_results_single_drug = get_results_single_pheno_group(drug, in_folder)

    # full list of mutations tested in both models
    all_mutations = list(set(WHO_results_single_drug.mutation).union(set(ALL_results_single_drug.mutation)))

    R_interim_lst = []
    R_lst = []
    S_interim_lst = []
    S_lst = []
    uncertain_lst = []
    discrepant_or_lst = []
    
    for mutation in all_mutations:

        # if a mutation was not tested in a model, don't use it for resolving gradings between the datasets, it will be handled later. Fill with empty string for now
        if len(ALL_results_single_drug.query("mutation==@mutation")) == 0:
            ALL_conf = ""
        else:
            ALL_conf = ALL_results_single_drug.query("mutation==@mutation")["regression_confidence"].values[0]

        if len(WHO_results_single_drug.query("mutation==@mutation")) == 0:
            WHO_conf = ""
        else:
            WHO_conf = WHO_results_single_drug.query("mutation==@mutation")["regression_confidence"].values[0]        

        both_gradings = [WHO_conf, ALL_conf]

        # row 1, Table 3
        # one top, the other interim --> top grading
        if "Assoc w R" in both_gradings and "Assoc w R - Interim" in both_gradings:
            R_lst.append(mutation)

        # row 2, Table 3
        if "Assoc w S" in both_gradings and "Assoc w S - Interim" in both_gradings:
            S_lst.append(mutation)

        # if both are Interim, make Uncertain
        if (WHO_conf == ALL_conf) and ((WHO_conf == "Assoc w R - Interim") | (ALL_conf == "Assoc w S - Interim")):
            uncertain_lst.append(mutation)

        # rows 3-5, Table 3
        if "Uncertain" in both_gradings or "Neutral" in both_gradings:
            # row 3
            if "Assoc w R" in both_gradings:
                R_interim_lst.append(mutation)
            # row 4
            elif "Assoc w S" in both_gradings:
                S_interim_lst.append(mutation)
            # row 5
            # if one is Interim and the other is Uncertain/Neutral, make Uncertain
            elif "Assoc w R - Interim" in both_gradings or "Assoc w S - Interim" in both_gradings:
                uncertain_lst.append(mutation)

        # row 6, Table 3
        # if the two phenotypic categories disagree in the sign of the OR (and have significant ORs), make uncertain
        # this includes both the Interim and top categories
        if "Assoc w R" in WHO_conf and "Assoc w S" in ALL_conf:
            uncertain_lst.append(mutation)
            discrepant_or_lst.append(mutation)

        if "Assoc w S" in WHO_conf and "Assoc w R" in ALL_conf:
            uncertain_lst.append(mutation)
            discrepant_or_lst.append(mutation)

    # check that the 4 up/downgrade lists are mutually exclusive (otherwise would indicate a bug)
    assert len(set(R_interim_lst).intersection(S_interim_lst)) == 0
    assert len(set(R_interim_lst).intersection(uncertain_lst)) == 0
    assert len(set(S_interim_lst).intersection(uncertain_lst)) == 0
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
    # this will also ensure that Neutral mutations are only called from the WHO dataset because there are no changes made to Neutrals after this
    final_df["regression_confidence"] = final_df["WHO_regression_confidence"].fillna(final_df["ALL_regression_confidence"])

    # upgrades using the lists above
    final_df.loc[final_df["mutation"].isin(R_interim_lst), "regression_confidence"] = "Assoc w R - Interim"
    final_df.loc[final_df["mutation"].isin(R_lst), "regression_confidence"] = "Assoc w R"

    final_df.loc[final_df["mutation"].isin(S_interim_lst), "regression_confidence"] = "Assoc w S - Interim"
    final_df.loc[final_df["mutation"].isin(S_lst), "regression_confidence"] = "Assoc w S"

    final_df.loc[final_df["mutation"].isin(uncertain_lst), "regression_confidence"] = "Uncertain"

    # take the non-Ungraded grading. The next function will add "ALL/WHO Evidence only" to the Reason column for these
    final_df.loc[(pd.isnull(final_df['WHO_Odds_Ratio'])) & (~pd.isnull(final_df['ALL_Odds_Ratio'])), 'regression_confidence'] = final_df['ALL_regression_confidence']
    final_df.loc[(pd.isnull(final_df['ALL_Odds_Ratio'])) & (~pd.isnull(final_df['WHO_Odds_Ratio'])), 'regression_confidence'] = final_df['WHO_regression_confidence']

    # rename columns for consistency with SOLO
    final_df.rename(columns={'WHO_regression_confidence': 'Initial confidence grading WHO dataset',
                             'ALL_regression_confidence': 'Initial confidence grading ALL dataset',
                             'regression_confidence': 'FINAL CONFIDENCE GRADING'
                            }, inplace=True)
    
    # check that no mutations have been duplicated
    assert final_df.mutation.nunique() == len(final_df)

    # fix LoF naming
    final_df["mutation"] = final_df.mutation.str.replace("lof", "LoF")
    final_df["predicted_effect"] = final_df.predicted_effect.str.replace("lof", "LoF")

    # reorder columns so that the MIC columns are at the end
    final_df = final_df[np.concatenate([final_df.columns[~final_df.columns.str.contains('MIC')],  final_df.columns[final_df.columns.str.contains('MIC')]])]

    # keep only variants in the WHO V2 report
    final_df = final_df.query("mutation in @who_variants.mutation.values")
    
    # any mutations that were not in any regression model are added back in here as Uncertain
    missing_mut_df = who_variants.query("Drug==@drug & tier in @tiers_lst & mutation not in @final_df.mutation.values")[['mutation', 'predicted_effect']]
    missing_mut_df['FINAL CONFIDENCE GRADING'] = 'Uncertain'
    
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
write_results_for_all_drugs(drugs_lst, "BINARY", "FINAL", tiers_lst=[1])


############################################### ADD REASON COLUMN DESCRIBING HOW THE MERGED GRADING WAS ARRIVED AT ###############################################


def complete_reason_column(df, WHO_col='Initial confidence grading WHO dataset', ALL_col='Initial confidence grading ALL dataset'):
    
    # gradings agree
    df.loc[df[WHO_col]==df[ALL_col], 'Reason'] = 'WHO ALL Same Grading'

    df.loc[(df['Reason']=='WHO ALL Same Grading') & (df[WHO_col].str.contains('Interim')), 'Reason'] = 'Both Interim'
    
    # only tested in one dataset
    df.loc[(pd.isnull(df["WHO_Odds_Ratio"])) & (~pd.isnull(df["ALL_Odds_Ratio"])), "Reason"] = "ALL Evidence Only"
    df.loc[(pd.isnull(df["ALL_Odds_Ratio"])) & (~pd.isnull(df["WHO_Odds_Ratio"])), "Reason"] = "WHO Evidence Only"

    # one grading is probable and the other is neutral --> final = Uncertain
    # and WHO = Uncertain and ALL = Probable
    # already took care of when ALL = Uncertain, final = Uncertain
    df.loc[(df[WHO_col].isin(['Neutral', 'Uncertain'])) & (df[ALL_col].str.contains('Interim')), 'Reason'] = 'Insufficient evidence'
    df.loc[(df[ALL_col].isin(['Neutral', 'Uncertain'])) & (df[WHO_col].str.contains('Interim')), 'Reason'] = 'Insufficient evidence'

    df.loc[(df[WHO_col].isin(['Neutral', 'Uncertain'])) & (df[ALL_col].isin(['Assoc w R', 'Assoc w S'])), 'Reason'] = 'Upgrade to Interim'
    df.loc[(df[ALL_col].isin(['Neutral', 'Uncertain'])) & (df[WHO_col].isin(['Assoc w R', 'Assoc w S'])), 'Reason'] = 'Downgrade to Interim'

    df.loc[(df[WHO_col].isin(['Assoc w R', 'Assoc w S'])) & (df[ALL_col].isin(['Assoc w R - Interim', 'Assoc w S - Interim'])), 'Reason'] = 'Strong evidence of assoc'
    df.loc[(df[ALL_col].isin(['Assoc w R', 'Assoc w S'])) & (df[WHO_col].isin(['Assoc w R - Interim', 'Assoc w S - Interim'])), 'Reason'] = 'Strong evidence of assoc'
    
    df.loc[(df[WHO_col]=='Neutral') & (df[ALL_col]=='Uncertain'), 'Reason'] = 'WHO Neutral'
    df.loc[(df[ALL_col]=='Neutral') & (df[WHO_col]=='Uncertain'), 'Reason'] = 'WHO Uncertain'

    df['Reason'] = df['Reason'].replace('nan', np.nan)
    assert sum(pd.isnull(df['Reason'])) == 0

    return df

# read in all single drug results, add Reason column
results_all_drugs = []

for drug in drugs_lst:
    df = pd.read_csv(f"../results/FINAL/{drug}.csv")
    df['Drug'] = drug

    # rename final grading column to initial before performing the final upgrade step below
    df.rename(columns={'FINAL CONFIDENCE GRADING': 'REGRESSION FINAL CONFIDENCE GRADING'}, inplace=True)
    results_all_drugs.append(df)

results_all_drugs = pd.concat(results_all_drugs, axis=0)
results_all_drugs = complete_reason_column(results_all_drugs)

alpha = 0.05

# upgrade variants that are significant in both permutation test and LRT in both datasets. Don't upgrade silent variants
results_all_drugs.loc[(~results_all_drugs['predicted_effect'].isin(silent_lst)) &
                      (results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING'] == 'Uncertain') &
                      (results_all_drugs['WHO_Odds_Ratio'] > 1) & (results_all_drugs['WHO_BH_pval'] <= alpha) & (results_all_drugs['WHO_BH_LRT_pval'] <= alpha) & 
                      (results_all_drugs['ALL_Odds_Ratio'] > 1) & (results_all_drugs['ALL_BH_pval'] <= alpha) & (results_all_drugs['ALL_BH_LRT_pval'] <= alpha),
                      ['REGRESSION FINAL CONFIDENCE GRADING', 'Reason']
                     ] = ['Assoc w R - Interim', 'WHO ALL significant both tests']

results_all_drugs.loc[(~results_all_drugs['predicted_effect'].isin(silent_lst)) &
                      (results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING'] == 'Uncertain') &
                      (results_all_drugs['WHO_Odds_Ratio'] < 1) & (results_all_drugs['WHO_BH_pval'] <= alpha) & (results_all_drugs['WHO_BH_LRT_pval'] <= alpha) & 
                      (results_all_drugs['ALL_Odds_Ratio'] < 1) & (results_all_drugs['ALL_BH_pval'] <= alpha) & (results_all_drugs['ALL_BH_LRT_pval'] <= alpha),
                      ['REGRESSION FINAL CONFIDENCE GRADING', 'Reason']
                     ] = ['Assoc w S - Interim', 'WHO ALL significant both tests']

results_all_drugs.loc[(results_all_drugs['Reason']=='WHO ALL significant both tests') & 
                      (results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING']=='Assoc w R - Interim') & 
                      (results_all_drugs['WHO_R_PPV_LB'] >= 0.25) &
                      (results_all_drugs['ALL_R_PPV_LB'] >= 0.25),
                      'REGRESSION FINAL CONFIDENCE GRADING'
                     ] = 'Assoc w R'

results_all_drugs.loc[(results_all_drugs['Reason']=='WHO ALL significant both tests') & 
                      (results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING']=='Assoc w S - Interim') & 
                      (results_all_drugs['WHO_S_PPV_LB'] >= 0.25) &
                      (results_all_drugs['ALL_S_PPV_LB'] >= 0.25),
                      'REGRESSION FINAL CONFIDENCE GRADING'
                     ] = 'Assoc w S'

# relaxed thresholds for pncA
results_all_drugs.loc[(results_all_drugs['Reason']=='WHO ALL significant both tests') & 
                      (results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING']=='Assoc w R - Interim') & 
                      (results_all_drugs['mutation'].str.contains('pncA')) & 
                      (results_all_drugs['WHO_R_PPV'] >= 0.5) &
                      (results_all_drugs['ALL_R_PPV'] >= 0.5),
                      'REGRESSION FINAL CONFIDENCE GRADING'
                     ] = 'Assoc w R'

results_all_drugs.loc[(results_all_drugs['Reason']=='WHO ALL significant both tests') & 
                      (results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING']=='Assoc w S - Interim') & 
                      (results_all_drugs['mutation'].str.contains('pncA')) & 
                      (results_all_drugs['WHO_S_PPV'] >= 0.5) &
                      (results_all_drugs['ALL_S_PPV'] >= 0.5),
                      'REGRESSION FINAL CONFIDENCE GRADING'
                     ] = 'Assoc w S'

# downgrade silent variants if they are not significant in both tests in both datasets to Uncertain if they are not Neutral
# exclude Uncertain here so that only new Uncertain mutations are annotated like this in the Reason column
results_all_drugs.loc[(results_all_drugs['predicted_effect'].isin(silent_lst)) &
                      (~results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING'].isin(['Neutral', 'Uncertain'])) &
                      (~((results_all_drugs['WHO_BH_pval'] <= alpha) & 
                         (results_all_drugs['WHO_BH_LRT_pval'] <= alpha) & 
                         (results_all_drugs['ALL_BH_pval'] <= alpha) & 
                         (results_all_drugs['ALL_BH_LRT_pval'] <= alpha)
                        )),
                      ['REGRESSION FINAL CONFIDENCE GRADING', 'Reason']
                     ] = ['Uncertain', 'Silent variant downgrade']

# not tested in either dataset
results_all_drugs.loc[(pd.isnull(results_all_drugs["WHO_Odds_Ratio"])) & (pd.isnull(results_all_drugs["ALL_Odds_Ratio"])), "Reason"] = "Not Graded"

# keep only variants that are in the catalog for comparison. Then add in ones that were not graded by regression and replace them with Uncertain
# replace any variant not in the dataframe with Uncertain
results_all_drugs = results_all_drugs.merge(who_variants[['Drug', 'mutation', 'SOLO INITIAL CONFIDENCE GRADING', 'SOLO FINAL CONFIDENCE GRADING']], how='right')
results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING'] = results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING'].fillna('Uncertain')

# fill in any NaNs with Uncertain. Shouldn't be any, but there are 6 in the SOLO INITIAL column
results_all_drugs[['SOLO INITIAL CONFIDENCE GRADING', 'SOLO FINAL CONFIDENCE GRADING']] = results_all_drugs[['SOLO INITIAL CONFIDENCE GRADING', 'SOLO FINAL CONFIDENCE GRADING']].fillna('3) Uncertain significance')

results_all_drugs.sort_values(['Drug', 'ALL_Odds_Ratio'], ascending=[True, False]).to_csv(out_fName, index=False)