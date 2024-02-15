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

drugs_lst = list(drug_abbr_dict.keys())

silent_lst = ['synonymous_variant', 'stop_retained_variant', 'initiator_codon_variant']


############################ GET DISCREPANCIES FOR NON-INDEL VARIANTS BETWEEN THE UNPOOLED AND POOLED MODELS (NON-SILENT ONLY) ############################


def resolve_pooled_unpooled_model_results(drug, pooled_dir="POOLED", unpooled_dir="UNPOOLED", include_silent=False):
    '''
    This function returns a dataframe of variants with different gradings between pooled and unpooled models. These variants are NOT LoF or inframe variants, nor the component variants of pooled LoF and inframe variants.

    There are several variants that are graded Neutral in one model and Uncertain in another. These are converted to Uncertain

    In the final results, we prioritize the results from unpooled models, then update the final confidence grading for any variants with differing gradings between pooled and unpooled models. These final results are then saved to the results/FINAL folder.
    '''
    
    final_col = 'RESOLVED FINAL GRADING'

    pooled_results = pd.read_csv(f"../results/{pooled_dir}/{drug}.csv")
    unpooled_results = pd.read_csv(f"../results/{unpooled_dir}/{drug}.csv")

    # there should not be any differences in the mutations in the two sets. These lists exclude all pooled mutations and all components of pooled variants
    # i.e. there are no LoF variants and also no frameshift variants because they are components of LoF variants
    assert len(set(pooled_results.mutation).symmetric_difference(set(unpooled_results.mutation))) == 0

    if not include_silent:
        pooled_results = pooled_results.query("predicted_effect not in @silent_lst]")
        unpooled_results = unpooled_results.query("predicted_effect not in @silent_lst")

    combined_df = unpooled_results[["mutation", "FINAL CONFIDENCE GRADING"]].merge(pooled_results[["mutation", "FINAL CONFIDENCE GRADING"]], on="mutation", suffixes=["_unpooled", "_pooled"], how="outer")

    # most of the variants have the same grading in pooled and unpooled models, so take that one
    combined_df.loc[combined_df['FINAL CONFIDENCE GRADING_unpooled'] == combined_df['FINAL CONFIDENCE GRADING_pooled'], final_col] = combined_df['FINAL CONFIDENCE GRADING_unpooled']

    # add solo gradings for comparison
    combined_df = combined_df.merge(who_variants.query("drug==@drug")[["variant", "FINAL CONFIDENCE GRADING"]].rename(columns={"variant": "mutation", "FINAL CONFIDENCE GRADING": "SOLO_FINAL_GRADING"}), how="left")

    # print the drug name if there are differences for some variants that are NOT Uncertain/Neutral discrepancies
    diff_df = combined_df.loc[pd.isnull(combined_df[final_col])].reset_index(drop=True)
    del combined_df

    # resolve differences
    # can assume that there are no cases of one being assoc w R and the other being assoc w S because already assigned those to uncertain within each model above
    for i, row in diff_df.iterrows():

        # only variants with differences between pooled and unpooled models have NaNs in this column
        if pd.isnull(row[final_col]):

            if 'Uncertain' in [row['FINAL CONFIDENCE GRADING_unpooled'], row['FINAL CONFIDENCE GRADING_pooled']]: 
    
                # make Uncertain + Neutral (regardless of order) = Neutral
                if 'Neutral' in [row['FINAL CONFIDENCE GRADING_unpooled'], row['FINAL CONFIDENCE GRADING_pooled']]: 
                    diff_df.loc[i, final_col] = 'Neutral'
                    
                # if one is uncertain and the other is interim, then make uncertain
                elif 'Interim' in row['FINAL CONFIDENCE GRADING_unpooled'] or 'Interim' in row['FINAL CONFIDENCE GRADING_pooled']:
                    diff_df.loc[i, final_col] = 'Uncertain'

                # if one is uncertain and the other is probable, make interim
                elif 'Probable Assoc w R' in [row['FINAL CONFIDENCE GRADING_unpooled'], row['FINAL CONFIDENCE GRADING_pooled']]:
                    diff_df.loc[i, final_col] = 'Assoc w R - Interim'

                elif 'Probable Assoc w S' in [row['FINAL CONFIDENCE GRADING_unpooled'], row['FINAL CONFIDENCE GRADING_pooled']]:
                    diff_df.loc[i, final_col] = 'Assoc w S - Interim'

                # if the other does not contain Interim, then it must be Assoc w R/S, so make the final grading interim
                else:
                    if row['FINAL CONFIDENCE GRADING_unpooled'] == 'Uncertain':
                        diff_df.loc[i, final_col] = row['FINAL CONFIDENCE GRADING_pooled'] + ' - Interim'
                    else:
                        diff_df.loc[i, final_col] = row['FINAL CONFIDENCE GRADING_unpooled'] + ' - Interim'
        
            # if one is interim and other is Assoc w R/S, then make interim
            else:
                if 'Interim' in row['FINAL CONFIDENCE GRADING_unpooled'] and 'Interim' not in row['FINAL CONFIDENCE GRADING_pooled']:
                    diff_df.loc[i, final_col] = row['FINAL CONFIDENCE GRADING_unpooled']

                if 'Probable' in row['FINAL CONFIDENCE GRADING_unpooled'] and 'Probable' not in row['FINAL CONFIDENCE GRADING_pooled']:
                    diff_df.loc[i, final_col] = row['FINAL CONFIDENCE GRADING_unpooled']
                
                if 'Interim' in row['FINAL CONFIDENCE GRADING_pooled'] and 'Interim' not in row['FINAL CONFIDENCE GRADING_unpooled']:
                    diff_df.loc[i, final_col] = row['FINAL CONFIDENCE GRADING_pooled']

                if 'Probable' in row['FINAL CONFIDENCE GRADING_pooled'] and 'Probable' not in row['FINAL CONFIDENCE GRADING_unpooled']:
                    diff_df.loc[i, final_col] = row['FINAL CONFIDENCE GRADING_pooled']
            
    if len(diff_df) > 0:
        return diff_df
    else:
        return None


pooled_unpooled_resolved = {}

for drug in np.sort(drugs_lst):
    pooled_unpooled_resolved[drug] = resolve_pooled_unpooled_model_results(drug, pooled_dir="POOLED", unpooled_dir="UNPOOLED", include_silent=True)

df_table_S4 = []

for drug, single_drug_df in pooled_unpooled_resolved.items():

    if single_drug_df is not None:

        # keep only those where both confidence gradings are not in Neutral, Uncertain
        keep_df = single_drug_df.loc[~((single_drug_df['FINAL CONFIDENCE GRADING_unpooled'].isin(['Neutral', 'Uncertain'])) & (single_drug_df['FINAL CONFIDENCE GRADING_pooled'].isin(['Neutral', 'Uncertain'])))]

        keep_df['Drug'] = drug
        df_table_S4.append(keep_df)

df_table_S4 = pd.concat(df_table_S4)
print(f"{len(df_table_S4)} variants have significant discrepancies between the pooled and unpooled models")
df_table_S4[['Drug', 'mutation']].to_csv("../supplement/Table_S4_variants.csv", index=False)


############################ RESOLVE DISCREPANCIES FOR NON-INDEL VARIANTS BETWEEN THE UNPOOLED AND POOLED MODELS (NON-SILENT ONLY) ############################ 


def write_final_results_dataframe_single_drug(drug, pooled_unpooled_resolved, unpooled_dir, output_dir):

    unpooled_results = pd.read_csv(f"../results/{unpooled_dir}/{drug}.csv").rename(columns={'FINAL CONFIDENCE GRADING': 'UNPOOLED CONFIDENCE GRADING'})

    if pooled_unpooled_resolved[drug] is not None:
        
        update_dict = dict(zip(pooled_unpooled_resolved[drug]['mutation'], pooled_unpooled_resolved[drug]['RESOLVED FINAL GRADING']))
        pooled_result_dict = dict(zip(pooled_unpooled_resolved[drug]['mutation'], pooled_unpooled_resolved[drug]['FINAL CONFIDENCE GRADING_pooled']))
        
        # fill empty with the unpooled confidence grading because if they are not in the dictionary, then they had the same results across pooled and unpooled
        unpooled_results['POOLED CONFIDENCE GRADING'] = unpooled_results['mutation'].map(pooled_result_dict)
        unpooled_results.loc[~pd.isnull(unpooled_results['POOLED CONFIDENCE GRADING']), 'Reason'] = 'Pooled Unpooled Different'
        unpooled_results['POOLED CONFIDENCE GRADING'] = unpooled_results['POOLED CONFIDENCE GRADING'].fillna(unpooled_results['UNPOOLED CONFIDENCE GRADING'])
        unpooled_results['FINAL CONFIDENCE GRADING'] = unpooled_results['mutation'].map(update_dict).fillna(unpooled_results['UNPOOLED CONFIDENCE GRADING'])

    # no differences, so just copy the dataframe columns
    else:
        unpooled_results['POOLED CONFIDENCE GRADING'] = unpooled_results['UNPOOLED CONFIDENCE GRADING']
        unpooled_results['FINAL CONFIDENCE GRADING'] = unpooled_results['UNPOOLED CONFIDENCE GRADING']
    
    if not os.path.isdir(f"../results/{output_dir}"):
        os.mkdir(f"../results/{output_dir}")

    unpooled_results['FINAL CONFIDENCE GRADING'] = unpooled_results['FINAL CONFIDENCE GRADING'].replace("Probable Assoc w R", "Assoc w R - Interim").replace("Probable Assoc w S", "Assoc w S - Interim")

    # add published catalog results for easy comparison
    unpooled_results.merge(who_variants.query("drug==@drug")[['variant', 'INITIAL CONFIDENCE GRADING', 'FINAL CONFIDENCE GRADING']].rename(columns={'variant': 'mutation', 'INITIAL CONFIDENCE GRADING': 'SOLO INITIAL CONFIDENCE GRADING', 'FINAL CONFIDENCE GRADING': 'SOLO FINAL CONFIDENCE GRADING'}), on='mutation', how='left').to_csv(f"../results/{output_dir}/{drug}.csv", index=False)


# FINAL SINGLE-DRUG RESULTS, SAVED AS CSV FILES IN THE FINAL FOLDER
for drug in np.sort(drugs_lst):
    write_final_results_dataframe_single_drug(drug, pooled_unpooled_resolved, "UNPOOLED", "FINAL")


############################################### ADD REASON COLUMN DESCRIBING HOW THE MERGED GRADING WAS ARRIVED AT ###############################################


def complete_reason_column(df, WHO_col='Initial confidence grading WHO dataset', ALL_col='Initial confidence grading ALL dataset'):
    '''
    Already handled the following mutations:
    
        1. discrepant ORs (significant odds ratios in opposite directions between WHO and ALL)
        2. Ungraded
        3. Pooled/unpooled different

    Reason = 'Insufficient evidence' means that one grading was Neutral / Uncertain and the other was Probable Assoc, so there is not enough evidence to conclude an association or the lack of an association with resistance
    
    '''

    # gradings agree, then separate so that these don't get affected by the later steps
    df.loc[df[WHO_col]==df[ALL_col], 'Reason'] = 'WHO ALL Same Grading'
    
    # only tested in one dataset
    df.loc[(pd.isnull(df["WHO_Odds_Ratio"])) & (~pd.isnull(df["ALL_Odds_Ratio"])), "Reason"] = "ALL Evidence Only"
    df.loc[(pd.isnull(df["ALL_Odds_Ratio"])) & (~pd.isnull(df["WHO_Odds_Ratio"])), "Reason"] = "WHO Evidence Only"

    df_finished = df.loc[~pd.isnull(df["Reason"])].reset_index(drop=True)
    df = df.loc[pd.isnull(df["Reason"])].reset_index(drop=True)

    df.loc[(df[ALL_col]=='Uncertain') & (~df[WHO_col].isin(['Uncertain', 'Ungraded'])), 'Reason'] = 'ALL Uncertain'

    # both gradings probable --> final = Uncertain
    df.loc[(df[WHO_col].str.contains('Probable')) & (df[ALL_col].str.contains('Probable')), 'Reason'] = 'Both Probable'

    # upgrades using ALL evidence
    df.loc[(df[ALL_col]=='Assoc w R') & (df[WHO_col].isin(['Uncertain', 'Neutral', 'Probable Assoc w R'])), 'Reason'] = 'Upgrade using ALL Evidence'
    df.loc[(df[ALL_col]=='Assoc w S') & (df[WHO_col].isin(['Uncertain', 'Neutral', 'Probable Assoc w S'])), 'Reason'] = 'Upgrade using ALL Evidence'

    # downgrades using ALL evidence
    df.loc[(df[WHO_col]=='Assoc w R') & (df[ALL_col] == 'Probable Assoc w R'), 'Reason'] = 'Downgrade using ALL Evidence'
    df.loc[(df[WHO_col]=='Assoc w S') & (df[ALL_col] == 'Probable Assoc w S'), 'Reason'] = 'Downgrade using ALL Evidence'

    # one grading is probable and the other is neutral --> final = Uncertain
    # and WHO = Uncertain and ALL = Probable
    # already took care of when ALL = Uncertain, final = Uncertain
    df.loc[(df[WHO_col].isin(['Neutral', 'Uncertain'])) & (df[ALL_col].str.contains('Probable')), 'Reason'] = 'Insufficient evidence'
    df.loc[(df[ALL_col] == 'Neutral') & (df[WHO_col].str.contains('Probable')), 'Reason'] = 'Insufficient evidence'

    # WHO = Uncertain, ALL = Neutral, final = Neutral
    df.loc[(df[WHO_col]=='Uncertain') & (df[ALL_col]=='Neutral'), 'Reason'] = 'Neutral in ALL'

    return pd.concat([df, df_finished], axis=0).sort_values(["Drug", "ALL_Odds_Ratio"], ascending=[True, False])


# read in all single drug results, add Reason column
results_all_drugs = []

for drug in drugs_lst:
    df = pd.read_csv(f"../results/FINAL/{drug}.csv")
    df['Drug'] = drug

    # rename final grading column to initial before performing the final upgrade step below
    df.rename(columns={'FINAL CONFIDENCE GRADING': 'REGRESSION INITIAL CONFIDENCE GRADING'}, inplace=True)
    results_all_drugs.append(df)

results_all_drugs = pd.concat(results_all_drugs, axis=0)
results_all_drugs = complete_reason_column(results_all_drugs)


############################################### UPGRADE VARIANTS SIGNIFICANT IN BOTH PERMUTATION AND LRT IN BOTH WHO AND ALL ###############################################


alpha = 0.05

# don't change gradings for silent variants
# R-associated
results_all_drugs.loc[(results_all_drugs['WHO_Odds_Ratio'] > 1) & (results_all_drugs['WHO_BH_pval'] <= alpha) & (results_all_drugs['WHO_BH_LRT_pval'] <= alpha) &
                      (results_all_drugs['ALL_Odds_Ratio'] > 1) & (results_all_drugs['ALL_BH_pval'] <= alpha) & (results_all_drugs['ALL_BH_LRT_pval'] <= alpha) & 
                      (results_all_drugs['REGRESSION INITIAL CONFIDENCE GRADING'] == 'Uncertain') & 
                      (~results_all_drugs['predicted_effect'].isin(silent_lst)),
                      'REGRESSION FINAL CONFIDENCE GRADING'                   
                    ] = 'Assoc w R - Interim'

# S-associated
results_all_drugs.loc[(results_all_drugs['WHO_Odds_Ratio'] < 1) & (results_all_drugs['WHO_BH_pval'] <= alpha) & (results_all_drugs['WHO_BH_LRT_pval'] <= alpha) &
                      (results_all_drugs['ALL_Odds_Ratio'] < 1) & (results_all_drugs['ALL_BH_pval'] <= alpha) & (results_all_drugs['ALL_BH_LRT_pval'] <= alpha) & 
                      (results_all_drugs['REGRESSION INITIAL CONFIDENCE GRADING'] == 'Uncertain') & 
                      (~results_all_drugs['predicted_effect'].isin(silent_lst)),
                     'REGRESSION FINAL CONFIDENCE GRADING'                      
                     ] = 'Assoc w S - Interim'

results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING'] = results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING'].replace('nan', np.nan)

# anything that wasn't changed will have the same grading as the initial column
results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING'] = results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING'].fillna(results_all_drugs['REGRESSION INITIAL CONFIDENCE GRADING'])

# keep only variants that are in the catalog for comparison
results_all_drugs = results_all_drugs.loc[~pd.isnull(results_all_drugs['SOLO FINAL CONFIDENCE GRADING'])]

# sort by drug and ALL OR and save
results_all_drugs.sort_values(['Drug', 'ALL_Odds_Ratio'], ascending=[True, False]).to_csv("../results/Regression_Final_Feb2024_Tier1.csv", index=False)