import numpy as np
import pandas as pd
import scipy.stats as st
import glob, os, sys, yaml, subprocess, itertools, sparse, warnings
from functools import reduce
warnings.filterwarnings(action='ignore')

drug_gene_mapping = pd.read_csv("./data/drug_gene_mapping.csv")
samples_summary = pd.read_csv("./data/samples_summary.csv")

# final results from the V2 paper
who_variants = pd.read_csv("./results/WHO-catalog-V2-tier1.csv")
del who_variants['mutation']
who_variants.rename(columns={'drug': 'Drug', 'variant': 'mutation', 'effect': 'predicted_effect', 'INITIAL CONFIDENCE GRADING': 'SOLO INITIAL CONFIDENCE GRADING', 'FINAL CONFIDENCE GRADING': 'SOLO FINAL CONFIDENCE GRADING'}, inplace=True)

# utils files are in a separate folder
sys.path.append("utils")
from stats_utils import *
from data_utils import *

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
cols_lst = ['Odds_Ratio', 'pval', 'BH_pval', 'neutral_pval', 'BH_neutral_pval', 'LRT_pval', 'BH_LRT_pval', 'Present_SR', 'Present_R', 'Present_S', 'Absent_S', 'Absent_R', 'R_PPV', 'S_PPV', 'NPV', 'Sens', 'Spec', 'R_PPV_LB', 'R_PPV_UB', 'S_PPV_LB', 'S_PPV_UB', 'NPV_LB', 'NPV_UB', 'Sens_LB', 'Sens_UB', 'Spec_LB', 'Spec_UB', 'regression_confidence']


def get_results_single_pheno_group(drug, excel_dir):

    full_results_excel = pd.read_excel(f"./results/{excel_dir}/{drug}.xlsx", sheet_name=None)
    
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
    save_df["REGRESSION FINAL CONFIDENCE GRADING"] = save_df["WHO_regression_confidence"]

    # rename columns for consistency with SOLO, then save
    save_df.rename(columns={'WHO_regression_confidence': 'Initial confidence grading WHO dataset'}).to_csv(f"./results/{out_folder}/{drug}.csv", index=False)



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
    NotR_interim_lst = []
    NotR_lst = []
    uncertain_lst = []
    discrepant_or_lst = []
    
    for mutation in all_mutations:

        # if a mutation was not tested in a model, it is Uncertain
        if len(ALL_results_single_drug.query("mutation==@mutation")) == 0:
            ALL_conf = "Ungraded"
        else:
            ALL_conf = ALL_results_single_drug.query("mutation==@mutation")["regression_confidence"].values[0]

        if len(WHO_results_single_drug.query("mutation==@mutation")) == 0:
            WHO_conf = "Ungraded"
        else:
            WHO_conf = WHO_results_single_drug.query("mutation==@mutation")["regression_confidence"].values[0]        

        both_gradings = [WHO_conf, ALL_conf]

        # grading agreement, so upgrade to groups 1/5
        if WHO_conf == ALL_conf:
            if WHO_conf == "Assoc w R":
                R_lst.append(mutation)
            elif WHO_conf in ["Not assoc w R", "Neutral"]:
                NotR_lst.append(mutation)

        # agreement also
        if 'Not assoc w R' in both_gradings and 'Neutral' in both_gradings:
            NotR_lst.append(mutation)

        if 'Not assoc w R' in both_gradings and 'Uncertain' in both_gradings:
            NotR_interim_lst.append(mutation)

        # due to potential biases in sampling for the WHO only dataset, downgrade these disagreements to Uncertain
        if WHO_conf == 'Assoc w R':
            if ALL_conf in ['Neutral', 'Uncertain', 'Ungraded']:
                uncertain_lst.append(mutation)

        # HOWEVER, because the ALL dataset is larger and more representative, downgrade these diagreements to Interim if WHO = Uncertain. If WHO = Neutral, it's more uncertain
        if ALL_conf == 'Assoc w R':
            if WHO_conf in ['Uncertain', 'Ungraded']:
                R_interim_lst.append(mutation)
            
            elif WHO_conf == 'Neutral':
                uncertain_lst.append(mutation)

        if ALL_conf == 'Not assoc w R' and WHO_conf == 'Ungraded':
            NotR_interim_lst.append(mutation)

        # if WHO = Neutral and ALL = Uncertain, Group 5
        if WHO_conf == 'Neutral' and ALL_conf in ['Uncertain', 'Ungraded']:
            NotR_lst.append(mutation)

        # if ALL = Neutral and WHO = Uncertain (NOT UNGRADED, so it's at least present in the WHO dataset), Group 4
        if ALL_conf == 'Neutral':
            
            if WHO_conf == 'Uncertain':
                NotR_interim_lst.append(mutation)

            elif WHO_conf == 'Ungraded':
                uncertain_lst.append(mutation)
                
        # if the two phenotypic categories disagree in the sign of the OR (and have significant ORs), make uncertain
        # this includes both the Interim and top categories
        if "Assoc w R" in WHO_conf and "Not assoc w R" in ALL_conf:
            uncertain_lst.append(mutation)
            discrepant_or_lst.append(mutation)

        if "Not assoc w R" in WHO_conf and "Assoc w R" in ALL_conf:
            uncertain_lst.append(mutation)
            discrepant_or_lst.append(mutation)

    # check that the 4 up/downgrade lists are mutually exclusive (otherwise would indicate a bug
    assert len(set(R_interim_lst).intersection(NotR_interim_lst)) == 0
    assert len(set(R_interim_lst).intersection(uncertain_lst)) == 0
    assert len(set(NotR_interim_lst).intersection(uncertain_lst)) == 0
    assert len(set(R_lst).intersection(NotR_lst)) == 0
    assert len(set(R_interim_lst).intersection(R_lst)) == 0
    assert len(set(NotR_interim_lst).intersection(NotR_lst)) == 0

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

    # when the dataframes are combined, the float NaNs become string nans, so replace them
    final_df[['WHO_regression_confidence', 'ALL_regression_confidence']] = final_df[['WHO_regression_confidence', 'ALL_regression_confidence']].replace('nan', np.nan)
    
    # start with WHO confidences first, then make up- or downgrades depending on the ALL results
    # this will also ensure that Neutral mutations are only called from the WHO dataset because there are no changes made to Neutrals after this
    final_df["regression_confidence"] = final_df["WHO_regression_confidence"].fillna(final_df["ALL_regression_confidence"])
    
    # upgrades using the lists above
    final_df.loc[final_df["mutation"].isin(R_interim_lst), "regression_confidence"] = "Assoc w R - Interim"
    final_df.loc[final_df["mutation"].isin(R_lst), "regression_confidence"] = "Assoc w R"

    final_df.loc[final_df["mutation"].isin(NotR_interim_lst), "regression_confidence"] = "Not assoc w R - Interim"
    final_df.loc[(final_df["mutation"].isin(NotR_lst)) | (final_df['WHO_regression_confidence']=='Neutral'), "regression_confidence"] = "Not assoc w R"

    final_df.loc[final_df["mutation"].isin(uncertain_lst), "regression_confidence"] = "Uncertain"

    # rename columns for consistency with SOLO
    final_df.rename(columns={'WHO_regression_confidence': 'Initial confidence grading WHO dataset',
                             'ALL_regression_confidence': 'Initial confidence grading ALL dataset',
                             'regression_confidence': 'REGRESSION FINAL CONFIDENCE GRADING'
                            }, inplace=True)

    # add reason column for the discrepant OR mutations
    final_df.loc[final_df['mutation'].isin(discrepant_or_lst), 'Reason'] = 'Significant Opposite OR Signs'
    
    # check that no mutations have been duplicated
    assert final_df.mutation.nunique() == len(final_df)

    # reorder columns so that the MIC columns are at the end
    final_df = final_df[np.concatenate([final_df.columns[~final_df.columns.str.contains('MIC')],  final_df.columns[final_df.columns.str.contains('MIC')]])]

    # keep only variants in the WHO V2 report
    final_df = final_df.query("mutation in @who_variants.mutation.values")
    
    # any mutations that were not in any regression model are added back in here as Uncertain
    missing_mut_df = who_variants.query("Drug==@drug & tier in @tiers_lst & mutation not in @final_df.mutation.values")[['mutation', 'predicted_effect']]
    missing_mut_df[['Reason', 'REGRESSION FINAL CONFIDENCE GRADING']] = ['Not Graded', 'Uncertain']
    
    pd.concat([final_df, missing_mut_df], axis=0).sort_values("WHO_Odds_Ratio", ascending=False).to_csv(f"./results/{out_folder}/{drug}.csv", index=False)



def write_results_for_all_drugs(drugs_lst, in_folder, out_folder, tiers_lst=[1]):
    '''
    Function to clean / combine all results for all drugs
    '''
    if not os.path.isdir(f"./results/{out_folder}"):
        os.mkdir(f"./results/{out_folder}")
    
    for drug in np.sort(drugs_lst):
    
        if drug == "Pretomanid":
            clean_WHO_results_write_to_csv(drug, in_folder, out_folder, tiers_lst=[1])            
        else:
            combine_WHO_ALL_results_write_to_csv(drug, in_folder, out_folder, tiers_lst=[1])
    
        print(drug)


############################################### ADD REASON COLUMN DESCRIBING HOW THE MERGED GRADING WAS ARRIVED AT ###############################################


def complete_reason_column(df, WHO_col='Initial confidence grading WHO dataset', ALL_col='Initial confidence grading ALL dataset'):
    
    # gradings agree -- because the cases where the gradings are anything other than Uncertain are taken care of later, this column becomes only Uncertain
    df.loc[df[WHO_col]==df[ALL_col], 'Reason'] = 'Both Uncertain'
    
    # only tested in one dataset
    df.loc[(pd.isnull(df["WHO_Odds_Ratio"])) & (~pd.isnull(df["ALL_Odds_Ratio"])), "Reason"] = "ALL Evidence Only"
    df.loc[(pd.isnull(df["ALL_Odds_Ratio"])) & (~pd.isnull(df["WHO_Odds_Ratio"])), "Reason"] = "WHO Evidence Only"

    # one grading is Assoc w R and the other is Neutral --> final = Uncertain
    df.loc[(df[WHO_col]=='Neutral') & (df[ALL_col]=='Assoc w R'), 'Reason'] = 'Discrepant Neutral and Assoc'
    df.loc[(df[ALL_col]=='Neutral') & (df[WHO_col]=='Assoc w R'), 'Reason'] = 'Discrepant Neutral and Assoc'

    df.loc[(df[WHO_col].isin(['Uncertain'])) & (df[ALL_col].isin(['Assoc w R', 'Not assoc w R'])), 'Reason'] = 'Upgrade to Interim'

    # treat them differently because overcalling resistance is worse than overcalling non-resistance
    df.loc[(df[ALL_col].isin(['Uncertain'])) & (df[WHO_col].isin(['Not assoc w R'])), 'Reason'] = 'Downgrade to Interim'
    df.loc[(df[ALL_col].isin(['Uncertain', 'Neutral'])) & (df[WHO_col].isin(['Assoc w R'])), 'Reason'] = 'Downgrade to Uncertain'

    df.loc[(df[WHO_col] == 'Assoc w R') & (df[ALL_col] == 'Assoc w R'), 'Reason'] = 'Both Assoc w R'
    df.loc[(df[WHO_col].isin(['Neutral', 'Not assoc w R'])) & (df[ALL_col].isin(['Neutral', 'Not assoc w R'])), 'Reason'] = 'Both Not assoc w R'

    df.loc[(df[WHO_col]=='Neutral') & (df[ALL_col]=='Uncertain'), 'Reason'] = 'WHO Neutral only'
    df.loc[(df[ALL_col]=='Neutral') & (df[WHO_col]=='Uncertain'), 'Reason'] = 'ALL Neutral only'

    df['Reason'] = df['Reason'].replace('nan', np.nan)
    assert sum(pd.isnull(df['Reason'])) == 0
    return df


drugs_lst = list(drug_abbr_dict.keys())
write_results_for_all_drugs(drugs_lst, "BINARY", "FINAL", tiers_lst=[1])

# read in all single drug results, add Reason column
results_all_drugs = []

for drug in drugs_lst:
    df = pd.read_csv(f"./results/FINAL/{drug}.csv")
    df['Drug'] = drug
    results_all_drugs.append(df)

results_all_drugs = pd.concat(results_all_drugs, axis=0)
results_all_drugs = complete_reason_column(results_all_drugs)

# keep only variants that are in the catalog for comparison. Then add in ones that were not graded by regression and replace them with Uncertain
# replace any variant not in the dataframe with Uncertain
results_all_drugs = results_all_drugs.merge(who_variants[['Drug', 'mutation', 'SOLO INITIAL CONFIDENCE GRADING', 'SOLO FINAL CONFIDENCE GRADING']], how='right')

# add numbers to the groups. Easiest to do it here at the last step instead of individually in the function above
results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING'] = results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING'].map({'Assoc w R': '1) Assoc w R',
                                                                                                                         'Assoc w R - Interim': '2) Assoc w R - Interim',
                                                                                                                         'Uncertain': '3) Uncertain significance',
                                                                                                                         'Not assoc w R - Interim': '4) Not assoc w R - Interim',
                                                                                                                         'Not assoc w R': '5) Not assoc w R'
                                                                                                                        })

# upgrade LoF component variants if the pooled LoF variant for a given gene to the pooled grading
results_all_drugs['gene'] = results_all_drugs['mutation'].str.split('_').str[0]

lof_lst = ["frameshift", "start_lost", "stop_gained", "feature_ablation"]

lof_gradings = results_all_drugs.loc[(results_all_drugs["predicted_effect"] == 'LoF') & (results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING'] != '3) Uncertain significance')][['Drug', 'gene', 'mutation', 'REGRESSION FINAL CONFIDENCE GRADING']].reset_index(drop=True)

for i, row in lof_gradings.iterrows():
    drug = row['Drug']
    gene = row['gene']
    grading = row['REGRESSION FINAL CONFIDENCE GRADING']

    # if in the top groups, downgrade to interim
    if grading == '1) Assoc w R':
        grading = '2) Assoc w R - Interim'

    if grading == '5) Not assoc w R':
        grading = '4) Not assoc w R - Interim'

    results_all_drugs.loc[(results_all_drugs['Drug']==drug) & (results_all_drugs['gene']==gene) & (results_all_drugs['predicted_effect'].isin(lof_lst)) & (results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING'] == '3) Uncertain significance'), ['REGRESSION GRADING + LOF UPGRADE', 'Reason']] = [grading, 'LoF Upgrade']

# fill in any NaNs with Uncertain
results_all_drugs[['REGRESSION FINAL CONFIDENCE GRADING', 'SOLO INITIAL CONFIDENCE GRADING', 'SOLO FINAL CONFIDENCE GRADING']] = results_all_drugs[['REGRESSION FINAL CONFIDENCE GRADING', 'SOLO INITIAL CONFIDENCE GRADING', 'SOLO FINAL CONFIDENCE GRADING']].fillna('3) Uncertain significance')

results_all_drugs['REGRESSION GRADING + LOF UPGRADE'] = results_all_drugs['REGRESSION GRADING + LOF UPGRADE'].fillna(results_all_drugs['REGRESSION FINAL CONFIDENCE GRADING'])

# add grading rules results for Regression
results_all_drugs = add_grading_rules_regression(results_all_drugs)

# reorder columns so that Drug and gene come first
first_cols = ['Drug', 'gene', 'mutation']

# need to do it like this so that the existing order of columns is preserved
remaining_cols = [col for col in results_all_drugs.columns if col not in first_cols]

results_all_drugs = results_all_drugs[first_cols + remaining_cols]

results_all_drugs.sort_values(['Drug', 'ALL_Odds_Ratio'], ascending=[True, False]).to_csv(out_fName, index=False)