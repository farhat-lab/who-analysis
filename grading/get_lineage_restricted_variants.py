import numpy as np
import pandas as pd
from Bio import SeqIO, Seq
import glob, os, yaml, subprocess, itertools, sparse, sys

import warnings
warnings.filterwarnings("ignore")



def get_single_drug_genotypes(drug):
    
    genos_files = glob.glob(os.path.join(analysis_dir, drug, "genos*.csv.gz"))

    df_genos = pd.concat([pd.read_csv(fName, compression="gzip", low_memory=False, 
                                      usecols=["sample_id", "resolved_symbol", "variant_category", "variant_binary_status"]
                                     ) for fName in genos_files]).query("variant_binary_status==1").drop_duplicates()

    df_genos["mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
    del df_genos["resolved_symbol"]
    del df_genos["variant_category"]
    del df_gnoes["variant_binary_status"]
    
    return df_genos



def get_mutations_in_single_lineage(df_genos, mutation, mutation_lineage_dict):
    
    single_mut_maf = minor_allele_counts.loc[df_genos.query("mutation==@mutation")["sample_id"], :]

    # search only Coll, 2014 lineage markers
    single_mut_maf = single_mut_maf[single_mut_maf.columns[single_mut_maf.columns.isin(coll2014["position"].values)]]

    # get columns that are the same for all rows and also all equal to 1
    all_positive_cols = single_mut_maf.columns[(single_mut_maf.apply(lambda col: len(col.unique())==1 and col.unique()[0] == 1)).values].values

    if len(all_positive_cols) > 0:
        # keep only the lowest level lineage (to more easily see how restricted the variant is)
        lineages = coll2014.query("position in @all_positive_cols").sort_values("Lineage", ascending=True)["Lineage"].values[-1]
        # print(f"All isolates with {mutation} are in lineages {'/'.join(lineages)}")
        mutation_lineage_dict[mutation] = lineages
        
    return mutation_lineage_dict
        
        
        
def get_lineage_dataframe(df_genos):
    
    mutation_lineage_dict = {}

    for i, mutation in enumerate(df_genos.mutation.unique()):
        mutation_lineage_dict = get_mutations_in_single_lineage(df_genos, mutation, mutation_lineage_dict)
        if i % 1000 == 0:
            print(i)
    
    single_lineage_mutations = pd.DataFrame(mutation_lineage_dict, index=[0]).T.reset_index()
    single_lineage_mutations.columns = ["mutation", "lineage"]

    print(f"{len(single_lineage_mutations) / len(df_genos.mutation.unique())} of mutations occur in a single lineage")
    
    return single_lineage_mutations



minor_allele_counts = pd.read_pickle("../data/SNP_dataframe.pkl")
analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'


coll2014 = pd.read_csv("../data/coll2014_SNP_scheme.tsv", sep="\t")
coll2014["#lineage"] = coll2014["#lineage"].str.replace("lineage", "")
coll2014.rename(columns={"#lineage": "Lineage"}, inplace=True)


for drug in os.listdir(analysis_dir):
    
    if not os.path.isfile(os.path.join(analysis_dir, drug, "single_lineage_mutations.csv")):
    
        genos_files = glob.glob(os.path.join(analysis_dir, drug, "genos*.csv.gz"))
        print(f"{len(genos_files)} genotypes files for {drug}")
        df_genos = pd.concat([pd.read_csv(fName, compression="gzip", low_memory=False, 
                                          usecols=["sample_id", "resolved_symbol", "variant_category", "variant_binary_status"]
                                         ) for fName in genos_files]).query("variant_binary_status==1").drop_duplicates()

        df_genos["mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
        del df_genos["resolved_symbol"]
        del df_genos["variant_category"]

        single_lineage_mutations = get_lineage_dataframe(df_genos)
        print(single_lineage_mutations.shape)
        single_lineage_mutations.to_csv(os.path.join(analysis_dir, drug, "single_lineage_mutations.csv"), index=False)

    else:
        print(f"Already finished {drug}")