import numpy as np
import pandas as pd
import glob, os, yaml, sys, sparse, itertools
from Bio import Seq, SeqIO
import warnings
warnings.filterwarnings("ignore")


############# GENERATE TABLE OF NUMBERS OF PHENOTYPES AND GENOTYPES ACROSS TIERS AND DRUGS #############

# the purpose of this script is to see how much data we have and check if we're missing anything

snp_dir = "/n/data1/hms/dbmi/farhat/ye12/who/grm"
genos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/full_genotypes'
phenos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/phenotypes'

pheno_drugs = os.listdir(phenos_dir)
geno_drugs = os.listdir(genos_dir)

drugs_for_analysis = list(set(geno_drugs).intersection(set(pheno_drugs)))
print(len(drugs_for_analysis), "drugs with phenotypes and genotypes")

lineages = pd.read_pickle("data/combined_lineage_sample_IDs.pkl")

# remake minor alleles dataframe every time with updated data
print("Creating matrix of minor allele counts")

snp_files = [pd.read_csv(os.path.join(snp_dir, fName)) for fName in os.listdir(snp_dir)]
print(len(snp_files))

snp_combined = pd.concat(snp_files, axis=0)
snp_combined.columns = ["sample_id", "position", "nucleotide", "dp"]
snp_combined = snp_combined.drop_duplicates()

# drop sites that are in drug resistance loci. This is a little strict because it removes entire genes, but works fine.
drugs_loci = pd.read_csv("data/drugs_loci.csv")

# add 1 to the start because it's 0-indexed
drugs_loci["Start"] += 1
assert sum(drugs_loci["End"] <= drugs_loci["Start"]) == 0

# get all positions in resistance loci
remove_pos = [list(range(int(row["Start"]), int(row["End"])+1)) for _, row in drugs_loci.iterrows()]
remove_pos = list(itertools.chain.from_iterable(remove_pos))

num_pos = len(snp_combined.position.unique())
snp_combined = snp_combined.query("position not in @remove_pos")
print(f"Dropped {num_pos-len(snp_combined.position.unique())} positions in resistance-determing regions")

# pivot to matrix, NaNs with the reference alleles and then get the major allele at every site.
print("Pivoting to a matrix")
matrix = snp_combined.pivot(index="sample_id", columns="position", values="nucleotide")
assert np.nan not in matrix.values
major_alleles = matrix.mode(axis=0)

# put into dataframe to compare with the SNP dataframe
major_alleles_df = pd.concat([major_alleles]*len(matrix), ignore_index=True)
major_alleles_df.index = matrix.index.values

assert matrix.shape == major_alleles_df.shape
minor_allele_counts = (matrix != major_alleles_df).astype(int)

# drop any columns that are 0 (major allele everywhere). Easiest to do this with dropna
minor_allele_counts = minor_allele_counts.replace(0, np.nan).dropna(how='all', axis=1).fillna(0).astype(int)

# to save in sparse format, need to put the column names and indices into the dataframe, everything must be numerical
print("Saving minor allele counts dataframe")
save_matrix = minor_allele_counts.copy()
save_matrix.loc[0, :] = save_matrix.columns

# sort -- the first value is 0, which is a placeholder for the sample_id
save_matrix = save_matrix.sort_values("sample_id", ascending=True)

# put the sample_ids into the main body of the matrix and convert everything to integers
save_matrix = save_matrix.reset_index().astype(int)

# check that numbers of columns and rows have each increased by 1 and save
assert sum(np.array(save_matrix.shape) - np.array(minor_allele_counts.shape) == np.ones(2)) == 2
sparse.save_npz("data/minor_allele_counts", sparse.COO(save_matrix.values))
del save_matrix
    
    
minor_allele_counts_samples = minor_allele_counts.index.values
del minor_allele_counts
    
    
def compute_number_frameshift_by_tier(drug, tiers_lst):
    
    # first get all the genotype files associated with the drug
    geno_files = []

    for subdir in os.listdir(os.path.join(genos_dir, drug)):

        # subdirectory (tiers)
        full_subdir = os.path.join(genos_dir, drug, subdir)

        # the last character is the tier number
        if full_subdir[-1] in tiers_lst:
            for fName in os.listdir(full_subdir):
                if "run" in fName:
                    geno_files.append(os.path.join(full_subdir, fName))
                    
    dfs_lst = [pd.read_csv(fName) for fName in geno_files]
        
    if len(dfs_lst) > 0:
        df_model = pd.concat(dfs_lst)

        # get all frameshift mutations, only those for which variant_allele_frequency != 0
        frameshift = df_model.loc[(df_model["predicted_effect"] == "frameshift") & 
                                  (~pd.isnull(df_model["variant_allele_frequency"])) &
                                  (df_model["variant_allele_frequency"] > 0)
                                 ]

        # (sample, gene) pairs with multiple frameshift mutations
        lof_multi_fs = pd.DataFrame(frameshift.groupby(["sample_id", "resolved_symbol"])["predicted_effect"].count()).query("predicted_effect > 1").reset_index()
        return len(lof_multi_fs)
    else:
        return np.nan
    
    
    
summary_df = pd.DataFrame(columns=["Drug", "Phenos", "Genos", "Lineages", "SNP_Matrix", "Tier1_Multi_FS", "Tier2_Multi_FS"])
i = 0

for drug in drugs_for_analysis:
    
    # first get phenotypes
    # get all CSV files containing phenotypes
    pheno_files = os.listdir(os.path.join(phenos_dir, drug))
    
    # read them all in, concatenate, and get the number of samples
    phenos = pd.concat([pd.read_csv(os.path.join(phenos_dir, drug, fName), usecols=["sample_id"]) for fName in pheno_files if "run" in fName], axis=0)
    
    # just do tier 1, all samples in tier 1 are represented in tier 2
    geno_files = [os.path.join(genos_dir, drug, "tier=1", fName) for fName in os.listdir(os.path.join(genos_dir, drug, "tier=1")) if "run" in fName]
    genos = pd.concat([pd.read_csv(fName, usecols=["sample_id"]) for fName in geno_files], axis=0)
    
    assert len(genos.sample_id.unique()) == len(phenos.sample_id.unique())
    num_with_snps = set(minor_allele_counts_samples).intersection(genos.sample_id.unique())
    samples_with_lineages = lineages.loc[lineages["Sample ID"].isin(genos["sample_id"])]
        
    # get the number of isolates with multiple frameshift mutations in them, by tier
    num_fs_tier1 = compute_number_frameshift_by_tier(drug, ['1'])
    num_fs_tier2 = compute_number_frameshift_by_tier(drug, ['2'])
    
    # might also want to compute the number of heterozygous alleles, by tier

    summary_df.loc[i] = [drug.split("=")[1], 
                         len(phenos.sample_id.unique()), 
                         len(genos.sample_id.unique()), 
                         len(samples_with_lineages),
                         len(num_with_snps),
                         num_fs_tier1,
                         num_fs_tier2
                        ]
    i += 1
        
    print("Finished", drug.split("=")[1])
        
# already checked that the numbers of samples with genotypes and phenotypes are the same
# now check that the number of genotypes is equal to the number of lineages 

# BDQ and LZD have fewer lineages. Not all VCF files have been mapped though. Need to troubleshoot!
# assert sum(summary_df["Genos"] != summary_df["Lineages"]) == 0

summary_df.to_csv("data/samples_summary.csv", index=False)