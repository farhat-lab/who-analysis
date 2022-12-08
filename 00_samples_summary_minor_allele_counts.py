import numpy as np
import pandas as pd
import glob, os, yaml, sparse, itertools, sys, warnings
from Bio import Seq, SeqIO
warnings.filterwarnings("ignore")


############# GENERATE THE MINOR ALLELE COUNTS DATAFRAME AND THE TABLE OF NUMBERS OF PHENOTYPES AND GENOTYPES ACROSS TIERS AND DRUGS #############

# The purpose of this script is to see how much data we have and check if we're missing anything. 
# It also generates the minor allele counts dataframe from the GRM directory. 
# Principal components are computed from this dataframe for lineage structure correction.


# default for Farhat analysis: input_data_dir = "/n/data1/hms/dbmi/farhat/ye12/who"
_, input_data_dir = sys.argv

snp_dir = os.path.join(input_data_dir, "grm")
genos_dir = os.path.join(input_data_dir, "full_genotypes")
phenos_dir = os.path.join(input_data_dir, "phenotypes")
mic_dir = os.path.join(input_data_dir, "mic")

pheno_drugs = os.listdir(phenos_dir)
geno_drugs = os.listdir(genos_dir)
mic_drugs = os.listdir(mic_dir)

drugs_for_analysis = list(set(geno_drugs).intersection(set(pheno_drugs)).intersection(set(mic_drugs)))
print(len(drugs_for_analysis), "drugs with phenotypes and genotypes")

lineages = pd.read_pickle("data/combined_lineage_sample_IDs.pkl")

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
print(f"Dropped {num_pos-len(snp_combined.position.unique())} positions in resistance-determining regions")

# pivot to matrix. There should not be any NaNs because the data is complete (i.e. reference alleles are included), then get the major allele at every site.
print("Pivoting to a matrix")
matrix = snp_combined.pivot(index="sample_id", columns="position", values="nucleotide")
assert np.nan not in matrix.values
major_alleles = matrix.mode(axis=0)

# put into dataframe to compare with the SNP dataframe. Most efficient way is to make a dataframe of major alleles where every row is the same. 
major_alleles_df = pd.concat([major_alleles]*len(matrix), ignore_index=True)
major_alleles_df.index = matrix.index.values

assert matrix.shape == major_alleles_df.shape
minor_allele_counts = (matrix != major_alleles_df).astype(int)

# drop any columns that are 0 (major allele everywhere). Easiest to do this with dropna -- convert 0s to NaNs, drop, then back to 0s.
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
    
    
def compute_num_mutations(drug, tiers_lst):
    
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

        # get all mutations that are positive for LOF. Ignore ambiguous variants in this count
        lof = df_model.loc[(df_model["predicted_effect"].isin(["frameshift", "start_lost", "stop_gained", "feature_ablation"])) & 
                           (df_model["variant_binary_status"] == 1)
                          ]

        # get all mutations that are positive for inframe mutations. Ignore ambiguous variants in this count
        inframe = df_model.loc[(df_model["predicted_effect"].str.contains("inframe")) & 
                               (df_model["variant_binary_status"] == 1)
                              ]

        return len(lof["sample_id"].unique()), len(inframe["sample_id"].unique())
    
    else:
        return np.nan, np.nan
    

summary_df = pd.DataFrame(columns=["Drug", "Genos", "Binary_Phenos", "SNP_Matrix", "MICs", "Lineages", "Tier1_LOF", "Tier1_Inframe", "Tier2_LOF", "Tier2_Inframe"])
i = 0

print("Counting data for each drug")
for drug in drugs_for_analysis:
    
    # get all CSV files containing binary phenotypes
    pheno_files = os.listdir(os.path.join(phenos_dir, drug))
    phenos = pd.concat([pd.read_csv(os.path.join(phenos_dir, drug, fName), usecols=["sample_id"]) for fName in pheno_files if "run" in fName], axis=0)
    
    # just do tier 1, all samples in tier 1 are represented in tier 2
    geno_files = [os.path.join(genos_dir, drug, "tier=1", fName) for fName in os.listdir(os.path.join(genos_dir, drug, "tier=1")) if "run" in fName]
    genos = pd.concat([pd.read_csv(fName, usecols=["sample_id"]) for fName in geno_files], axis=0)
    
    # get all CSV files containing MICs
    mic_files = os.listdir(os.path.join(mic_dir, drug))
    mics = pd.concat([pd.read_csv(os.path.join(mic_dir, drug, fName), usecols=["sample_id"]) for fName in mic_files if "run" in fName], axis=0)
    
    # get numbers of samples represented in the GRM folder and with lineages (this will be less because not all sample_ids were matched to VCF files)
    num_with_snps = set(minor_allele_counts_samples).intersection(genos.sample_id.unique())
    samples_with_lineages = lineages.loc[lineages["Sample ID"].isin(genos["sample_id"])]
        
    # get the numbers of isolates, by tier
    num_lof_tier1, num_inframe_tier1 = compute_num_mutations(drug, ['1'])
    num_lof_tier2, num_inframe_tier2 = compute_num_mutations(drug, ['2'])
    
    summary_df.loc[i] = [drug.split("=")[1], 
                         len(genos.sample_id.unique()), 
                         len(phenos.sample_id.unique()), 
                         len(num_with_snps),
                         len(mics.sample_id.unique()),
                         len(samples_with_lineages),
                         num_lof_tier1,
                         num_inframe_tier1,
                         num_lof_tier2,
                         num_inframe_tier2
                        ]
    i += 1
        
    print("Finished", drug.split("=")[1])
    
    
# check and save
if len(summary_df.query("Genos != Binary_Phenos")) > 0:
    print("There are different numbers of genotypes and phenotypes")
if len(summary_df.query("Genos != SNP_Matrix")) > 0:
    print("There are different numbers of genotypes and SNP matrix entries")
    
summary_df.sort_values("Drug", ascending=True).to_csv("data/samples_summary.csv", index=False)
