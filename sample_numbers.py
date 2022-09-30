import numpy as np
import pandas as pd
import glob, os, yaml, sys, sparse, itertools
from Bio import Seq, SeqIO
import warnings
warnings.filterwarnings("ignore")


############# GENERATE TABLE OF NUMBERS OF PHENOTYPES AND GENOTYPES ACROSS TIERS AND DRUGS #############

# the purpose of this script is to see how much data we have and check if we're missing anything

narrow_dir = '/n/data1/hms/dbmi/farhat/ye12/who/narrow_format' 
latest_narrow_dir = '/n/data1/hms/dbmi/farhat/ye12/who/latest_narrow_format' 
genos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/full_genotypes'
phenos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/phenotypes'

pheno_drugs = os.listdir(phenos_dir)
geno_drugs = os.listdir(genos_dir)

drugs_for_analysis = list(set(geno_drugs).intersection(set(pheno_drugs)))
print(len(drugs_for_analysis), "drugs with phenotypes and genotypes")

lineages = pd.read_pickle("data/combined_lineage_sample_IDs.pkl")

# get saved SNP matrix
if not os.path.isfile("data/minor_allele_counts.npz"): 
    print("Creating matrix of minor allele counts")
    
    print("Reading in all files in the narrow directories")
    narrow_files = [pd.read_csv(os.path.join(narrow_dir, fName)) for fName in os.listdir(narrow_dir)]
    narrow = pd.concat(narrow_files, axis=0)
    print(len(narrow_files), "files in the narrow directory")

    # ignore last (4th) column
    latest_narrow_files = [pd.read_csv(os.path.join(latest_narrow_dir, fName), 
                                       usecols=[0, 1, 2]) for fName in os.listdir(latest_narrow_dir)]
    latest_narrow = pd.concat(latest_narrow_files, axis=0)
    print(len(latest_narrow_files), "files in the latest narrow directory")
    
    # if this is 0, then it means that narrow is a subset of latest_narrow
    if len(set(narrow.sample_id) - set(latest_narrow.sample_id)) == 0:
        print("narrow directory contains a subset of the data in the latest narrow directory")
        new_samples = list(set(latest_narrow.sample_id) - set(narrow.sample_id))
        narrow_keep = pd.concat([narrow, latest_narrow.query("sample_id in @new_samples").rename(columns={latest_narrow.columns[-1]: "nucleotide"})], axis=0)
    # if not, need to concatenate the full dataframes and then drop duplicates
    else:
        all_samples = list(set(narrow.sample_id).union(set(latest_narrow.sample_id)))
        narrow_keep = pd.concat([narrow, latest_narrow.rename(columns={latest_narrow.columns[-1]: "nucleotide"})], axis=0).drop_duplicates()
    print(len(narrow_keep.position.unique()), "positions before dropping resistance-determing regions")

    # read in dataframe of loci associated with drug resistance
    drugs_loci = pd.read_csv("data/drugs_loci.csv")

    # add 1 to the start because it's 0-indexed
    drugs_loci["Start"] += 1
    assert sum(drugs_loci["End"] <= drugs_loci["Start"]) == 0

    # get all positions in resistance loci
    remove_pos = [list(range(int(row["Start"]), int(row["End"])+1)) for _, row in drugs_loci.iterrows()]
    remove_pos = list(itertools.chain.from_iterable(remove_pos))
    
    narrow_keep = narrow_keep.query("position not in @remove_pos")
    print(len(narrow_keep.position.unique()), "positions after dropping resistance-determing regions")
    
    # read in the H37Rv reference genome to get reference nucleotides
    genome = SeqIO.read("/n/data1/hms/dbmi/farhat/Sanjana/GCF_000195955.2_ASM19595v2_genomic.gbff", "genbank")
    genome_seq = list(genome.seq)
    
    # add reference alleles to the dataframe, then get only rows where the allele is not reference
    pos_ref_dict = dict(zip(narrow_keep.position.unique(), np.array([genome_seq[pos-1] for pos in narrow_keep.position.unique()])))
    narrow_keep["reference"] = narrow_keep["position"].map(pos_ref_dict)
    narrow_keep = narrow_keep.query("nucleotide != reference")
    
    # pivot to matrix, NaNs with the reference alleles and then get the major allele at every site. This is a very long step
    print("Pivoting to a matrix")
    matrix = narrow_keep.pivot(index="sample_id", columns="position", values="nucleotide")
    matrix = matrix.fillna(pos_ref_dict)
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


else:
    print("Loading in existing minor alleles dataframe")
    minor_allele_counts = sparse.load_npz("data/minor_allele_counts.npz").todense()

    # convert to dataframe
    minor_allele_counts = pd.DataFrame(minor_allele_counts)
    minor_allele_counts.columns = minor_allele_counts.iloc[0, :]
    minor_allele_counts = minor_allele_counts.iloc[1:, :]
    minor_allele_counts.rename(columns={0:"sample_id"}, inplace=True)
    minor_allele_counts["sample_id"] = minor_allele_counts["sample_id"].astype(int)

    # make sample ids the index again
    minor_allele_counts = minor_allele_counts.set_index("sample_id")
    
    

        
def compute_number_frameshift_by_tier(drug, tiers_lst, synonymous=False):
    
    geno_files = []

    for subdir in os.listdir(os.path.join(genos_dir, drug)):

        # subdirectory (tiers)
        full_subdir = os.path.join(genos_dir, drug, subdir)

        # the last character is the tier number
        if subdir[-1] in tiers_lst:

            # the last character is the tier number
            for fName in os.listdir(full_subdir):

                # some hidden files (i.e. Git files) are present, so ignore them
                if fName[0] != ".":
                    geno_files.append(os.path.join(full_subdir, fName))

    dfs_lst = []
    for i, fName in enumerate(geno_files):
        df = pd.read_csv(fName)

        # keep all variants
        if synonymous:
            dfs_lst.append(df)
        else:
            # P = coding variants, C = synonymous or upstream variants (get only upstream variants by getting only negative positions), and N = non-coding variants on rrs/rrl
            # deletion does not contain the p/c/n prefix
            dfs_lst.append(df.query("Effect not in ['synonymous_variant', 'stop_retained_variant']"))

    if len(dfs_lst) > 0:
        df_model = pd.concat(dfs_lst)

        # get all frameshift mutations, only those for which variant_allele_frequency != 0
        frameshift = df_model.loc[(df_model["variant_category"].str.contains("fs")) & 
                                  (~pd.isnull(df_model["variant_allele_frequency"])) &
                                  (df_model["variant_allele_frequency"] > 0)
                                 ]

        # (sample, gene) pairs with multiple frameshift mutations
        lof_multi_fs = pd.DataFrame(frameshift.groupby(["sample_id", "resolved_symbol"])["variant_category"].count()).query("variant_category > 1").reset_index()
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
    phenos = pd.concat([pd.read_csv(os.path.join(phenos_dir, drug, fName), usecols=["sample_id"]) for fName in pheno_files], axis=0)
    
    # just do tier 1, all samples in tier 1 are represented in tier 2
    geno_files = [os.path.join(genos_dir, drug, "tier=1", fName) for fName in os.listdir(os.path.join(genos_dir, drug, "tier=1"))]
    genos = pd.concat([pd.read_csv(fName, usecols=["sample_id"]) for fName in geno_files], axis=0)
    
    assert len(genos.sample_id.unique()) == len(phenos.sample_id.unique())
    num_with_snps = minor_allele_counts.index.intersection(genos.sample_id.unique())
    samples_with_lineages = lineages.loc[lineages["Sample ID"].isin(genos["sample_id"])]
        
    # get the number of isolates with multiple frameshift mutations in them, by tier
    num_fs_tier1 = compute_number_frameshift_by_tier(drug, ['1'], synonymous=False)
    num_fs_tier2 = compute_number_frameshift_by_tier(drug, ['2'], synonymous=False)
    
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
# (if so, don't need to keep figuring out the VCF files that don't have sample IDs)
assert sum(summary_df["Genos"] != summmary_df["Lineages"]) == 0
summary_df.to_csv("data/samples_summary.csv", index=False)