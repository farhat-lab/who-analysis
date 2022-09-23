import numpy as np
import pandas as pd
import glob, os, yaml, sys, sparse
import warnings
warnings.filterwarnings("ignore")


############# GENERATE TABLE OF NUMBERS OF PHENOTYPES AND GENOTYPES ACROSS TIERS AND DRUGS #############

# the purpose of this script is to see how much data we have and check if we're missing anything

genos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/full_genotypes'
phenos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/phenotypes'

pheno_drugs = os.listdir(phenos_dir)
geno_drugs = os.listdir(genos_dir)

drugs_for_analysis = list(set(geno_drugs).intersection(set(pheno_drugs)))
print(len(drugs_for_analysis), "drugs with phenotypes and genotypes")


# get saved SNP matrix
if not os.path.isfile("data/minor_allele_counts.npz"): 
    print("Creating matrix of minor allele counts")
    # read in dataframe of loci associated with drug resistance
    drugs_loci = pd.read_csv("data/drugs_loci.csv")

    # add 1 to the start because it's 0-indexed
    drugs_loci["Start"] += 1
    assert sum(drugs_loci["End"] <= drugs_loci["Start"]) == 0

    # get all positions in resistance loci
    remove_pos = [list(range(int(row["Start"]), int(row["End"])+1)) for _, row in drugs_loci.iterrows()]
    remove_pos = list(itertools.chain.from_iterable(remove_pos))
    print(f"{len(remove_pos)} positions in resistance-determining regions will be removed")

    matrices = [pd.read_csv(os.path.join(matrix_dir, fName)) for fName in os.listdir(matrix_dir)]
    matrices_combined = pd.concat(matrices, axis=0).set_index("sample_id")

    # convert column names to integers because remove_pos are integers
    matrices_combined.columns = matrices_combined.columns.astype(int)

    # remove positions in resistance-determining genes
    matrices_combined = matrices_combined[matrices_combined.columns[~matrices_combined.columns.isin(remove_pos)]]

    assert np.nan not in matrices_combined.values

    # get the major alleles. Then compare --> set 1 for minor alleles, 0 for major
    major_alleles = matrices_combined.mode(axis=0)

    # put into dataframe to compare with the SNP dataframe
    major_alleles_df = pd.concat([major_alleles]*len(matrices_combined), ignore_index=True)
    major_alleles_df.index = matrices_combined.index.values

    assert matrices_combined.shape == major_alleles_df.shape
    minor_allele_counts = (matrices_combined != major_alleles_df).astype(int)

    # to save in sparse format, need to put the column names and indices into the dataframe, everything must be numerical
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
    minor_allele_counts = sparse.load_npz("data/minor_allele_counts.npz").todense()

    # convert to dataframe
    minor_allele_counts = pd.DataFrame(minor_allele_counts)
    minor_allele_counts.columns = minor_allele_counts.iloc[0, :]
    minor_allele_counts = minor_allele_counts.iloc[1:, :]
    minor_allele_counts.rename(columns={0:"sample_id"}, inplace=True)
    minor_allele_counts["sample_id"] = minor_allele_counts["sample_id"].astype(int)

    # make sample ids the index again
    minor_allele_counts = minor_allele_counts.set_index("sample_id")
    
    
    
summary_df = pd.DataFrame(columns=["Drug", "Tier", "Num_Phenos", "Num_Genos", "Num_SNP_Matrix"])
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
        
    if len(genos) < len(phenos):
        num_with_snps = minor_allele_counts.index.intersection(genos.sample_id.unique())
    else:
        num_with_snps = minor_allele_counts.index.intersection(phenos.sample_id.unique())

    summary_df.loc[i] = [drug.split("=")[1], 
                         1, 
                         len(phenos.sample_id.unique()), 
                         len(genos.sample_id.unique()), 
                         len(num_with_snps)]
    i += 1
        
    print("Finished", drug.split("=")[1])
    
    
assert sum(summary_df["Num_Phenos"] != summary_df["Num_Genos"]) == 0
summary_df.to_csv("data/num_avail_samples.csv", index=False)