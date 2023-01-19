import numpy as np
import pandas as pd
import glob, os, yaml, tracemalloc, itertools, sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Bio import Seq, SeqIO


############# GENERATE THE EIGENVECTORS OF THE GENETIC RELATEDNESS MATRIX FOR ALL SAMPLES AND THE TABLE OF NUMBERS OF PHENOTYPES AND GENOTYPES ACROSS TIERS AND DRUGS #############

# The purpose of this script is to see how much data we have and check if we're missing anything. 
# It also generates the minor allele counts dataframe from the GRM directory, performs PCA, and saves the eigenvectors. 
# The eigenvectors are needed for lineage structure correction. The values for samples are extracted from this master dataframe.


# starting the memory monitoring -- this script needs ~125 GB
tracemalloc.start()

_, input_data_dir = sys.argv


################## STEP 1: CREATE THE MINOR ALLELE COUNTS DATAFRAME ##################


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
print(f"{len(snp_files)} SNP files to read from")

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
del matrix

# drop any columns that are 0 (major allele everywhere)
minor_allele_counts = minor_allele_counts[minor_allele_counts.columns[~((minor_allele_counts == 0).all())]]


###### STEP 2: COMPUTE THE GENETIC RELATEDNESS MATRIX, WHICH IS THE COVARIANCE OF THE MINOR ALLELE COUNTS DATAFRAME. SAVE EIGENVECTORS ######


# rows are samples, columns are sites
grm = np.cov(minor_allele_counts.values)
grm = pd.DataFrame(grm)
print(f"GRM shape: {grm.shape}")

minor_allele_counts_samples = minor_allele_counts.index.values
del minor_allele_counts

# scale before transforming
num_PCs = 10
pca = PCA(n_components=num_PCs)
scaler = StandardScaler()
pca.fit(scaler.fit_transform(grm))
del grm

print(f"Explained variance ratios of {num_PCs} principal components: {pca.explained_variance_ratio_}")
print(f"Sum of explained variance ratios: {np.sum(pca.explained_variance_ratio_)}")
np.save("data/pca_explained_var", pca.explained_variance_ratio_)

eigenvec = pca.components_.T
eigenvec_df = pd.DataFrame(eigenvec)
eigenvec_df["sample_id"] = minor_allele_counts_samples
eigenvec_df = eigenvec_df.set_index("sample_id")
eigenvec_df.columns = [f"PC{num}" for num in range(num_PCs)]
eigenvec_df.to_csv(f"data/eigenvec_{num_PCs}PC.csv")
    
    
###### STEP 3: COMPUTE NUMBERS OF SAMPLES WE HAVE DATA FOR, SEPARATED BY DRUG AND GENE TIER. INCLUDE NUMBER OF LINEAGES FOUND AND NUMBERS OF SAMPLES WITH LOF AND INFRAME (POOLED) MUTATIONS ######
    
    
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
                    
    dfs_lst = [pd.read_csv(fName, low_memory=False) for fName in geno_files]
        
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
                         num_inframe_tier2,
                        ]
    i += 1
        
    print("Finished", drug.split("=")[1])
    
    
# check and save
if len(summary_df.query("Genos != Binary_Phenos")) > 0:
    print("There are different numbers of genotypes and phenotypes")
if len(summary_df.query("Genos != SNP_Matrix")) > 0:
    print("There are different numbers of genotypes and SNP matrix entries")
    
summary_df.sort_values("Drug", ascending=True).to_csv("data/samples_summary.csv", index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB")