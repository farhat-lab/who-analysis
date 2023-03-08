import numpy as np
import pandas as pd
import glob, os, yaml, tracemalloc, itertools, sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# starting the memory monitoring -- this script needs ~125 GB
tracemalloc.start()

# mixed_sites_thresh = 100
# data_dir = "/n/data1/hms/dbmi/farhat/ye12/who/grm_low_VAF"
_, data_dir, mixed_sites_thresh = sys.argv
mixed_sites_thresh = int(mixed_sites_thresh)

coll2014 = pd.read_csv("data/coll2014_SNP_scheme.tsv", sep="\t")
coll2014["#lineage"] = coll2014["#lineage"].str.replace("lineage", "")
coll2014.rename(columns={"#lineage": "lineage"}, inplace=True)


# 0.1% VAF dataframe
low_VAF_minor_alleles = pd.read_csv(f"{data_dir}/{os.listdir(data_dir)[0]}")


############################################ GET SITES WITH MANY MIXED CALLS ############################################


# mixed calls information
mixed_calls = pd.read_excel("data/mixed_site_counts.xlsx", sheet_name=None)

if len(mixed_calls) == 1:
    mixed_calls = mixed_calls[list(mixed_calls.keys())[0]]

low_VAF_sites = low_VAF_minor_alleles["position"].unique()
mixed_sites = mixed_calls.query("genotype_count > @mixed_sites_thresh").position.unique()


###################### GET SITES IN DRUG RESISTANCE LOCI THAT ARE NOT IN THE COLL 2014 SCHEME ######################


# drop sites that are in drug resistance loci
drugs_loci = pd.read_csv("data/drugs_loci.csv")

# add 1 to the start because it's 0-indexed
drugs_loci["Start"] += 1
assert sum(drugs_loci["End"] <= drugs_loci["Start"]) == 0

# get all positions in resistance loci
drug_res_sites = [list(range(int(row["Start"]), int(row["End"])+1)) for _, row in drugs_loci.iterrows()]
drug_res_sites = list(itertools.chain.from_iterable(drug_res_sites))
drug_res_sites = set(drug_res_sites) - set(coll2014["position"].values.astype(int))


############################## GET HOMPLASIC SITES THAT ARE NOT IN THE COLL 2014 SCHEME ##############################


homoplasy = pd.read_excel("data/Vargas_homoplasy.xlsx")
if len(homoplasy) == 1:
    homoplasy = homoplasy[list(homoplasy.keys())[0]]
    
homoplasic_sites = set(homoplasy["H37Rv Position"].values.astype(int)) - set(coll2014["position"].values.astype(int))


########################################### GET ONLY SAMPLES IN THE DATASET ###########################################

    
eigenvec_df = pd.read_csv("data/eigenvec_100PC.csv", usecols=[0])
keep_samples = eigenvec_df["sample_id"].values


############################## REMOVE ALL SITES TO BE DROPPED FROM THE ORIGINAL VAF DF ##############################

    
keep_sites = list(set(low_VAF_sites) - set(mixed_sites) - set(drug_res_sites) - set(homoplasic_sites))

print(f"Keeping {len(keep_sites)}/{len(low_VAF_sites)} sites for PCA")
print(f"Keeping {len(keep_samples)}/{len(low_VAF_minor_alleles['sample_id'].unique())} samples for PCA")


low_VAF_minor_alleles = low_VAF_minor_alleles.query("position in @keep_sites & sample_id in @keep_samples")

# add 0s for these samples after pivoting
missing_samples = list(set(keep_samples) - set(low_VAF_minor_alleles["sample_id"].unique()))
print(f"{len(missing_samples)} missing samples to fill in with 0s")



###################################### PIVOT TO MATRIX AND PERFORM PCA ######################################


print("Pivoting to a matrix...")
# during the pivoting, if the number of positions is less than the value above, it's because for the remaining samples, all alleles were major
# so the entire column would be 0, and it's irrelevant information anyway
low_VAF_minor_alleles = low_VAF_minor_alleles.pivot(index="sample_id", columns="position", values="minor_count").fillna(0).astype(int)
print(low_VAF_minor_alleles.shape)

# add in missing samples, which should be all 0
low_VAF_minor_alleles = pd.concat([low_VAF_minor_alleles, pd.DataFrame([[0]*low_VAF_minor_alleles.shape[1]], columns=low_VAF_minor_alleles.columns, index=missing_samples)], axis=0)
print(low_VAF_minor_alleles.shape)

low_VAF_minor_alleles.to_pickle(f"{data_dir}/minor_allele_counts.pkl")

# rows are samples, columns are sites
print("Computing the GRM...")
scaler = StandardScaler()
grm = np.cov(scaler.fit_transform(low_VAF_minor_alleles.values))
print(f"GRM shape: {grm.shape}")

minor_allele_counts_samples = low_VAF_minor_alleles.index.values
del low_VAF_minor_alleles

# scale before transforming
print("Performing PCA...")
num_PCs = 100
pca = PCA(n_components=num_PCs)
pca.fit(scaler.fit_transform(grm))
del grm

print(f"Sum of explained variance ratios: {np.sum(pca.explained_variance_ratio_)}")
np.save("data/pca_explained_var_low_VAF", pca.explained_variance_ratio_)

eigenvec = pca.components_.T
eigenvec_df = pd.DataFrame(eigenvec)
eigenvec_df["sample_id"] = minor_allele_counts_samples
eigenvec_df = eigenvec_df.set_index("sample_id")
eigenvec_df.columns = [f"PC{num}" for num in range(num_PCs)]
eigenvec_df.to_csv(f"data/eigenvec_{num_PCs}PC_low_VAF.csv")

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB")