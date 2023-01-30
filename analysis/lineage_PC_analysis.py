import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
import seaborn as sns

import glob, os, yaml, subprocess, itertools, sparse, sys
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")

_, out_dir = sys.argv


########## READ IN THE MODEL DATAFRAME ##########

print("Reading in model input matrix")
model_matrix = pd.read_pickle(os.path.join(out_dir, "model_matrix.pkl"))
model_analysis = pd.read_csv(os.path.join(out_dir, "model_analysis.csv"))

# number of principal components with positive coefficients (OR > 1)
sig_PC_df = model_analysis.loc[(model_analysis["orig_variant"].str.contains("PC")) & 
                              (((model_analysis["coef_LB"] > 0) & (model_analysis["coef_UB"] > 0)) |
                              ((model_analysis["coef_LB"] < 0) & (model_analysis["coef_UB"] < 0)))
                              ]

# plot the first 2 PCs that have coefficients with the largest absolute values
sig_PC_df["abs_coef"] = np.abs(sig_PC_df["coef"])
sig_PCs = sig_PC_df.sort_values("abs_coef", ascending=False)["orig_variant"].values
del model_analysis

# stop the program if there are no significant principal components
if len(sig_PCs) == 0:
    print("There are no significant principal components in this regression model. Exiting script...")
    exit()
else:
    print(sig_PCs)
    

########## READ IN THE MINOR ALLELE COUNTS DATAFRAME ##########

print("Reading in minor alleles dataframe")
minor_allele_counts = sparse.load_npz("data/minor_allele_counts.npz").todense()

# convert to dataframe
minor_allele_counts = pd.DataFrame(minor_allele_counts)
minor_allele_counts.columns = minor_allele_counts.iloc[0, :]
minor_allele_counts = minor_allele_counts.iloc[1:, :]
minor_allele_counts.rename(columns={0:"sample_id"}, inplace=True)
minor_allele_counts["sample_id"] = minor_allele_counts["sample_id"].astype(int)

# make sample ids the index again
minor_allele_counts = minor_allele_counts.set_index("sample_id")


########## READ IN AND CLEAN THE LINEAGES DATAFRAME ##########


print("Reading in lineages dataframe")

# the missing ones might be M. cannettii, most similar to L6 based on the other lineage callers. This file has 70,567 samples
lineages = pd.read_pickle("data/combined_lineage_sample_IDs.pkl")
lineages["Lineage"] = lineages["Lineage"].fillna("6")
lineages["Lineage_1"] = lineages["Lineage_1"].fillna("6")

lineages = lineages[["Sample Name", "Sample ID", "Lineage_1"]]
lineages["Lineage"] = [str(val).split(".")[0] for val in lineages["Lineage_1"].values]
lineages.loc[lineages["Lineage"].str.contains("BOV"), "Lineage"] = "M. bovis"

assert len(lineages.loc[pd.isnull(lineages["Lineage"])]) == 0


########## KEEP ONLY ISOLATES WITH ALL 3 PIECES OF DATA ##########


# get only isolates with data for everyting: SNP matrix, in the model, and lineages
keep_isolates = np.array(list(set(minor_allele_counts.index).intersection(model_matrix.index).intersection(lineages["Sample ID"])))

minor_allele_counts = minor_allele_counts.loc[keep_isolates, :].sort_values(by="sample_id", axis='index')
lineages = lineages.loc[lineages["Sample ID"].isin(keep_isolates)].sort_values("Sample ID")

del model_matrix


########## COMPUTE GENETIC RELATEDNESS MATRIX ##########

print("Computing the GRM and performing PCA")
grm = np.cov(minor_allele_counts.values)

scaler = StandardScaler()
n = 5
pca = PCA(n_components=n)
pca.fit(scaler.fit_transform(grm))
    
eigenvec = pca.components_.T
eigenvec_df = pd.DataFrame(eigenvec)
eigenvec_df.columns = [f"PC{i}" for i in range(n)]

eigenvec_df["Sample ID"] = minor_allele_counts.index.values
eigenvec_df = eigenvec_df.merge(lineages, on="Sample ID", how="inner")

fig, ax = plt.subplots()

# plot the first 2 principal components in the list of significant ones. Lower numbers explain the data more
if len(sig_PCs) >= 2:
    sns.scatterplot(data=eigenvec_df, x=sig_PCs[0], y=sig_PCs[1], hue="Lineage", hue_order=['1', '2', '3', '4', '5', '6', '7', 'M. bovis'], ax=ax)
# if there is only 1 significant PC, then plot it against the 5th PC, which will explain the data the least. Basically an arbitrary decision because we only care about the other axis
else:
    sns.scatterplot(data=eigenvec_df, x=sig_PCs[0], y="PC5", hue="Lineage", hue_order=['1', '2', '3', '4', '5', '6', '7', 'M. bovis'], ax=ax)
ax.legend(bbox_to_anchor=(1.1, 1.05))
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(out_dir, "significant_pc_plot.png"), dpi=300)