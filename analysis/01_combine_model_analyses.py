import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
import scipy.stats as st
import sys
import glob, os, yaml
import warnings
warnings.filterwarnings("ignore")
analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'


_, drug = sys.argv

# get the core model for the drug: tier 1 only, WHO phenos only, no synonymous mutations, pooling both LOF and inframe mutations
core_model_path = os.path.join(analysis_dir, drug, "tiers=1/phenos=WHO/dropAF_noSyn")
print(core_model_path)
core_analysis = pd.read_csv(os.path.join(core_model_path, "model_analysis.csv"))
core_analysis["Tier"] = 1
core_analysis["Phenos"] = "WHO"
core_analysis["unpooled"] = 0
core_analysis["synonymous"] = 0

# put the core analysis dataframe into both lists
binary_analyses = [core_analysis]
het_analyses = [core_analysis]

for tier in os.listdir(os.path.join(analysis_dir, drug)):
    tiers_path = os.path.join(analysis_dir, drug, tier)
    if os.path.isdir(tiers_path):
        for pheno in os.listdir(tiers_path):
            phenos_path = os.path.join(analysis_dir, drug, tier, pheno)
            for model in os.listdir(phenos_path):
                model_path = os.path.join(analysis_dir, drug, tier, pheno, model)
                    
                if model_path != core_model_path:
                    add_analysis = pd.read_csv(os.path.join(model_path, "model_analysis.csv"))
                    add_analysis["Tier"] = [2 if "+2" in model_path else 1][0]
                    add_analysis["Phenos"] = ["ALL" if "ALL" in model_path else "WHO"][0]
                    add_analysis["unpooled"] = int("unpooled" in model_path)
                    add_analysis["synonymous"] = int("withSyn" in model_path)
                    
                    if "encodeAF" not in model_path:
                        binary_analyses.append(add_analysis)
                    else:
                        het_analyses.append(add_analysis)
    else:
        print(f"Skipping {tiers_path} because it is not a directory")
      

# merge dataframes and write both to different files
merged_binary = pd.concat(binary_analyses, axis=0).drop_duplicates("orig_variant", keep="first")
del merged_binary["genome_index"]
merged_binary.sort_values("coef", ascending=False).to_csv(os.path.join(os.path.join(analysis_dir, drug, "final_analysis.csv")), index=False)

merged_het = pd.concat(het_analyses, axis=0).drop_duplicates("orig_variant", keep="first")
del merged_het["genome_index"]
merged_het.sort_values("coef", ascending=False).to_csv(os.path.join(os.path.join(analysis_dir, drug, "het_analysis.csv")), index=False)