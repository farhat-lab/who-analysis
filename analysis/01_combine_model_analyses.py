import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
import scipy.stats as st
import sys

import glob, os, yaml
import warnings
warnings.filterwarnings("ignore")
import tracemalloc
analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'


# starting the memory monitoring
tracemalloc.start()

_, drug = sys.argv

# get the core model for the drug: tier 1 only, WHO phenos only, no synonymous mutations, pool LOF (if it's not different from unpool LOF, use the unpooled model)
core_model_path = os.path.join(analysis_dir, drug, "tiers=1/phenos=WHO/dropAF_noSyn")

core_analysis = pd.read_csv(os.path.join(core_model_path, "model_analysis.csv"))
core_analysis["Tier"] = 1
core_analysis["Phenos"] = "WHO"
core_analysis["unpooled"] = 0
core_analysis["synonymous"] = 0

# create dictionary of the additional model analyses
add_analyses = {}

for tier in os.listdir(os.path.join(out_dir, drug)):
    tiers_path = os.path.join(out_dir, drug, tier)
    if os.path.isdir(tiers_path):
        for pheno in os.listdir(tiers_path):
            phenos_path = os.path.join(out_dir, drug, tier, pheno)
            for model in os.listdir(phenos_path):
                model_path = os.path.join(out_dir, drug, tier, pheno, model)

                analysis_fName = os.path.join(model_path, "model_analysis.csv")
                if os.path.isfile(analysis_fName):
                    
                    # only include models where HETs were dropped because we can not compute univariate stats for continuous AFs
                    if (model_path != core_model_path) & ("encodeAF" not in model_path):
                        add_analysis = pd.read_csv(analysis_fName)
                        add_path = model_path.split(os.path.join(out_dir, drug))[1].strip("/").replace("dropAF_", "")

                        add_analysis["Tier"] = [2 if "+2" in add_path else 1][0]
                        add_analysis["Phenos"] = ["ALL" if "ALL" in add_path else "WHO"][0]
                        add_analysis["unpooled"] = int("unpooled" in add_path)
                        add_analysis["synonymous"] = int("withSyn" in add_path)

                        # keep all significant variants at an FDR threshold of 0.01
                        add_analyses[add_path] = add_analysis
                else:
                    print(f"No model analysis file in {analysis_fName}")
    else:
        print(f"Skipping {tiers_path} because it is not a directory")
      
keys_lst = list(add_analyses.keys())

# merge the core analysis dataframe with the first additional dataframe
merged_df = pd.concat([core_analysis, add_analyses[keys_lst[0]]], axis=0).drop_duplicates("orig_variant", keep="first")

# merge the merged dataframe with the remaining additional analyses, dropping duplicates as before
for key in keys_lst[1:]:
    merged_df = pd.concat([merged_df, add_analyses[key]], axis=0).drop_duplicates("orig_variant", keep="first")

merged_df.to_csv(os.path.join(os.path.join(out_dir, drug, "final_analysis.csv")), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()

# write peak memory usage in GB
with open("memory_usage.log", "a+") as file:
    file.write(f"{os.path.basename(__file__)}: {script_memory} GB\n")