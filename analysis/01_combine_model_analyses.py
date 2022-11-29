import numpy as np
import pandas as pd
import sys, glob, os, yaml
import warnings
warnings.filterwarnings("ignore")
analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'


_, drug = sys.argv

analyses = []
        
analysis_paths = ["tiers=1/phenos=WHO/dropAF_noSyn",
                  "tiers=1/phenos=WHO/dropAF_noSyn_unpooled",
                  "tiers=1/phenos=WHO/dropAF_withSyn",
                  "tiers=1+2/phenos=WHO/dropAF_noSyn",
                  "tiers=1+2/phenos=WHO/dropAF_noSyn_unpooled",
                  "tiers=1+2/phenos=WHO/dropAF_withSyn",
                  "tiers=1/phenos=ALL/dropAF_noSyn",
                  "tiers=1/phenos=ALL/dropAF_noSyn_unpooled",
                  "tiers=1/phenos=ALL/dropAF_withSyn",
                  "tiers=1+2/phenos=ALL/dropAF_noSyn",
                  "tiers=1+2/phenos=ALL/dropAF_noSyn_unpooled",
                  "tiers=1+2/phenos=ALL/dropAF_withSyn",
                  "tiers=1/phenos=WHO/encodeAF_noSyn",
                  "tiers=1+2/phenos=WHO/encodeAF_noSyn",
                  "tiers=1/phenos=ALL/encodeAF_noSyn",
                  "tiers=1+2/phenos=ALL/encodeAF_noSyn",
]


for path in analysis_paths:
    
    model_path = os.path.join(analysis_dir, drug, path)
    
    add_analysis = pd.read_csv(os.path.join(model_path, "model_analysis.csv"))
    add_analysis["Tier"] = [2 if "+2" in model_path else 1][0]
    add_analysis["Phenos"] = ["ALL" if "ALL" in model_path else "WHO"][0]
    add_analysis["unpooled"] = int("unpooled" in model_path)
    add_analysis["synonymous"] = int("withSyn" in model_path)
    add_analysis["HET"] = ["DROP" if "drop" in model_path else "AF"][0]
    analyses.append(add_analysis)
    
# keep the first instance of every variant, which is in line with the order of models above
# IMPORTANT: THE ORDERING OF THE MODELS ABOVE MUST BE CORRECT
merged_df = pd.concat(analyses, axis=0)
merged_df.to_csv(os.path.join(os.path.join(analysis_dir, drug, "full_analysis.csv")), index=False)

# merged_df = merged_df.drop_duplicates(["variant"], keep="first")

# check that that NaN p-values occur only for variants with 0 standard error
if len(merged_df.loc[pd.isnull(merged_df["pval"])]) != 0:
    print(f"{drug} analysis dataframe contains NaN p-values")
else:
    print(f"\nFinished {drug}!")

# merged_df.to_csv(os.path.join(os.path.join(analysis_dir, drug, "final_analysis.csv")), index=False)