import numpy as np
import pandas as pd
import sys, glob, os, yaml
import warnings
warnings.filterwarnings("ignore")
analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'


_, drug = sys.argv

# put the core analysis dataframe into both lists
analyses = []
insig_features = []
        
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
    
    # read in dataframes of insignificant features 
    insig_features.append(pd.read_csv(os.path.join(model_path, "insignificant_features.csv")))
      
# keep the first instance of every variant, which is in line with the order of models above
# IMPORTANT: THE ORDERING OF THE MODELS ABOVE MUST BE CORRECT
merged_df = pd.concat(analyses, axis=0).drop_duplicates(["variant"], keep="first")

# check that that NaN p-values occur only for variants with 0 standard error
if len(merged_df.loc[pd.isnull(merged_df["pval"])]) != 0:
    print(f"{drug} analysis dataframe contains NaN p-values")
else:
    print(f"Finished {drug}!")

merged_df.to_csv(os.path.join(os.path.join(analysis_dir, drug, "final_analysis.csv")), index=False)

# exclude variants that are in merged_df because this combined dataframe should contain only variants that are not found to be significant in ANY model
merged_df_insig = pd.concat(insig_features, axis=0).drop_duplicates(["variant"], keep="first")
merged_df_insig.query("variant not in @merged_df.variant").to_csv(os.path.join(os.path.join(analysis_dir, drug, "all_insignificant_features.csv")), index=False)