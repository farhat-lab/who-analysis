import numpy as np
import pandas as pd
import glob, os, yaml, subprocess, sparse, sys


# genotypes broken down by gene (tier1 and tier2)
geno_dir = "/n/data1/hms/dbmi/farhat/ye12/who/full_genotypes"
analysis_dir = "/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue"

drugs_lst = os.listdir(analysis_dir)
drug_gene_df = pd.DataFrame(columns=["Drug", "Gene", "Tier"])

for drug in drugs_lst:
    
    tiers_dir = os.listdir(os.path.join(geno_dir, f"drug_name={drug}"))
    print(f"Found {len(tiers_dir)} tiers for {drug}")
    
    for single_tier_dir in tiers_dir:

        tier = os.path.join(geno_dir, f"drug_name={drug}", single_tier_dir)
        run_names = os.listdir(tier)
        print(f"Found {len(run_names)} tier {single_tier_dir[-1]} files for {drug}")

        for fName in run_names:

            full_name = os.path.join(tier, fName)
            # use awk to get the unique fields in the second column, which is the gene
            command = "awk -F',' 'NR>1{a[$2]++} END{for(b in a) print b}' " + full_name 
            proc = subprocess.Popen(command, 
                                    shell=True,
                                    encoding='utf8', 
                                    stdout=subprocess.PIPE
                                   )

            output, _ = proc.communicate()       
            
            # remove newline character from the end first, then split by newline character for cases of multiple genes
            for gene in output.strip("\n").split("\n"):
                drug_gene_df = pd.concat([drug_gene_df, pd.DataFrame({"Drug": drug, "Gene": gene, "Tier": int(single_tier_dir[-1])}, index=[0])], ignore_index=True) 
                
                
drug_gene_df.to_csv("data/drug_gene_mapping.csv", index=False)