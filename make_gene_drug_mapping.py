import numpy as np
import pandas as pd
import glob, os, yaml, subprocess, sparse, sys, tracemalloc

_, input_data_dir = sys.argv

# starting the memory monitoring
tracemalloc.start()

genos_dir = os.path.join(input_data_dir, "full_genotypes")
phenos_dir = os.path.join(input_data_dir, "phenotypes")
mic_dir = os.path.join(input_data_dir, "mic")

pheno_drugs = os.listdir(phenos_dir)
geno_drugs = os.listdir(genos_dir)
mic_drugs = os.listdir(mic_dir)

drugs_lst = np.sort(list(set(geno_drugs).intersection(set(pheno_drugs)).intersection(set(mic_drugs))))
print(len(drugs_lst), "drugs with phenotypes and genotypes")

drug_gene_df = pd.DataFrame(columns=["Drug", "Gene", "Tier"])

for drug in drugs_lst:
    
    drug = drug.split('=')[1]
    
    tiers_dir = os.listdir(os.path.join(genos_dir, f"drug_name={drug}"))
    
    for single_tier_dir in tiers_dir:

        tier = os.path.join(genos_dir, f"drug_name={drug}", single_tier_dir)
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

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB")