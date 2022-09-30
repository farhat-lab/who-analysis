import numpy as np
import pandas as pd
import glob, os, subprocess, sys


#  each VCF file is in a folder with the run name
vcf_dir = "/n/data1/hms/dbmi/farhat/ye12/who/vcf_files"
vcf_dirs_files = glob.glob(vcf_dir + "/*")
print(len(vcf_dirs_files), "VCF files")


def get_lineages(vcf_dirs_files):
    
    # keep track of failures
    problem_dirs = []
    
    lineages = []
    
    for i, single_dir in enumerate(vcf_dirs_files):
        
        fName = os.listdir(single_dir)
        if len(fName) > 1:
            problem_dirs.append(single_dir)
        else:
            fName = os.path.join(single_dir, fName[0])        
        
        if not os.path.isfile(fName):
            raise ValueError(f"{fName} is not a file")
        else:
            proc = subprocess.Popen(f"fast-lineage-caller {fName} --noheader --count", shell=True, encoding='utf8', stdout=subprocess.PIPE)
            output = proc.communicate()[0]

            # the second value is the Freschi et al lineage
            lineages.append(output.split("\t")[1].replace("lineage", ""))
            
        if i % 1000 == 0:
            print(i)
        
    return pd.DataFrame({"VCF_Dir": vcf_dirs_files, "Lineage": lineages}), problem_dirs


lineages, problem_dirs = get_lineages(vcf_dirs_files)
print(f"{len(lineages)}/{len(vcf_dirs_files)}")
print(len(problem_dirs), "directories contain multiple files:\n")
print(problem_dirs)

# split multiple lineages per isolate
split_lineages = lineages["Lineage"].str.split(",", expand=True)
split_lineages.columns = [f"Lineage_{num}" for num in np.arange(len(split_lineages.columns))+1]

# separate lineage and SNP count for each one
for col in split_lineages:
    count_col = f"Count_{col.split('_')[1]}"
    split_lineages[[col, count_col]] = split_lineages[col].str.split("(", expand=True)
    split_lineages[count_col] = split_lineages[count_col].str.strip(")")

# save
lineages = pd.concat([lineages, split_lineages], axis=1)

# multiple directories contain the same VCF files?
#lineages = lineages.drop_duplicates(subset=["Sample Name", "Sample ID", "Lineage"])
lineages.to_csv("data/lineages.csv", index=False)