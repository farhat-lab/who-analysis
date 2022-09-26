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
    
    isolates = []
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
            isolates.append(os.path.basename(fName).split("-")[0])
            
        if i % 1000 == 0:
            print(i)
        
    return pd.DataFrame({"Isolate": isolates, "Lineage": lineages}), problem_dirs


lineages, problem_dirs = get_lineages(vcf_dirs_files)
print(f"{len(lineages)}/{len(vcf_dirs_files)}")
print(len(problem_dirs), "problematic directories:\n")
print(problem_dirs)

lineages.to_csv("data/lineages.csv", index=False)