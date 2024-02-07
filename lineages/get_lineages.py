import numpy as np
import pandas as pd
import glob, os, subprocess, sys

# pass in 1. a CSV file containing full pathgs to VCF files and 2. the name of the output CSV file you want to save the data to
_, vcf_files_fName, lineages_out_file = sys.argv

vcf_files_lst = pd.read_csv(vcf_files_fName, header=None)[0].values

#  each VCF file is in a folder with the run name
print(len(vcf_files_lst), "VCF files")

# check that the output file is in a valid directory
if not os.path.isdir(os.path.dirname(lineages_out_file)):
    raise ValueError(f"{os.path.dirname(lineages_out_file)} directory does not exist!")
    
lineages_df = pd.DataFrame(columns=["File_Name", "Coll2014", 'Freschi2020', 'Lipworth2019', 'Shitikov2017', 'Stucki2016'])

for i, vcf_fName in enumerate(vcf_files_lst):
    
    if not os.path.isfile(vcf_fName):
        raise ValueError(f"{vcf_fName} is not a file")
    else:
        proc = subprocess.Popen(f"fast-lineage-caller {vcf_fName} --noheader --pass", shell=True, encoding='utf8', stdout=subprocess.PIPE)
        output = proc.communicate()[0].replace('\n', '') # remove newline characters
        
        # order: Isolate, Coll 2014, Freschi 2020, lipworth2019, shitikov2017, stucki2016
        # output is tab-separated, so separate that way
        lineages_df.loc[i, :] = [vcf_fName, 
                                 output.split("\t")[1].replace("lineage", ""), 
                                 output.split("\t")[2],
                                 output.split("\t")[3],
                                 output.split("\t")[4],
                                 output.split("\t")[5]
                                ]

    # print progress and save intermediate versions
    if i % 1000 == 0:
        lineages_df.to_csv(lineages_out_file, index=False)
        print(i)
        
print(f"Finished {len(lineages_df)}/{len(vcf_files_lst)}")
lineages_df.to_csv(lineages_out_file, index=False)