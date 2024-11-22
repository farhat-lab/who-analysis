import numpy as np
import pandas as pd
import glob, os, yaml, tracemalloc, itertools, sys, argparse, pickle, subprocess


for drug in os.listdir("/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue"):

    #cat * > /n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue/Amikacin/genos_1.csv

    out_fName = f"/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue/{drug}/genos_2.csv.gz"
    
    try:
        if not os.path.isfile(out_fName):
            df_combined = pd.concat([pd.read_csv(fName, low_memory=False) for fName in glob.glob(f"/n/scratch/users/s/sak0914/who-mutation-catalogue-raw-data/full_genotypes/drug_name={drug}/tier=2/*")])
    
            df_combined.to_csv(out_fName, compression='gzip', index=False)
        print(f"Finished {drug}")
    except:
        print(f"Failed {drug}")