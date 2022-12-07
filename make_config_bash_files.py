import sys, glob, os, yaml
import numpy as np
import pandas as pd


###### CONFIG FILES FOR THE BINARY ANALYSIS: SHOULD BE 16 TOTAL ######

# order of parameters to be updated: pheno_category_lst, tiers_lst, unpooled, synonymous, amb_mode
all_combos = [[["WHO"], ["1"], False, False, "DROP"],
              [["WHO"], ["1"], True, False, "DROP"],
              [["WHO"], ["1"], False, True, "DROP"],
              [["WHO"], ["1", "2"], False, False, "DROP"],
              [["WHO"], ["1", "2"], True, False, "DROP"],
              [["WHO"], ["1", "2"], False, True, "DROP"],
              [["ALL"], ["1"], False, False, "DROP"],
              [["ALL"], ["1"], True, False, "DROP"],
              [["ALL"], ["1"], False, True, "DROP"],
              [["ALL"], ["1", "2"], False, False, "DROP"],
              [["ALL"], ["1", "2"], True, False, "DROP"],
              [["ALL"], ["1", "2"], False, True, "DROP"],
              [["WHO"], ["1"], False, False, "AF"],
              [["WHO"], ["1", "2"], False, False, "AF"],
              [["ALL"], ["1"], False, False, "AF"],
              [["ALL"], ["1", "2"], False, False, "AF"]
            ]

# example set of kwargs
kwargs = yaml.safe_load(open("config.yaml"))


# config files run from 1 - len(all_combos)
for i in list(range(1, len(all_combos)+1)):
        
    # if the number is less than 10, add a 0 in front of it to keep them in order
    if i < 10:
        num_str = f"0{i}"
    else:
        num_str = str(i)
    
    with open(f"config_files/binary_{num_str}.yaml", "w+") as file:
        
        # constant for all cases
        kwargs["binary"] = True
        kwargs["atu_analysis"] = False
        
        # update param combinations and write to the file
        param_dict = dict(zip(["pheno_category_lst", "tiers_lst", "unpooled", "synonymous", "amb_mode"], all_combos[i-1]))
        kwargs.update(param_dict)
        yaml.dump(kwargs, file, default_flow_style=False, sort_keys=False)
    
    i += 1
    
    
    
###### BASH SCRIPTS FOR EACH DRUG: SHOULD BE 15 TOTAL ######
    
# manually write bash_scripts/run_AMI.sh, then copied everything else from there and updated drug names
    
drug_abbr_dict = {"Delamanid": "DLM",
                  "Bedaquiline": "BDQ",
                  "Clofazimine": "CFZ",
                  "Ethionamide": "ETH",
                  "Linezolid": "LZD",
                  "Moxifloxacin": "MXF",
                  "Capreomycin": "CAP",
                  "Amikacin": "AMI",
                  "Pyrazinamide": "PZA",
                  "Kanamycin": "KAN",
                  "Levofloxacin": "LEV",
                  "Streptomycin": "STM",
                  "Ethambutol": "EMB",
                  "Isoniazid": "INH",
                  "Rifampicin": "RIF"
                 }

drug_names = list(drug_abbr_dict.keys())
    
with open("bash_scripts/run_AMI.sh", "r+") as file:
    lines = file.readlines()

# copy the bash script from run_AMI.sh to all drugs
for i, drug in enumerate(drug_names):

    with open(f"bash_scripts/run_{drug_abbr_dict[drug]}.sh", "w+") as new_file:

        for line in lines:

            # update drug name and WHO abbreviation
            if "drug=" in line:
                new_file.write(f'drug="{drug}"\n')
            elif "drug_abbr=" in line:
                new_file.write(f'drug_abbr="{drug_abbr_dict[drug]}"\n')
            else:
                new_file.write(line)
                
                
    
# ###### TODO: CONFIG FILES FOR THE MIC ANALYSIS: SHOULD BE 8 TOTAL (SO FAR) ######

# # not relevant, but the parameter will get ignored in the scripts
# phenos = ["WHO"]
# tiers = [["1"], ["1", "2"]]
# unpooled = [False, True]
# syn = [False, True]
# amb_mode = ["DROP", "AF"]

# all_combos = list(itertools.product(*[phenos, tiers, unpooled, syn, amb_mode]))
# print(len(all_combos))

# # example set of kwargs
# kwargs = yaml.safe_load(open("config.yaml"))

# # config files run from 1 - len(all_combos)
# for i in list(range(len(1, all_combos+1))):
        
#     # if the number is less than 10, add a 0 in front of it to keep them in order
#     if i < 10:
#         num_str = f"0{i}"
#     else:
#         num_str = str(i)
    
#     with open(f"config_files/mic_{num_str}.yaml", "r+") as file:
        
#         kwargs["binary"] = True
#         kwargs["atu_analysis"] = False
#         yaml.dump(kwargs, file, default_flow_style=False, sort_keys=False)
    
#     i += 1


###### TODO: CONFIG FILES FOR THE CC vs. CC-ATU ANALYSIS: SHOULD BE 8 TOTAL (SO FAR) ######

# order of parameters to be updated:, tiers_lst, unpooled, atu_analysis_type
all_combos = [[["1"], False, "CC"],
              [["1"], True, "CC"],
              [["1"], False, "CC-ATU"],
              [["1"], True, "CC-ATU"],
              [["1", "2"], False, "CC"],
              [["1", "2"], True, "CC"],
              [["1", "2"], False, "CC-ATU"],
              [["1", "2"], True, "CC-ATU"]
            ]

# example set of kwargs
kwargs = yaml.safe_load(open("config.yaml"))

# config files run from 1 - len(all_combos)
for i in list(range(1, len(all_combos)+1)):
        
    # if the number is less than 10, add a 0 in front of it to keep them in order
    if i < 10:
        num_str = f"0{i}"
    else:
        num_str = str(i)
    
    with open(f"config_files/atu_{num_str}.yaml", "w+") as file:
        
        # constant for all cases
        kwargs["binary"] = True
        kwargs["atu_analysis"] = True
        kwargs["synonymous"] = False
        kwargs["amb_mode"] = "DROP"
        
        # not relevant, but set them all to WHO here
        kwargs["pheno_category_lst"] = "WHO"
        
        # update param combinations and write to the file
        param_dict = dict(zip(["tiers_lst", "unpooled", "atu_analysis_type"], all_combos[i-1]))
        kwargs.update(param_dict)
        yaml.dump(kwargs, file, default_flow_style=False, sort_keys=False)
    
    i += 1