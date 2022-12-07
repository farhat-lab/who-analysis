



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
    
    
    
###### CONFIG FILES FOR THE MIC ANALYSIS: SHOULD BE 8 TOTAL (SO FAR) ######

# not relevant 
phenos = ["WHO"]
tiers = [["1"], ["1", "2"]]
unpooled = [False, True]
syn = [False, True]
amb_mode = ["DROP", "AF"]

all_combos = list(itertools.product(*[phenos, tiers, unpooled, syn, amb_mode]))
print(len(all_combos))

# example set of kwargs
kwargs = yaml.safe_load(open("config.yaml"))

# config files run from 1 - len(all_combos)
for i in list(range(len(1, all_combos+1))):
        
    # if the number is less than 10, add a 0 in front of it to keep them in order
    if i < 10:
        num_str = f"0{i}"
    else:
        num_str = str(i)
    
    with open(f"config_files/binary_{num_str}.yaml", "r+") as file:
        
        kwargs["binary"] = True
        kwargs["atu_analysis"] = False
        yaml.dump(kwargs, file, default_flow_style=False, sort_keys=False)
    
    i += 1