import numpy as np
import pandas as pd
import glob, os, yaml, sys, subprocess
import warnings
warnings.filterwarnings("ignore")
import tracemalloc
drug_gene_mapping = pd.read_csv("data/drug_gene_mapping.csv")


######################### STEP 0: READ IN PARAMETERS FILE AND MAKE OUTPUT DIRECTORIES #########################


# starting the memory monitoring
tracemalloc.start()

_, config_file, drug, drug_WHO_abbr = sys.argv

kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
binary = kwargs["binary"]
atu_analysis = kwargs["atu_analysis"]
input_data_dir = kwargs["input_dir"]
analysis_dir = kwargs["output_dir"]

# double check. If running CC vs. CC-ATU analysis, they are binary phenotypes
if atu_analysis:
    binary = True
    
pheno_category_lst = kwargs["pheno_category_lst"]
# make sure that both phenotypes are included
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
    pheno_category_lst = ["ALL", "WHO"]
else:
    phenos_name = "WHO"

missing_isolate_thresh = kwargs["missing_isolate_thresh"]
missing_feature_thresh = kwargs["missing_feature_thresh"]
amb_mode = kwargs["amb_mode"]
AF_thresh = kwargs["AF_thresh"]
impute = kwargs["impute"]
synonymous = kwargs["synonymous"]
pool_type = kwargs["pool_type"]

if amb_mode == "DROP":
    model_prefix = "dropAF"
elif amb_mode == "AF":
    model_prefix = "encodeAF"
elif amb_mode == "BINARY":
    model_prefix = "binarizeAF"
else:
    raise ValueError(f"{amb_mode} is not a valid mode for handling intermediate AFs")

if synonymous:
    model_prefix += "_withSyn"
else:
    model_prefix += "_noSyn"
    
model_prefix += f"_{pool_type}"
    
# add to config file for use in the second and third scripts
kwargs["model_prefix"] = model_prefix

with open(config_file, "w") as file:
    yaml.dump(kwargs, file, default_flow_style=False, sort_keys=False)
  
if binary:
    if atu_analysis:
        out_dir = os.path.join(analysis_dir, drug, "ATU", f"tiers={'+'.join(tiers_lst)}", model_prefix)
    else:
        out_dir = os.path.join(analysis_dir, drug, "BINARY", f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}", model_prefix)
else:
    out_dir = os.path.join(analysis_dir, drug, "MIC", f"tiers={'+'.join(tiers_lst)}", model_prefix)

# create all directories down to dropped_features, which will contain text files of the features dropped during data processing
if not os.path.isdir(os.path.join(out_dir, "dropped_features")):
    os.makedirs(os.path.join(out_dir, "dropped_features"))
    
print(f"\nSaving model results to {out_dir}")            

if binary:
    phenos_dir = os.path.join(input_data_dir, "phenotypes", f"drug_name={drug}")
    pheno_col = "phenotype"
    if atu_analysis:
        phenos_file = os.path.join(analysis_dir, drug, "phenos_atu.csv")
    else:
        phenos_file = os.path.join(analysis_dir, drug, "phenos_binary.csv")
else:
    phenos_dir = os.path.join(input_data_dir, "mic", f"drug_name={drug}")
    phenos_file = os.path.join(analysis_dir, drug, "phenos_mic.csv")
    pheno_col = "mic_value"

    
# # this is mainly for the CC vs. CC-ATU analysis, which use the same genotype dataframes. Only the phenotypes are different
# if os.path.isfile(os.path.join(out_dir, "model_matrix.pkl")):
#     print("Model matrix already exists. Proceeding with modeling")
#     exit()

                
# def remove_features_save_list(matrix, fName, dropNA=False):
#     '''
#     This function saves the names of features that were dropped during a given processing step.
#     An output file is only written if features were dropped (i.e., no empty files).
#     '''
    
#     # original numbers of features and samples
#     matrix_features = matrix.columns
#     num_samples = len(matrix)

#     # drop any isolates with missingness
#     if dropNA:
#         matrix = matrix.dropna(axis=0, how="any")

#     # remove any features that have no signal after isolates were dropped, then save the list of dropped features
#     matrix = matrix[matrix.columns[~((matrix == 0).all())]]

#     # get the dropped features = features in 1 that are not in 2
#     dropped_feat = list(set(matrix_features) - set(matrix.columns))
#     num_dropped_samples = num_samples - len(matrix)
    
#     if len(dropped_feat) > 0:
#         with open(fName, "w+") as file:
#             for feature in dropped_feat:
#                 file.write(feature + "\n")
                
#     # if no samples with NaNs were dropped, then the number of samples should not have changed
#     if dropNA:
#         print(f"    Dropped {num_dropped_samples}/{num_samples} samples with any missingness and {len(dropped_feat)} associated features")
#     else:
#         print(f"    Dropped {len(dropped_feat)} features that are not present in any sample")
#         assert num_dropped_samples == 0
    
#     return matrix
    

######################### STEP 1: GET ALL AVAILABLE PHENOTYPES, PROCESS THEM, AND SAVE TO A GENERAL PHENOTYPES FILE FOR EACH DRUG #########################


def get_mic_midpoints(mic_df, pheno_col):
    '''
    This function processes the MIC data from string ranges to float midpoints.  
    '''
    mic_sep = mic_df[pheno_col].str.split(",", expand=True)
    mic_sep.columns = ["MIC_lower", "MIC_upper"]

    mic_sep["Lower_bracket"] = mic_sep["MIC_lower"].str[0] #.map(bracket_mapping)
    mic_sep["Upper_bracket"] = mic_sep["MIC_upper"].str[-1] #.map(bracket_mapping)

    mic_sep["MIC_lower"] = mic_sep["MIC_lower"].str[1:]
    mic_sep["MIC_upper"] = mic_sep["MIC_upper"].str[:-1]
    mic_sep = mic_sep.replace("", np.nan)

    mic_sep[["MIC_lower", "MIC_upper"]] = mic_sep[["MIC_lower", "MIC_upper"]].astype(float)
    mic_sep = pd.concat([mic_df[["sample_id", "medium"]], mic_sep], axis=1)

    # exclude isolates with unknown lower concentrations, indicated by square bracket in the lower bound
    mic_sep = mic_sep.query("Lower_bracket != '['")
    
    # some mislabeling, where the upper bracket is a parentheses. Can't be possible because the upper bound had to have been tested
    mic_sep.loc[(mic_sep["MIC_lower"] == 0), "Upper_bracket"] = "]"
    
    # upper bracket parentheses should be [max_MIC, NaN), so check this
    assert len(mic_sep.loc[(mic_sep["Upper_bracket"] == ")") &
                           (~pd.isnull(mic_sep["MIC_upper"]))
                          ]) == 0
    
    # if the upper bound is NaN, then the MIC (midpoint) should be the lower bound, which is the maximum concentration tested
    mic_sep.loc[pd.isnull(mic_sep["MIC_upper"]), pheno_col] = mic_sep.loc[pd.isnull(mic_sep["MIC_upper"])]["MIC_lower"]
    
    # otherwise, take the average
    mic_sep.loc[~pd.isnull(mic_sep["MIC_upper"]), pheno_col] = np.mean([mic_sep.loc[~pd.isnull(mic_sep["MIC_upper"])]["MIC_lower"], mic_sep.loc[~pd.isnull(mic_sep["MIC_upper"])]["MIC_upper"]], axis=0)
    
    # check that there are no NaNs in the MIC column
    assert sum(mic_sep[pheno_col].isna()) == 0
    return mic_sep.drop_duplicates()

    
if not os.path.isfile(phenos_file):
        
    # read them all in, concatenate, and get the number of samples
    df_phenos = pd.concat([pd.read_csv(os.path.join(phenos_dir, fName)) for fName in os.listdir(phenos_dir) if "run" in fName], axis=0)
    
    # sometimes the data has duplicates
    df_phenos = df_phenos.drop_duplicates(keep="first").reset_index(drop=True)
    
    if binary:
        if atu_analysis:
            df_phenos = df_phenos.loc[df_phenos["phenotypic_category"].str.contains("CC")]
        else:
            df_phenos = df_phenos.loc[~df_phenos["phenotypic_category"].str.contains("CC")]
        
        print(f"Phenotypic categories: {df_phenos.phenotypic_category.unique()}")
        if len(df_phenos) == 0:
            print("There are no phenotypes for this analysis. Quitting this model")
            exit()
    else:
        df_phenos["medium"] = df_phenos["medium"].replace("Middlebrook7H10", "7H10")

    # Drop samples with multiple recorded phenotypes
    drop_samples = df_phenos.groupby(["sample_id"]).nunique().query(f"{pheno_col} > 1").index.values
     
    # the ATU dataframe has duplicates -- each sample has a phenotype for CC and one for CC-ATU
    if not atu_analysis:
        if len(drop_samples) > 0:
            print(f"    Dropping {len(drop_samples)} of {len(df_phenos['sample_id'].unique())} isolates with multiple recorded phenotypes")
            df_phenos = df_phenos.query("sample_id not in @drop_samples")
    else:
        if len(drop_samples) == 0:
            print("Phenotypes for all samples are the same for CC and CC-ATU designations. Quitting this model")
            exit()
        else:
            print(f"    {len(drop_samples)} of {len(df_phenos['sample_id'].unique())} isolates have different phenotypes using different CCs")
            
        # check that all samples are present twice in the ATU analysis dataframe
        assert sum(df_phenos.groupby(["sample_id"]).count()[pheno_col].unique() != np.array([2])) == 0

    # check that there is resistance data for all samples
    assert sum(pd.isnull(df_phenos[pheno_col])) == 0
    
    # additional checks
    if binary:
        assert sum(np.unique(df_phenos["phenotype"]) != np.array(['R', 'S'])) == 0
        df_phenos["phenotype"] = df_phenos["phenotype"].map({'S': 0, 'R': 1})
    else:
        df_phenos = get_mic_midpoints(df_phenos, pheno_col)
        print(f"Min MIC: {np.min(df_phenos[pheno_col].values)}, Max MIC: {np.max(df_phenos[pheno_col].values)}")
        
    # column not needed, so remove to save space
    if "box" in df_phenos.columns:
        del df_phenos["box"]
    
    # this is the phenotypes file for all models for the drug. 
    df_phenos.to_csv(phenos_file, index=False)
else:
    df_phenos = pd.read_csv(phenos_file)


# get only isolates with the desired phenotypic category for the binary model
if binary and not atu_analysis:
    df_phenos = df_phenos.query("phenotypic_category in @pheno_category_lst")
    
# this only happens for Pretomanid because there are no WHO phenotypes
if len(df_phenos) == 0:
    print(f"There are no {' and '.join(pheno_category_lst)} phenotypes for this model")
    exit()
    

######################### STEP 2: GET ALL AVAILABLE GENOTYPES #########################
          
        
genos_dir = os.path.join(input_data_dir, "full_genotypes")
tier1_genos_file = os.path.join(analysis_dir, drug, "genos_1.csv.gz")
tier2_genos_file = os.path.join(analysis_dir, drug, "genos_2.csv.gz")


def create_master_genos_files(drug):

    tier_paths = glob.glob(os.path.join(genos_dir, f"drug_name={drug}", "*"))

    for dir_name in tier_paths:
        
        if dir_name[-1] in ["1", "2"]:
            
            tier = dir_name[-1]
            num_files = len(os.listdir(dir_name))

            # concatenate all files in the directory and save to a gzipped csv file with the tier number as the suffix
            # 5th column is the neutral column, but it's all NaN, so remove to save space
            command = f"awk '(NR == 1) || (FNR > 1)' {dir_name}/* | cut --complement -d ',' -f 5 | gzip > {analysis_dir}/{drug}/genos_{tier}.csv.gz"
            subprocess.run(command, shell=True)
            
            print(f"Created {analysis_dir}/{drug}/genos_{tier}.csv.gz from {num_files} files")

if not os.path.isfile(tier1_genos_file):
    print("Creating master genotype dataframes...")
    create_master_genos_files(drug)
        
# this only happens for Pretomanid because there are no Tier 2 genes
if "2" in tiers_lst and not os.path.isfile(tier2_genos_file):
    print("There are no tier 2 genes. Quitting this model")
    exit()

# read in only the genotypes files for the tiers for this model
df_model = pd.concat([pd.read_csv(os.path.join(analysis_dir, drug, f"genos_{num}.csv.gz"), compression="gzip", low_memory=False) for num in tiers_lst])

# then keep only samples of the desired phenotype
df_model = df_model.loc[df_model["sample_id"].isin(df_phenos["sample_id"])]

# drop synonymous variants, unless the boolean is True
if not synonymous:
    df_model = df_model.query("predicted_effect not in ['synonymous_variant', 'stop_retained_variant', 'initiator_codon_variant']").reset_index(drop=True)

    
######################### STEP 3: POOL LOF MUTATIONS, IF INDICATED BY THE MODEL PARAMS #########################


def pool_mutations(df, effect_lst, pool_col):
    
    df.loc[df["predicted_effect"].isin(effect_lst), ["variant_category", "position"]] = [pool_col, np.nan]

    # sort descending to keep the largest variant_binary_status and variant_allele_frequency first. In this way, pooled mutations that are actually present are preserved
    df_pooled = df.query("variant_category == @pool_col").sort_values(by=["variant_binary_status", "variant_allele_frequency"], ascending=False, na_position="last").drop_duplicates(subset=["sample_id", "resolved_symbol"], keep="first")

    # combine with the unpooled variants and the other variants and return
    return pd.concat([df_pooled, df.query("variant_category != @pool_col")], axis=0)


# options for pool_type are unpooled, poolSeparate, and poolALL
if pool_type == "poolSeparate":
    print("Pooling LOF and inframe mutations separately")
    df_model = pool_mutations(df_model, ["frameshift", "start_lost", "stop_gained", "feature_ablation"], "lof")
    df_model = pool_mutations(df_model, ["inframe_insertion", "inframe_deletion"], "inframe")

elif pool_type == "poolALL":
    print("Pooling LOF and inframe mutations together into a single feature")
    df_model = pool_mutations(df_model, ["frameshift", "start_lost", "stop_gained", "feature_ablation", "inframe_insertion", "inframe_deletion"], "lof_all")


######################### STEP 4: PROCESS AMBIGUOUS ALLELES -- I.E. THOSE WITH 0.25 <= AF <= 0.75 #########################


# set variants with AF <= the threshold as wild-type and AF > the threshold as alternative
if amb_mode == "BINARY":
    print(f"Binarizing ambiguous variants with AF threshold of {AF_thresh}")
    df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] <= AF_thresh), "variant_binary_status"] = 0
    df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] > AF_thresh), "variant_binary_status"] = 1

# use ambiguous AF as the matrix value for variants with AF > 0.25. Below 0.25, the AF measurements aren't reliable
elif amb_mode == "AF":
    print("Encoding ambiguous variants with their AF")
    # encode all variants with AF > 0.25 with their AF
    df_model.loc[df_model["variant_allele_frequency"] > 0.25, "variant_binary_status"] = df_model.loc[df_model["variant_allele_frequency"] > 0.25, "variant_allele_frequency"].values
   
# drop all isolates with ambiguous variants with ANY AF below the threshold. DON'T DROP FEATURES BECAUSE MIGHT DROP SOMETHING RELEVANT
elif amb_mode == "DROP":
    
    pre_dropAmb_mutations = df_model.query("variant_binary_status==1")["resolved_symbol"] + "_" + df_model.query("variant_binary_status==1")["variant_category"]
    
    drop_isolates = df_model.query("variant_allele_frequency > 0.25 & variant_allele_frequency < 0.75").sample_id.unique()
    print(f"    Dropped {len(drop_isolates)} isolates with any intermediate AFs. Remainder are binary")
    df_model = df_model.query("sample_id not in @drop_isolates")    
    
    # get the features in the dataframe after dropping isolates with ambiguous allele fractions, then save to a file if there are any dropped features
    post_dropAmb_mutations = df_model.query("variant_binary_status==1")["resolved_symbol"] + "_" + df_model.query("variant_binary_status==1")["variant_category"]
    
    # get the dropped features that are in pre_dropAmb_mutations but not in post_dropAmb_mutations, then write them to a file
    dropped_feat = list(set(pre_dropAmb_mutations) - set(post_dropAmb_mutations))
    if len(dropped_feat) > 0:
        with open(os.path.join(out_dir, "dropped_features/isolates_with_amb.txt"), "w+") as file:
            for feature in dropped_feat:
                file.write(feature + "\n")
    
# check after this step that the only NaNs left are truly missing data --> NaN in variant_binary_status must also be NaN in variant_allele_frequency
assert len(df_model.loc[(~pd.isnull(df_model["variant_allele_frequency"])) & (pd.isnull(df_model["variant_binary_status"]))]) == 0


######################### STEP 5: PIVOT TO MATRIX AND DROP MISSINGNESS AND ANY FEATURES THAT ARE ALL 0 #########################


# 1 = alternative allele, 0 = reference allele, NaN = missing
df_model["mutation"] = df_model["resolved_symbol"] + "_" + df_model["variant_category"]

# drop any duplicates. Preferentially keep variant_binary_status = 1, so sort descending and keep first
df_model = df_model.sort_values("variant_binary_status", ascending=False).drop_duplicates(["sample_id", "mutation"], keep="first")
matrix = df_model.pivot(index="sample_id", columns="mutation", values="variant_binary_status")
del df_model
print(f"    Initially {matrix.shape[0]} samples and {matrix.shape[1]} features")         

def remove_features_save_list(matrix, fName, dropNA=False):
    
    if dropNA:
        init_samples = matrix.index.values
        matrix = matrix.dropna(axis=0)
        next_samples = matrix.index.values

    # drop features with no signal (0 everywhere)
    #drop_features = matrix.columns[((matrix == 0).all())]
    #matrix = matrix[matrix.columns[~((matrix == 0).all())]]
    
    drop_features = matrix.loc[:, matrix.nunique() == 1]
    matrix = matrix.loc[:, matrix.nunique() > 1]

    if len(drop_features) > 0:
        with open(fName, "w+") as file:
            for feature in drop_features:
                file.write(feature + "\n")
                
    if dropNA:
        print(f"Dropped {len(set(init_samples)-set(next_samples))} isolates due to missigness and {len(drop_features)} associated features")
    else:
        print(f"Dropped {len(drop_features)} features with no signal")
                
    return matrix


# remove features with no signal
matrix = remove_features_save_list(matrix, os.path.join(out_dir, "dropped_features/no_signal.txt"), dropNA=False)

# remove isolates with any missingness, then keep track of the dropped features
matrix = remove_features_save_list(matrix, os.path.join(out_dir, "dropped_features/isolates_dropped.txt"), dropNA=True)

# there should not be any more NaNs
assert sum(pd.isnull(np.unique(matrix.values))) == 0

# in this case, only 2 possible values -- 0 (ref), 1 (alt) because we already dropped NaNs
if amb_mode.upper() in ["BINARY", "DROP"]:
    assert len(np.unique(matrix.values)) <= 2
# the smallest value will be 0. Check that the second smallest value is greater than 0.25 (below this, AFs are not really reliable)
else:
    assert np.sort(np.unique(matrix.values))[1] > 0.25

print(f"    Kept {matrix.shape[0]} isolates and {matrix.shape[1]} genetic variants")
matrix.to_pickle(os.path.join(out_dir, "model_matrix.pkl"))

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB")