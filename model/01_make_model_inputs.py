import numpy as np
import pandas as pd
import glob, os, yaml, argparse, sys, subprocess, shutil
import warnings
warnings.filterwarnings("ignore")
import tracemalloc
drug_gene_mapping = pd.read_csv("data/drug_gene_mapping.csv")
sys.path.append("utils")
from data_utils import *

drug_abbr_dict = {"Delamanid": "DLM",
                  "Bedaquiline": "BDQ",
                  "Clofazimine": "CFZ",
                  "Ethionamide": "ETO",
                  "Linezolid": "LZD",
                  "Moxifloxacin": "MXF",
                  "Capreomycin": "CAP",
                  "Amikacin": "AMK",
                  "Pretomanid": "PMD",
                  "Pyrazinamide": "PZA",
                  "Kanamycin": "KAN",
                  "Levofloxacin": "LFX",
                  "Streptomycin": "STM",
                  "Ethambutol": "EMB",
                  "Isoniazid": "INH",
                  "Rifampicin": "RIF"
                 }


######################### STEP 0: READ IN PARAMETERS FILE AND MAKE OUTPUT DIRECTORIES #########################


# starting the memory monitoring
tracemalloc.start()

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", dest='config_file', default='config.ini', type=str, required=True)
parser.add_argument('-d', "--drug", dest='drug', type=str, required=True)
parser.add_argument('--MIC-single-medium', dest='keep_single_medium', action='store_true', help='If specified, keep only the most common media for the MIC models')

cmd_line_args = parser.parse_args()
config_file = cmd_line_args.config_file
drug = cmd_line_args.drug
keep_single_medium = cmd_line_args.keep_single_medium

drug_WHO_abbr = drug_abbr_dict[drug]
kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
binary = kwargs["binary"]
atu_analysis = kwargs["atu_analysis"]
input_data_dir = kwargs["input_data_dir"]
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

amb_mode = kwargs["amb_mode"]
AF_thresh = kwargs["AF_thresh"]
silent = kwargs["silent"]
pool_type = kwargs["pool_type"]
assert pool_type in ['unpooled', 'poolLoF']

if amb_mode == "DROP":
    model_prefix = "dropAF"
elif amb_mode == "AF":
    model_prefix = "encodeAF"
elif amb_mode == "BINARY":
    model_prefix = "binarizeAF"
else:
    raise ValueError(f"{amb_mode} is not a valid mode for handling intermediate AFs")

if silent:
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

if not os.path.isdir(os.path.join(out_dir, "dropped_isolates")):
    os.makedirs(os.path.join(out_dir, "dropped_isolates"))

print(f"\nSaving model input matrix to {out_dir}")

# model matrix already exists
if os.path.isfile(os.path.join(out_dir, "model_matrix.pkl")):
    exit()

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


######################### STEP 1: GET ALL AVAILABLE PHENOTYPES, PROCESS THEM, AND SAVE TO A GENERAL PHENOTYPES FILE FOR EACH DRUG #########################        

    
if not os.path.isfile(phenos_file):

    print(f"Creating phenotypes dataframe: {phenos_file}")
    
    # read them all in, concatenate, and get the number of samples
    df_phenos = pd.concat([pd.read_csv(os.path.join(phenos_dir, fName)) for fName in os.listdir(phenos_dir) if "run" in fName], axis=0)
    
    # sometimes the data has exact duplicate rows -- same sample ID, medium, phenotype, category, etc.
    df_phenos = df_phenos.drop_duplicates(keep="first").reset_index(drop=True)
    
    if binary:
        if atu_analysis:
            df_phenos = df_phenos.loc[df_phenos["phenotypic_category"].str.contains("CC")]
        else:
            df_phenos = df_phenos.loc[~df_phenos["phenotypic_category"].str.contains("CC")]
        
        if len(df_phenos) == 0:
            print("There are no phenotypes for this analysis. Quitting this model\n")
            exit()
            
        print(f"    Phenotypic categories: {df_phenos.phenotypic_category.unique()}")

        # Drop samples with multiple recorded binary phenotypes
        drop_samples = df_phenos.groupby("sample_id").nunique().query(f"{pheno_col} > 1").reset_index()["sample_id"].values
         
        # the ATU dataframe has duplicates -- each sample has a phenotype for CC and one for CC-ATU
        if not atu_analysis:
            if len(drop_samples) > 0:
                print(f"    Dropped {len(drop_samples)}/{len(df_phenos['sample_id'].unique())} isolates with multiple recorded phenotypes")
                df_phenos = df_phenos.query("sample_id not in @drop_samples")
        else:
            # check that all samples are present twice in the ATU analysis dataframe
            assert sum(df_phenos.groupby(["sample_id"]).count()[pheno_col].unique() != np.array([2])) == 0
            
            if len(drop_samples) == 0:
                print("Phenotypes for all samples are the same for CC and CC-ATU designations. Quitting this model\n")
                exit()
            else:
                print(f"    {len(drop_samples)}/{len(df_phenos['sample_id'].unique())} isolates have different phenotypes using different CCs")

        assert sum(np.unique(df_phenos["phenotype"]) != np.array(['R', 'S'])) == 0
        df_phenos["phenotype"] = df_phenos["phenotype"].map({'S': 0, 'R': 1})
        
    else:
        # standardize media names so that later when prioritizing media and normalizing MICs, this is correct 
        df_phenos["medium"] = df_phenos["medium"].replace("Middlebrook7H10", "7H10")

        # separate the string value for the MIC into separate lower and upper bounds and compute the midpoint, which is stored in pheno_col ("mic_value")
        df_phenos = get_mic_midpoints(df_phenos, pheno_col)

        # Drop samples with different MICs recorded in the same media
        drop_samples = df_phenos.groupby(["sample_id", "medium"]).nunique().query(f"{pheno_col} > 1").reset_index()["sample_id"].values

        if len(drop_samples) > 0:
            print(f"    Dropped {len(drop_samples)}/{len(df_phenos['sample_id'].unique())} isolates with multiple recorded MICs in the same media")
            df_phenos = df_phenos.query("sample_id not in @drop_samples")
                    
    # column not needed, so remove to save space
    if "box" in df_phenos.columns:
        del df_phenos["box"]

    # check that there is resistance data for all samples
    assert sum(pd.isnull(df_phenos[pheno_col])) == 0
    
    # this is the phenotypes file for all models for the drug. 
    df_phenos.to_csv(phenos_file, index=False)

# phenotypes CSV files exist
else:
    df_phenos = pd.read_csv(phenos_file)

# normalize MICs to the most common medium so that they are on the same MIC scale
# need to do this step here and again in the next script because need to get the exact samples that are kept for the model so that model_matrix only contains the samples to fit the model on
if not binary:
    
    cc_df = pd.read_csv("data/drug_CC.csv")

    if keep_single_medium or drug == 'Pretomanid':

        # no normalized value for Pretomanid because there are no WHO-approved critical concentrations, so we just use the most common one
        # or only keep the most common medium
        df_phenos, most_common_medium = normalize_MICs_return_dataframe(drug, df_phenos, cc_df, keep_single_medium=keep_single_medium)

        # non-normalized column name
        mic_col = 'mic_value'

    else:
        # first apply the media hierarchy to decide which of the measured MICs to keep for each isolate (for isolates with multiple MICs measured in different media)
        df_phenos = process_multiple_MICs_different_media(df_phenos)
        
        # then, drop any media that can't be normalized and normalize to the scale of the most common medium
        df_phenos, most_common_medium = normalize_MICs_return_dataframe(drug, df_phenos, cc_df, keep_single_medium=keep_single_medium)

        # normalized column name
        mic_col = 'norm_MIC'
            
    print(f"    Min MIC: {np.min(df_phenos[mic_col].values)}, Max MIC: {np.max(df_phenos[mic_col].values)} in {most_common_medium}")

# get only isolates with the desired phenotypic category for the binary model
if binary and not atu_analysis:
    df_phenos = df_phenos.query("phenotypic_category in @pheno_category_lst")
    
# this only happens for Pretomanid because there are no WHO phenotypes
if len(df_phenos) == 0:
    print(f"There are no {' and '.join(pheno_category_lst)} phenotypes for this model\n")
    exit()

if "ALL" in pheno_category_lst and len(df_phenos.query("phenotypic_category=='ALL'")) == 0:
    print("There are no ALL phenotypes for this model\n")
    exit()

    
######################### STEP 2: GET ALL AVAILABLE GENOTYPES #########################
          
        
genos_dir = os.path.join(input_data_dir, "full_genotypes")
tier1_genos_file = os.path.join(analysis_dir, drug, "genos_1.csv.gz")
tier2_genos_file = os.path.join(analysis_dir, drug, "genos_2.csv.gz")

if not os.path.isfile(tier1_genos_file):
    print("Creating master genotype dataframes...")
    create_master_genos_files(drug, genos_dir, analysis_dir, include_tier2=False)
        
# this only happens for Pretomanid because there are no Tier 2 genes
if "2" in tiers_lst and not os.path.isfile(tier2_genos_file):
    print("There are no tier 2 genes. Quitting this model")
    exit()

# read in only the genotypes files for the tiers for this model
df_model = pd.concat([pd.read_csv(os.path.join(analysis_dir, drug, f"genos_{num}.csv.gz"), compression="gzip", low_memory=False) for num in tiers_lst])

# then keep only samples with phenotypes
df_model = df_model.loc[df_model["sample_id"].isin(df_phenos["sample_id"])]

print(f"    {len(df_model.sample_id.unique())}/{len(df_phenos.sample_id.unique())} isolates have both genotypes and phenotypes")   

# if silent variants are to be included, check that they are present and would make the model different from the corresponding noSyn model
if silent:
    if len(df_model.query("predicted_effect in ['synonymous_variant', 'stop_retained_variant', 'initiator_codon_variant'] & variant_binary_status==1")) == 0:
        print("There are no synonymous variants. Quitting this model\n")
        exit()
# if not, drop them from the dataframe
else:
    df_model = df_model.query("predicted_effect not in ['synonymous_variant', 'stop_retained_variant', 'initiator_codon_variant']").reset_index(drop=True)    


# 1 = alternative allele, 0 = reference allele, NaN = missing
df_model["mutation"] = df_model["resolved_symbol"] + "_" + df_model["variant_category"]

# drop any duplicates. Preferentially keep variant_binary_status = 1, so sort descending and keep first. Not very relevant though because the duplicates are just NaNs (missing values)
df_model = df_model.sort_values("variant_binary_status", ascending=False).drop_duplicates(["sample_id", "mutation"], keep="first").reset_index(drop=True)

    
######################### STEP 3: POOL LoF AND INFRAME MUTATIONS, IF INDICATED BY THE MODEL PARAMS #########################

        
# options for pool_type are unpooled and poolLoF
if pool_type == "poolLoF":

    # get the number of unique LoF and inframe mutations per gene (because pooling is done on a per-gene basis) 
    num_unique_LoF_by_gene = {}
    
    for gene in df_model.resolved_symbol.unique():
        
        num_unique_LoF_by_gene[gene] = df_model.query('resolved_symbol == @gene & predicted_effect in ["frameshift", "start_lost", "stop_gained", "feature_ablation"] & variant_allele_frequency > 0.75').variant_category.nunique()

    # if all counts are less than or equal to 1 -- means that there is 0-1 mutations of each type, so pooling will not make a difference. Need at least 2 mutations for a difference. 
    if np.max(list(num_unique_LoF_by_gene.values())) <= 1:
        print(f"Pooling LoF mutations separately does not affect this model. Quitting this model...\n")
        shutil.rmtree(out_dir) # delete the entire directory
        exit()
        
    print("Pooling LoF mutations into a single feature")
    df_model = pool_mutations(df_model, ["frameshift", "start_lost", "stop_gained", "feature_ablation"], "LoF")



######################### STEP 4: PROCESS AMBIGUOUS ALLELES -- I.E. THOSE WITH 0.25 <= AF <= 0.75 #########################


# # set variants with AF <= the threshold as wild-type and AF > the threshold as alternative
# if amb_mode == "BINARY":
#     print(f"Binarizing ambiguous variants with AF threshold of {AF_thresh}")
#     df_model.loc[(df_model["variant_allele_frequency"] <= AF_thresh), "variant_binary_status"] = 0
#     df_model.loc[(df_model["variant_allele_frequency"] > AF_thresh), "variant_binary_status"] = 1
    
# # use ambiguous AF as the matrix value for variants with AF > 0.25. AF = 0.25 is considered absent. Below 0.25, the AF measurements aren't reliable
# elif amb_mode == "AF":
#     print("Encoding ambiguous variants with their AF")
#     # encode all variants with AF > 0.25 with their AF. Variants with AF <= 0.25 already have variant_binary_status = 0
#     df_model.loc[df_model["variant_allele_frequency"] > 0.25, "variant_binary_status"] = df_model.loc[df_model["variant_allele_frequency"] > 0.25, "variant_allele_frequency"].values
   
# drop all isolates with ambiguous variants with ANY AF below the threshold. Then remove features that are no longer present
# elif amb_mode == "DROP":

# get all mutations that are present in the dataset somewhere.
# NOTE: doing this before dropping mutations with no signal does not affect the no signal category 
pre_dropAmb_mutations = df_model.query("variant_binary_status==1").reset_index(drop=True)["resolved_symbol"] + "_" + df_model.query("variant_binary_status==1").reset_index(drop=True)["variant_category"]

# AF <= 0.25 --> ABSENT, AF > 0.75 --> PRESENT
drop_isolates = df_model.query("variant_allele_frequency > 0.25 & variant_allele_frequency <= 0.75").sample_id.unique()
print(f"    Dropped {len(drop_isolates)}/{len(df_model.sample_id.unique())} isolates with with any AFs in the range (0.25, 0.75]. Remainder are binary")
df_model = df_model.query("sample_id not in @drop_isolates")    

# save the dropped isolate IDs for counting later
if len(drop_isolates) > 0:
    pd.Series(drop_isolates).to_csv(os.path.join(out_dir, "dropped_isolates/isolates_with_amb.txt"), sep="\t", header=None, index=False)    

# get the features in the dataframe after dropping isolates with ambiguous allele fractions, then save to a file if there are any dropped features
post_dropAmb_mutations = df_model.query("variant_binary_status==1").reset_index(drop=True)["resolved_symbol"] + "_" + df_model.query("variant_binary_status==1").reset_index(drop=True)["variant_category"]

# get the dropped features that are in pre_dropAmb_mutations but not in post_dropAmb_mutations, then write them to a file
dropped_feat = list(set(pre_dropAmb_mutations) - set(post_dropAmb_mutations))

if len(dropped_feat) > 0:
    print(f"    Dropped {len(dropped_feat)} features present only in samples with intermediate AFs")
    pd.Series(dropped_feat).to_csv(os.path.join(out_dir, "dropped_features/isolates_with_amb.txt"), sep="\t", header=None, index=False)

df_model.loc[(df_model["variant_allele_frequency"] <= 0.25), "variant_binary_status"] = 0
df_model.loc[(df_model["variant_allele_frequency"] > 0.75), "variant_binary_status"] = 1
    
# check after this step that the only NaNs left are truly missing data --> NaN in variant_binary_status must also be NaN in variant_allele_frequency
assert len(df_model.loc[(~pd.isnull(df_model["variant_allele_frequency"])) & (pd.isnull(df_model["variant_binary_status"]))]) == 0
assert len(df_model.loc[(pd.isnull(df_model["variant_allele_frequency"])) & (~pd.isnull(df_model["variant_binary_status"]))]) == 0

# save isolates with missing variants for counting later
isolates_with_missingness = df_model.loc[pd.isnull(df_model['variant_allele_frequency'])].sample_id.unique()

if len(isolates_with_missingness) > 0:
    # remove isolates with missingness from the matrix after pivoting to better keep track of the associated features that are dropped
    pd.Series(isolates_with_missingness).to_csv(os.path.join(out_dir, "dropped_isolates/isolates_dropped.txt"), sep="\t", header=None, index=False)
    

######################### STEP 5: PIVOT TO MATRIX AND DROP MISSINGNESS AND ANY FEATURES THAT ARE ALL 0 #########################


# pivot to matrix
matrix = df_model.pivot(index="sample_id", columns="mutation", values="variant_binary_status")
del df_model
print(f"Initially {matrix.shape[0]} isolates and {matrix.shape[1]} features")         

# remove features with no signal
matrix = remove_features_save_list(matrix, os.path.join(out_dir, "dropped_features/no_signal.txt"), dropNA=False)

# remove isolates with any missingness, then keep track of the dropped features
matrix = remove_features_save_list(matrix, os.path.join(out_dir, "dropped_features/isolates_dropped.txt"), dropNA=True)

# there should not be any more NaNs
assert sum(pd.isnull(np.unique(matrix.values))) == 0

# in this case, only 2 possible values -- 0 (ref), 1 (alt) because we already dropped NaNs
if amb_mode.upper() in ["BINARY", "DROP"]:
    assert len(np.unique(matrix.values)) <= 2
# the smallest value will be 0. Check that the second smallest value is greater than 0.25
else:
    assert np.sort(np.unique(matrix.values))[1] > 0.25

print(f"Final: {matrix.shape[0]} isolates and {matrix.shape[1]} variants")
matrix.to_pickle(os.path.join(out_dir, "model_matrix.pkl"))

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"    {script_memory} GB\n")