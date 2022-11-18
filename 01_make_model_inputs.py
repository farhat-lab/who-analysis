import numpy as np
import pandas as pd
import glob, os, yaml, sys
import warnings
warnings.filterwarnings("ignore")
import tracemalloc
analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'


############# STEP 0: READ IN PARAMETERS FILE AND MAKE OUTPUT DIRECTORIES #############


# starting the memory monitoring
tracemalloc.start()

_, config_file, drug, drug_WHO_abbr = sys.argv

kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
binary = kwargs["binary"]
pheno_category_lst = kwargs["pheno_category_lst"]
# mics_category = kwargs["mics_category"]

missing_isolate_thresh = kwargs["missing_isolate_thresh"]
missing_feature_thresh = kwargs["missing_feature_thresh"]
amb_mode = kwargs["amb_mode"]
AF_thresh = kwargs["AF_thresh"]
impute = kwargs["impute"]
synonymous = kwargs["synonymous"]
unpooled = kwargs["unpooled"]

if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
    # make sure that both phenotypes are included in this case
    pheno_category_lst = ["ALL", "WHO"]
else:
    phenos_name = "WHO"


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
    
if unpooled:
    model_prefix += "_unpooled" 
    
# add to config file for use in the second and third scripts
kwargs["model_prefix"] = model_prefix

with open(config_file, "w") as file:
    yaml.dump(kwargs, file, default_flow_style=False, sort_keys=False)
  
out_dir = os.path.join(analysis_dir, drug, f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}", model_prefix)

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
print(f"\nSaving model results to {out_dir}")            

if binary:
    phenos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/phenotypes'
    phenos_file = os.path.join(analysis_dir, drug, "phenos_binary.csv")
    pheno_col = "phenotype"
else:
    phenos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/mic'
    phenos_file = os.path.join(analysis_dir, drug, "phenos_mic.csv")
    pheno_col = "mic_value"
    
phenos_dir = os.path.join(phenos_dir, f"drug_name={drug}")
genos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/full_genotypes'


############# STEP 1: GET ALL AVAILABLE PHENOTYPES, PROCESS THEM, AND SAVE TO A GENERAL PHENOTYPES FILE FOR EACH MODEL TYPE #############


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
    
    return mic_sep

    
if not os.path.isfile(phenos_file):
    
    # read them all in, concatenate, and get the number of samples
    df_phenos = pd.concat([pd.read_csv(os.path.join(phenos_dir, fName)) for fName in os.listdir(phenos_dir) if "run" in fName], axis=0)
    
    # when phenotypic_category == CC or CC-ATU, the phenotype is for a binarized MIC, so exclude those from the binary model. 
    if binary:
        df_phenos = df_phenos.loc[~df_phenos["phenotypic_category"].str.contains("CC")]
        print(f"Phenotypic categoryies: {df_phenos.phenotypic_category.unique()}")
    
    # Drop samples with multiple recorded phenotypes
    drop_samples = df_phenos.groupby(["sample_id"]).nunique().query(f"{pheno_col} > 1").index.values
        
    if len(drop_samples) > 0:
        print(f"    Dropping {len(drop_samples)} isolates with multiple recorded phenotypes")
        df_phenos = df_phenos.query("sample_id not in @drop_samples")
    
    # then drop any duplicated rows. There can be duplicates just because of a minor bug, so protect against that. 
    df_phenos = df_phenos.drop_duplicates(keep="first").reset_index(drop=True)

    # check that there is resistance data for all samples
    assert sum(pd.isnull(df_phenos[pheno_col])) == 0
    if binary:
        assert sum(np.unique(df_phenos["phenotype"]) != np.array(['R', 'S'])) == 0
        df_phenos["phenotype"] = df_phenos["phenotype"].map({'S': 0, 'R': 1})
    else:
        df_phenos = get_mic_midpoints(mic_df, pheno_col)
        
    # this is the phenotypes file for all models for the drug. 
    df_phenos.to_csv(phenos_file, index=False)
else:
    df_phenos = pd.read_csv(phenos_file)


# get only isolates with the desired phenotypic category for the binary model
if binary:
    df_phenos = df_phenos.query("phenotypic_category in @pheno_category_lst")
# else:
#     df_phenos = df_phenos.query("")


############# STEP 2: GET ALL AVAILABLE GENOTYPES #############
          
        
def read_in_all_genos(drug):
            
    # first get all the genotype files associated with the drug
    geno_files = []

    for subdir in os.listdir(os.path.join(genos_dir, f"drug_name={drug}")):

        # subdirectory (tiers)
        full_subdir = os.path.join(genos_dir, f"drug_name={drug}", subdir)

        # the last character is the tier number. Get variants from both tiers
        if full_subdir[-1] in ["1", "2"]:
            for fName in os.listdir(full_subdir):
                if "run" in fName:
                    geno_files.append(os.path.join(full_subdir, fName))

    print(f"    {len(geno_files)} files with genotypes")

    dfs_lst = []
    for i, fName in enumerate(geno_files):

        # print(f"Reading in genotypes dataframe {i+1}/{len(geno_files)}")
        df = pd.read_csv(fName, low_memory=False)
        dfs_lst.append(df)

    # fail-safe if there are duplicate rows
    return pd.concat(dfs_lst).drop_duplicates().reset_index(drop=True)


if not os.path.isfile(os.path.join(analysis_dir, drug, "genos.csv.gz")):
    # read in all available genotypes and save to a compressed file for each drug
    df_model = read_in_all_genos(drug)

    if len(df_model.loc[~pd.isnull(df_model["neutral"])]) == 0:
        del df_model["neutral"]
    df_model.to_csv(os.path.join(analysis_dir, drug, "genos.csv.gz"), compression="gzip", index=False)
else:
    df_model = pd.read_csv(os.path.join(analysis_dir, drug, "genos.csv.gz"), compression="gzip")

# keep only samples that are in the phenotypes dataframe for this model
df_model = df_model.loc[df_model["sample_id"].isin(df_phenos["sample_id"])]
    
if not synonymous:
    df_model = df_model.query("predicted_effect not in ['synonymous_variant', 'stop_retained_variant', 'initiator_codon_variant']").reset_index(drop=True)

############# STEP 3: POOL LOF MUTATIONS, IF INDICATED BY THE MODEL PARAMS #############


def pool_mutations(df, effect_lst, pool_col):
    
    df.loc[df["predicted_effect"].isin(effect_lst), ["variant_category", "position"]] = [pool_col, np.nan]

    # sort descending to keep the largest variant_binary_status and variant_allele_frequency first. In this way, pooled mutations that are actually present are preserved
    df_pooled = df.query("variant_category == @pool_col").sort_values(by=["variant_binary_status", "variant_allele_frequency"], ascending=False, na_position="last").drop_duplicates(subset=["sample_id", "resolved_symbol"], keep="first")

    # combine with the unpooled variants and the other variants and return
    return pd.concat([df_pooled, df.query("variant_category != @pool_col")], axis=0)


if not unpooled:
    print("Pooling LOF and inframe mutations ")
    df_model = pool_mutations(df_model, ["frameshift", "start_lost", "stop_gained", "feature_ablation"], "lof")
    df_model = pool_mutations(df_model, ["inframe_insertion", "inframe_deletion"], "inframe")


############# STEP 4: PROCESS AMBIGUOUS ALLELES -- I.E. THOSE WITH 0.25 <= AF <= 0.75 #############


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
    drop_isolates = df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] <= 0.75)].sample_id.unique()
    print(f"Dropped {len(drop_isolates)} isolates with any intermediate AFs. Remainder are binary")
    df_model = df_model.query("sample_id not in @drop_isolates")    
    
# check after this step that the only NaNs left are truly missing data --> NaN in variant_binary_status must also be NaN in variant_allele_frequency
assert len(df_model.loc[(~pd.isnull(df_model["variant_allele_frequency"])) & (pd.isnull(df_model["variant_binary_status"]))]) == 0


############# STEP 5: PIVOT TO MATRIX AND DROP HIGH-FREQUENCY MISSINGNESS #############


# 1 = alternative allele, 0 = reference allele, NaN = missing
df_model["mutation"] = df_model["resolved_symbol"] + "_" + df_model["variant_category"]

# drop more duplicates, but I think this might be because we have multiple data pulls at a time
# NaN is larger than any number, so sort ascending and keep first
df_model = df_model.sort_values("variant_binary_status").drop_duplicates(["sample_id", "mutation"], keep="first")

matrix = df_model.pivot(index="sample_id", columns="mutation", values="variant_binary_status")

# in this case, only 3 possible values -- 0 (ref), 1 (alt), NaN (missing)
if amb_mode.upper() in ["BINARY", "DROP"]:
    assert len(np.unique(matrix.values)) <= 3
# the smallest value will be 0. Check that the second smallest value is greater than 0.25 (below this, AFs are not really reliable)
else:
    assert np.sort(np.unique(matrix.values))[1] > 0.25

# compare proportions of missing isolates or variants to determine which is more problematic and drop that first. usually, isolates will be more problematic    
# maximum proportion of missing isolates per feature
max_prop_missing_isolates_per_feature = matrix.isna().sum(axis=0).max() / matrix.shape[0]
# maximum proportion of missing features per isolate
max_prop_missing_variants_per_isolate = matrix.isna().sum(axis=1).max() / matrix.shape[1]

# drop isolates first if there are isolates with a lot of missingness (many variants missing)
if max_prop_missing_variants_per_isolate >= max_prop_missing_isolates_per_feature:
    print(f"    Up to {round(max_prop_missing_variants_per_isolate*100, 2)}% of variants per isolate and {round(max_prop_missing_isolates_per_feature*100, 2)}% of isolates per variant have missing data")
    
    # drop isolates (rows) with missingness above the threshold (default = 1%)
    filtered_matrix = matrix.dropna(axis=0, thresh=(1-missing_isolate_thresh)*matrix.shape[1])
    
    print(f"    Dropped {matrix.shape[0] - filtered_matrix.shape[0]}/{matrix.shape[0]} isolates with >{int(missing_isolate_thresh*100)}% missingness")
    num_isolates_after_isolate_thresh = filtered_matrix.shape[0]
    num_features_after_isolate_thresh = filtered_matrix.shape[1]
    
    # drop features (columns) with missingness above the threshold (default = 1%)
    filtered_matrix = filtered_matrix.dropna(axis=1, thresh=(1-missing_feature_thresh)*filtered_matrix.shape[0])
    print(f"    Dropped {num_features_after_isolate_thresh - filtered_matrix.shape[1]}/{num_features_after_isolate_thresh} variants with >{int(missing_feature_thresh*100)}% missingness")
    num_isolates_after_thresholding = filtered_matrix.shape[0]
    
# drop variants first if there are variants with a lot of missingness (many isolates missing)
else:
    print(f"    Up to {round(max_prop_missing_isolates_per_feature*100, 2)}% of isolates per variant and {round(max_prop_missing_variants_per_isolate*100, 2)}% of variants per isolate have missing data")
    
    # drop features (columns) with missingness above the threshold (default = 1%)
    filtered_matrix = matrix.dropna(axis=1, thresh=(1-missing_feature_thresh)*matrix.shape[0])
    
    print(f"    Dropped {matrix.shape[1] - filtered_matrix.shape[1]}/{matrix.shape[1]} variants with >{int(missing_feature_thresh*100)}% missingness")
    num_isolates_after_variant_thresh = filtered_matrix.shape[0]
    num_variants_after_variant_thresh = filtered_matrix.shape[1]

    # drop isolates (rows) with missingness above the threshold (default = 1%)
    filtered_matrix = filtered_matrix.dropna(axis=0, thresh=(1-missing_isolate_thresh)*filtered_matrix.shape[1])
    print(f"    Dropped {num_isolates_after_variant_thresh - filtered_matrix.shape[0]}/{num_isolates_after_variant_thresh} isolates with >{int(missing_isolate_thresh*100)}% missingness")
    num_isolates_after_thresholding = filtered_matrix.shape[0]


############# STEP 6: IMPUTE OR DROP REMAINING NANS IN THE GENOTYPES -- THESE ARE LOW FREQUENCY MISSING DATA #############


# can only impute as this is written for binary phenotypes. Not going to impute anyway, so I'm not going to adapt this for MIC samples
if impute and binary:
    # use only samples that are in the filtered matrix for imputation
    df_phenos = df_phenos.query("sample_id in @filtered_matrix.index.values")
    
    impute_cols = filtered_matrix.columns[filtered_matrix.isna().any()]
    print(f"    There are NaNs in {len(impute_cols)}/{filtered_matrix.shape[1]} genetic features")

    if len(impute_cols) > 0:
        print("Imputing missing data")
        for i, col in enumerate(impute_cols):

            # isolates without a genotype for this column
            missing_isolates = filtered_matrix.loc[pd.isnull(filtered_matrix[col]), :].index.values

            susc_isolates = df_phenos.query("phenotype == 0").sample_id.values
            resist_isolates = df_phenos.query("phenotype == 1").sample_id.values

            # take the average of the genotype across isolates of the same class
            susc_impute = filtered_matrix.loc[filtered_matrix.index.isin(susc_isolates), col].dropna().mean()
            resist_impute = filtered_matrix.loc[filtered_matrix.index.isin(resist_isolates), col].dropna().mean()

            for isolate in missing_isolates:

                if df_phenos.query("sample_id == @isolate").phenotype.values[0] == 1:
                    filtered_matrix.loc[isolate, col] = resist_impute
                else:
                    filtered_matrix.loc[isolate, col] = susc_impute

# don't impute anything, simply drop all isolates (rows) with NaNs.
else:
    filtered_matrix.dropna(axis=0, inplace=True, how="any")
    print(f"    Dropped {num_isolates_after_thresholding - filtered_matrix.shape[0]} isolates with any remaining missingness")
       
        
# there should not be any more NaNs
assert sum(pd.isnull(np.unique(filtered_matrix.values))) == 0
print(f"    Kept {filtered_matrix.shape[0]} isolates and {filtered_matrix.shape[1]} genetic variants")
filtered_matrix.to_pickle(os.path.join(out_dir, "filt_matrix.pkl"))

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()

# write peak memory usage in GB
with open("memory_usage.log", "a+") as file:
    file.write(f"\n{out_dir}\n")
    file.write(f"{os.path.basename(__file__)}: {script_memory} GB\n")