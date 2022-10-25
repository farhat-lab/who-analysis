import numpy as np
import pandas as pd
import glob, os, yaml, sys
import warnings
warnings.filterwarnings("ignore")
from memory_profiler import profile

# open file for writing memory logs to. Overwrite file, otherwise it will become huge
mem_log=open('memory_usage.log','w+')

############# STEP 0: READ IN PARAMETERS FILE AND MAKE OUTPUT DIRECTORIES #############

_, config_file, drug, drug_WHO_abbr = sys.argv

kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
pheno_category_lst = kwargs["pheno_category_lst"]

missing_isolate_thresh = kwargs["missing_isolate_thresh"]
missing_feature_thresh = kwargs["missing_feature_thresh"]
het_mode = kwargs["het_mode"]
AF_thresh = kwargs["AF_thresh"]
impute = kwargs["impute"]
synonymous = kwargs["synonymous"]
pool_lof = kwargs["pool_lof"]

out_dir = '/n/data1/hms/dbmi/farhat/ye12/who/analysis'
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
    # make sure that both phenotypes are included in this case (in case user forgets to include WHO in the list)
    pheno_category_lst = ["ALL", "WHO"]
else:
    phenos_name = "WHO"


if het_mode == "DROP":
    model_prefix = "dropAF"
elif het_mode == "AF":
    model_prefix = "encodeAF"
elif het_mode == "BINARY":
    model_prefix = "binarizeAF"
else:
    raise ValueError(f"{het_mode} is not a valid mode for handling heterozygous alleles")

if synonymous:
    model_prefix += "_withSyn"
else:
    model_prefix += "_noSyn"
    
    
if pool_lof:
    model_prefix += "_poolLOF"
    
    
# add to config file for use in the second and third scripts
kwargs["model_prefix"] = model_prefix

with open(config_file, "w") as file:
    yaml.dump(kwargs, file, default_flow_style=False, sort_keys=False)
    
out_dir = os.path.join(out_dir, drug, f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}", model_prefix)

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
print(f"\nSaving results to {out_dir}")
    
genos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/full_genotypes'
phenos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/phenotypes'
phenos_dir = os.path.join(phenos_dir, f"drug_name={drug}")

############# STEP 1: GET ALL AVAILABLE PHENOTYPES #############


# read them all in, concatenate, and get the number of samples
df_phenos = pd.concat([pd.read_csv(os.path.join(phenos_dir, fName)) for fName in os.listdir(phenos_dir) if "run" in fName], axis=0)

if len(df_phenos.phenotypic_category.unique()) > 2:
    raise ValueError("More than 2 phenotypic categories in the dataframe!")

df_phenos = df_phenos.query("phenotypic_category in @pheno_category_lst")

# get the number of samples that have more than 1 phenotype recorded. Drop such samples (should be 0), but leave as a QC step for now. 
drop_samples = df_phenos.groupby(["sample_id"]).nunique().query("phenotype > 1").index.values
if len(drop_samples) > 0:
    print(f"    {len(drop_samples)} isolates are recorded as being both resistant and susceptible to {drug}")
    df_phenos = df_phenos.query("sample_id not in @drop_samples")
    
# then drop any duplicated phenotypes
df_phenos = df_phenos.drop_duplicates(keep="first").reset_index(drop=True)

# check that there is resistance data for all samples
assert sum(pd.isnull(df_phenos.phenotype)) == 0
assert sum(np.unique(df_phenos["phenotype"]) != np.array(['R', 'S'])) == 0
    
df_phenos["phenotype"] = df_phenos["phenotype"].map({'S': 0, 'R': 1})


############# STEP 2: GET ALL AVAILABLE GENOTYPES #############
          
        
@profile(stream=mem_log)
def read_in_data():
        
    # first get all the genotype files associated with the drug
    geno_files = []

    for subdir in os.listdir(os.path.join(genos_dir, f"drug_name={drug}")):

        # subdirectory (tiers)
        full_subdir = os.path.join(genos_dir, f"drug_name={drug}", subdir)

        # the last character is the tier number
        if full_subdir[-1] in tiers_lst:
            for fName in os.listdir(full_subdir):
                if "run" in fName:
                    geno_files.append(os.path.join(full_subdir, fName))

    print(f"    {len(df_phenos)} samples with phenotypes and {len(geno_files)} files with genotypes.")

    dfs_lst = []
    for i, fName in enumerate(geno_files):

        # print(f"Reading in genotypes dataframe {i+1}/{len(geno_files)}")
        # read in the dataframe
        df = pd.read_csv(fName)

        # get only genotypes for samples that have a phenotype
        df_avail_isolates = df.loc[df.sample_id.isin(df_phenos.sample_id)]

        # keep all variants
        if synonymous:
            dfs_lst.append(df_avail_isolates)
        else:
            # P = coding variants, C = synonymous or upstream variants (get only upstream variants by getting only negative positions), and N = non-coding variants on rrs/rrl
            # deletion does not contain the p/c/n prefix
            # synonymous variants = synonymous, change in start codon that produces V instead of M, and changes in stop codon that preserve stop
            dfs_lst.append(df_avail_isolates.query("predicted_effect not in ['synonymous_variant', 'stop_retained_variant', 'initiator_codon_variant']"))        


    # possible to have duplicated entries because they have different predicted effects
    # example: Met1fs is present in two lines because it has 2 predicted effects: frameshift and start lost
    # sort the dataframe by inverse, which keeps start_lost before frameshift, then drop_duplicates. 
    df_model = pd.concat(dfs_lst)
    return df_model.sort_values("predicted_effect", ascending=False).drop_duplicates(subset=["sample_id", "resolved_symbol", "variant_category", "variant_binary_status", "variant_allele_frequency"], keep="first").reset_index(drop=True)


df_model = read_in_data()


############# STEP 3: POOL LOF MUTATIONS, IF INDICATED BY THE MODEL PARAMS #############

@profile(stream=mem_log)
def pool_lof_mutations(df):
    '''
    resolved_symbol = gene
    
    Effect = lof for ALL frameshift, nonsense, loss of start, and large-scale deletion mutations. 
    
    This function creates a new column called lof, which is 1 for variants that are lof, 0 for frameshift mutations that are not lof, and nan for variants that
    couldn't be lof (synonymous, missense, etc.)
    
    LOF criteria = loss of start or stop codon, nonsense mutation, single frameshift mutation, large-scale deletion
    
    If one of the above criteria (except the frameshift mutation) co-occurs with multiple frameshift mutations in the same sample and gene, then an lof feature will be
    generated, and the frameshift mutations will remain as additional features. i.e. the LOF will not trump the multiple frameshift mutations. 
    '''
    
    ###### STEP 1: Assign all (sample, gene) pairs with a single frameshift mutation to LOF, and the remaining to not LOF ######
    
    # get all frameshift mutations and separate by the number of frameshifts per gene per sample
    frameshift = df.query("predicted_effect == 'frameshift'")

    # (sample, gene) pairs with a single frameshift mutation are LOF
    lof_single_fs = pd.DataFrame(frameshift.groupby(["sample_id", "resolved_symbol"])["predicted_effect"].count()).query("predicted_effect == 1").reset_index()

    # already 1 because variant_category is the counts column now
    lof_single_fs.rename(columns={"predicted_effect": "lof"}, inplace=True)

    # lof column now is 1 for (sample, gene) pairs with only 1 frameshift mutation and 0 for those with multiple frameshift mutations
    frameshift = frameshift.merge(lof_single_fs, on=["sample_id", "resolved_symbol"], how="outer")
    frameshift["lof"] = frameshift["lof"].fillna(0)

    # merge with original dataframe to get the rest of the columns back. predicted_effect is now lof
    df_with_lof = df.merge(frameshift[["sample_id", "resolved_symbol", "variant_category", "lof"]], on=["sample_id", "resolved_symbol", "variant_category"], how="outer")
    assert len(df) == len(df_with_lof)

    # value_counts drops all the NaNs when computing
    assert df_with_lof["lof"].value_counts(dropna=True).sum() == len(frameshift)

    ###### STEP 2: Assign loss of start, stop gained, and large-scale deletion to LOF ######

    # criteria for lof are: nonsense mutation, loss of start, single frameshift mutation. Get only those satisfying the first two criteria (last done above)
    df_with_lof.loc[(df_with_lof["variant_category"] == 'deletion') | 
                    (df_with_lof["predicted_effect"].isin(['stop_gained', 'start_lost'])), 'lof'
                   ] = 1
    
    # get only variants that are LOF
    df_lof = df_with_lof.query("lof == 1")
    
    ###### STEP 3: COMBINE LOF VARIANTS WITH NON-LOF VARIANTS TO GET A FULL DATAFRAME ######
    
    # this dataframe will be slightly smaller than the original because some lof mutations have been pooled
    
    # just keep 1 instance because the feature will become just lof. The row that is kept is arbitrary
    # groupby takes more steps because the rest of the columns need to be gotten again
    df_lof_pooled = df_lof.drop_duplicates(["sample_id", "resolved_symbol"], keep='first')
    
    # concatenate the dataframe without LOF variants with the dataframe of pooled LOF variants
    df_final = pd.concat([df_with_lof.query("lof != 1"), df_lof_pooled], axis=0)
    
    # the lof column will now be the variant category to use, so 
    # 1. replace non-lof frame-shift mutations (value = 0) with NaN 
    # 2. replace lof variants (value = 1) with the string lof
    # 3. fill the NaNs (non-lof) with the original variant_category column
    # 4. rename columns
    df_final["lof"] = df_final["lof"].replace(0, np.nan)
    df_final["lof"] = df_final["lof"].replace(1, "lof")
    df_final["lof"] = df_final["lof"].fillna(df_final["variant_category"])
    
    assert len(df_final["lof"].unique()) <= len(df_final["variant_category"].unique())
    return df_final.rename(columns={"variant_category": "variant_category_unpooled", "lof": "variant_category"})


if pool_lof:
    print("Pooling LOF mutations")
    df_model = pool_lof_mutations(df_model)
    

############# STEP 4: PROCESS HETEROZYGOUS ALLELES -- I.E. THOSE WITH 0.25 <= AF <= 0.75 #############


# set variants with AF <= the threshold as wild-type and AF > the threshold as alternative
if het_mode == "BINARY":
    print(f"Binarizing heterozygous variants with AF threshold of {AF_thresh}")
    df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] <= AF_thresh), "variant_binary_status"] = 0
    df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] > AF_thresh), "variant_binary_status"] = 1

# use heterozygous AF as the matrix value for variants with AF > 0.25. Below 0.25, the AF measurements aren't reliable
elif het_mode == "AF":
    print("Encoding heterozygous variants with AF")
    
    # encode only heterozygous variants with AF
    # df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (~pd.isnull(df_model["variant_allele_frequency"])), "variant_binary_status"] = df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (~pd.isnull(df_model["variant_allele_frequency"])), "variant_allele_frequency"].values
    
    # encode all variants with AF > 0.25 with their AF
    df_model.loc[df_model["variant_allele_frequency"] > 0.25, "variant_binary_status"] = df_model.loc[df_model["variant_allele_frequency"] > 0.25, "variant_allele_frequency"].values
   
# drop all isolates with heterozygous variants with ANY AF below the threshold. DON'T DROP FEATURES BECAUSE MIGHT DROP SOMETHING RELEVANT
elif het_mode == "DROP":
    drop_isolates = df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] <= 0.75)].sample_id.unique()
    print(f"Dropped {len(drop_isolates)} isolates with any heterozygous variants. Remainder are binary")
    df_model = df_model.query("sample_id not in @drop_isolates")
else:
    raise ValueError(f"{het_mode} is not a valid mode for handling heterozygous alleles")
    
    
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
if het_mode.upper() in ["BINARY", "DROP"]:
    assert len(np.unique(matrix.values)) <= 3
# the smallest value will be 0. Check that the second smallest value is greater than 0.25 (below this, AFs are not really reliable)
else:
    assert np.sort(np.unique(matrix.values))[1] > 0.25

# compare proportions of missing isolates or variants to determine which is more problematic and drop that first
# usually, isolates will be more problematic, but rrs/rrl are very problematic, so for ribosome-targeting drugs, need to drop variants first
    
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


if impute:
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

            if i % 100 == 0:
                print(i)
# don't impute anything, simply drop all isolates (rows) with NaNs.
else:
    filtered_matrix.dropna(axis=0, inplace=True, how="any")
    print(f"    Dropped {num_isolates_after_thresholding - filtered_matrix.shape[0]} isolates with any remaining missingness")
       
        
# there should not be any more NaNs
assert sum(pd.isnull(np.unique(filtered_matrix.values))) == 0
print(f"    Kept {filtered_matrix.shape[0]} isolates and {filtered_matrix.shape[1]} genetic variants")

# check if the filtered matrix has the same dimensions as the corresponding model without LOF pooling
# if they have the same shape, it means that LOF pooling did not make 
if pool_lof:
    df_model_no_poolLOF = pd.read_pickle(os.path.join(out_dir.replace("_poolLOF", ""), "filt_matrix.pkl"))
    if (len(df_model_no_poolLOF.index.symmetric_difference(filtered_matrix.index)) == 0) & (df_model_no_poolLOF.shape[1] == filtered_matrix.shape[1]):
        print("Pooling LOFs does not affect this model. Quitting this model...")
        os.rmdir(os.path.join(out_dir))
        exit()
        
filtered_matrix.to_pickle(os.path.join(out_dir, "filt_matrix.pkl"))

# keep only samples with genotypes available because everything should be represented, including samples without variants
df_phenos = df_phenos.query("sample_id in @filtered_matrix.index.values")
df_phenos.to_csv(os.path.join(out_dir, "phenos.csv"), index=False)