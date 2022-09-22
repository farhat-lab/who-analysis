import numpy as np
import pandas as pd
import glob, os, yaml, sys
import warnings
warnings.filterwarnings("ignore")


############# STEP 0: READ IN PARAMETERS FILE AND MAKE OUTPUT DIRECTORIES #############

_, config_file = sys.argv

kwargs = yaml.safe_load(open(config_file))

tiers_lst = kwargs["tiers_lst"]
drug = kwargs["drug"]
out_dir = kwargs["out_dir"]
model_prefix = kwargs["model_prefix"]
missing_thresh = kwargs["missing_thresh"]
het_mode = kwargs["het_mode"]
af_thresh = kwargs["AF_thresh"]
synonymous = kwargs["synonymous"]
pheno_category_lst = kwargs["pheno_category_lst"]
pool_lof = kwargs["pool_lof"]
impute = kwargs["impute"]


if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

if not os.path.isdir(os.path.join(out_dir, drug)):
    os.mkdir(os.path.join(out_dir, drug))
    
if not os.path.isdir(os.path.join(out_dir, drug, model_prefix)):
    os.mkdir(os.path.join(out_dir, drug, model_prefix))

genos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/full_genotypes'
phenos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/phenotypes'
phenos_dir = os.path.join(phenos_dir, f"drug_name={drug}")


############# STEP 1: GET ALL AVAILABLE PHENOTYPES #############


dfs_list_phenos = []

for fName in os.listdir(phenos_dir):
    dfs_list_phenos.append(pd.read_csv(os.path.join(phenos_dir, fName)))

df_phenos = pd.concat(dfs_list_phenos)
df_phenos = df_phenos.query("phenotype_category in @pheno_category_lst")

# check that there are no duplicated phenotypes
assert len(df_phenos) == len(df_phenos.sample_id.unique())

# check that there is resistance data for all samples
assert sum(pd.isnull(df_phenos.phenotype)) == 0
    
df_phenos["phenotype"] = df_phenos["phenotype"].map({'S': 0, 'R': 1})


############# STEP 2: GET ALL AVAILABLE GENOTYPES #############


# first get all the genotype files associated with the drug
geno_files = []

for subdir in os.listdir(os.path.join(genos_dir, f"drug_name={drug}")):
    
    # subdirectory (tiers)
    full_subdir = os.path.join(genos_dir, f"drug_name={drug}", subdir)

    # the last character is the tier number
    if subdir[-1] in tiers_lst:
        
        for fName in os.listdir(full_subdir):
            
            # some hidden files (i.e. Git files) are present, so ignore them
            if fName[0] != ".":
                geno_files.append(os.path.join(full_subdir, fName))
          
        
dfs_lst = []
for i, fName in enumerate(geno_files):
        
    print(f"Working on dataframe {i+1}/{len(geno_files)}")
    #print("   ", fName)

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
        dfs_lst.append(df_avail_isolates.query("Effect not in ['synonymous_variant', 'stop_retained_variant']"))
        

df_model = pd.concat(dfs_lst)


############# STEP 3: POOL LOF MUTATIONS, IF INDICATED BY THE MODEL PARAMS #############


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
    frameshift = df.loc[df["variant_category"].str.contains("fs")]
    
    # (sample, gene) pairs with a single frameshift mutation are LOF
    lof_single_fs = pd.DataFrame(frameshift.groupby(["sample_id", "resolved_symbol"])["variant_category"].count()).query("variant_category == 1").reset_index()

    # already 1 because variant_category is the counts column now
    lof_single_fs.rename(columns={"variant_category": "lof"}, inplace=True)

    # lof column now is 1 for (sample, gene) pairs with only 1 frameshift mutation and 0 for those with multiple frameshift mutations
    frameshift = frameshift.merge(lof_single_fs, on=["sample_id", "resolved_symbol"], how="outer")
    frameshift["lof"] = frameshift["lof"].fillna(0)
    
    # merge with original dataframe to get the rest of the columns back
    df_with_lof = df.merge(frameshift[["sample_id", "resolved_symbol", "variant_category", "lof"]], on=["sample_id", "resolved_symbol", "variant_category"], how="outer")
    assert len(df) == len(df_with_lof)
    del df
    
    # value_counts drops all the NaNs when computing
    assert df_with_lof["lof"].value_counts().sum() == len(frameshift)
    
    ###### STEP 2: Assign loss of start, early stop codon, and large-scale deletion to LOF ######
    
    # criteria for lof are: nonsense mutation, loss of start or stop, single frameshift mutation. Get only those satisfying the first two criteria (last done above)
    # "?" = start lost, * = nonsense, deletion is a large-scale deletion
    lof_criteria = ["nonsense", "\?", "\*", "deletion", "stop_lost"]
    df_with_lof.loc[df_with_lof["variant_category"].str.contains("|".join(lof_criteria)), "lof"] = 1
    
    # get only variants that are LOF
    df_lof = df_with_lof.query("lof == 1")
    
    ###### STEP 3: COMBINE LOF VARIANTS WITH NON-LOF VARIANTS TO GET A FULL DATAFRAME ######
    
    # this dataframe will be slightly smaller than the original because some lof mutations have been pooled
    
    # just keep 1 instance because the feature will become just lof. The row that is kept is arbitrary
    # groupby takes more steps because the rest of the columns need to be gotten again
    df_lof_pooled = df_lof.drop_duplicates(["sample_id", "resolved_symbol"], keep='first')
    
    # concatenate the dataframe without LOF variants with the dataframe of pooled LOF variants
    df_final = pd.concat([df_with_lof.query("lof != 1"), df_lof_pooled], axis=0)

    # there could be some variants with lof = 0 and Effect = lof because these are frameshift mutations that co-occur with other frameshifts
    # but there will not be any mutations with null in the lof column and whose Effect is lof
    assert len(df_final.loc[(pd.isnull(df_final["lof"])) & (df_final["Effect"] == 'lof')]) == 0
    
    # the lof column will now be the variant category to use, so 
    # 1. replace non-lof frame-shift mutations (value = 0) with NaN 
    # 2. replace lof variants (value = 1) with the string lof
    # 3. fill the NaNs (non-lof) with the original variant_category column
    # 4. rename columns
    df_final["lof"] = df_final["lof"].replace(0, np.nan)
    df_final["lof"] = df_final["lof"].replace(1, "lof")
    df_final["lof"] = df_final["lof"].fillna(df_final["variant_category"])
    return df_final.rename(columns={"variant_category": "variant_category_unpooled", "lof": "variant_category"})


if pool_lof:
    print("Pooling LOF mutations")
    df_model = pool_lof_mutations(df_model)


############# STEP 4: PROCESS HETEROZYGOUS ALLELES -- I.E. THOSE WITH 0.25 <= AF <= 0.75 #############


# set variants with AF <= the threshold as wild-type and AF > the threshold as alternative
if het_mode == "BINARY":
    print(f"    Binarizing heterozygous variants with AF threshold of {af_thresh}")
    df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] <= af_thresh), "variant_binary_status"] = 0
    df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] > af_thresh), "variant_binary_status"] = 1

# use heterozygous AF as the matrix value
elif het_mode == "AF":
    print("    Encoding heterozygous variants with AF")
    df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] <= af_thresh), "variant_binary_status"] = df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] <= af_thresh)]["variant_allele_frequency"].values

# drop all isolates with heterozygous variants with ANY AF below the threshold. DON'T DROP FEATURES BECAUSE MIGHT DROP SOMETHING RELEVANT
elif het_mode == "DROP":
    drop_isolates = df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] <= af_thresh)].sample_id.unique()
    print(f"    Dropped {len(drop_isolates)} isolates with any variant with an AF >= 0.25 and <= {af_thresh}")
    df_model = df_model.query("sample_id not in @drop_isolates")
else:
    raise ValueError(f"{het_mode} is not a valid mode for handling heterozygous alleles")
    
# check that the only NaNs in the variant binary status column are also NaN in the variant_allele_frequency column (truly missing data) 
#assert len(df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (~pd.isnull(df_model["variant_allele_frequency"]))]) == 0 


############# STEP 5: PIVOT TO MATRIX AND DROP ISOLATES WITH A LOT OF MISSINGNESS #############


# 1 = alternative allele, 0 = reference allele, NaN = missing
df_model["mutation"] = df_model["resolved_symbol"] + "_" + df_model["variant_category"]
matrix = df_model.pivot(index="sample_id", columns="mutation", values="variant_binary_status")

# in this case, only 3 possible values -- 0 (ref), 1 (alt), NaN (missing)
if het_mode.upper() in ["BINARY", "DROP"]:
    assert len(np.unique(matrix.values)) <= 3

# drop isolates (rows) with missingness above the threshold (default = 5%)
filtered_matrix = matrix.dropna(axis=0, thresh=(1-missing_thresh)*matrix.shape[1])
print(f"    Dropped {matrix.shape[0] - filtered_matrix.shape[0]} isolates with >{int(missing_thresh*100)}% missingness")
num_isolates_after_missing_thresh = filtered_matrix.shape[0]

############# PRINT THE NUMBER OF FEATURES WITH MISSING DATA #############
#print(len())

############# STEP 6: IMPUTE OR DROP REMAINING NANS IN THE GENOTYPES #############


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
    print(f"    Dropped {num_isolates_after_missing_thresh - filtered_matrix.shape[0]} isolates with any remaining missingness")
       
# there should not be any more NaNs
assert sum(pd.isnull(np.unique(filtered_matrix.values))) == 0

# this comparison is only for the missing isolates. Treating heterozygous alleles is done above, and that number is not considered here. 
print(f"    Kept {filtered_matrix.shape[0]} isolates and {filtered_matrix.shape[1]} features for the model")
filtered_matrix.to_pickle(os.path.join(out_dir, drug, model_prefix, "filt_matrix.pkl"))

# keep only samples with genotypes available because everything should be represented, including samples without variants
df_phenos = df_phenos.query("sample_id in @filtered_matrix.index.values")
df_phenos.to_csv(os.path.join(out_dir, drug, model_prefix, "phenos.csv"), index=False)