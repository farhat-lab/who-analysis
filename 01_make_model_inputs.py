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
include_synonymous = kwargs["include_synonymous"]
pheno_category_lst = kwargs["pheno_category_lst"]
impute = kwargs["impute"]

print("Tiers for this model:", tiers_lst)

if not os.path.isdir(os.path.join(out_dir, drug)):
    print(f"Creating output directory {os.path.join(out_dir, drug)}")
    os.mkdir(os.path.join(out_dir, drug))
    
if not os.path.isdir(os.path.join(out_dir, drug, model_prefix)):
    print(f"Creating output directory {os.path.join(out_dir, drug, model_prefix)}")
    os.mkdir(os.path.join(out_dir, drug, model_prefix))

genos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/full_genotypes'
phenos_dir = '/n/data1/hms/dbmi/farhat/ye12/who/phenotypes'
phenos_dir = os.path.join(phenos_dir, f"drug_name={drug}")


############# STEP 1: GET ALL AVAILABLE PHENOTYPES #############


dfs_list_phenos = []

for fName in os.listdir(phenos_dir):
    dfs_list_phenos.append(pd.read_csv(os.path.join(phenos_dir, fName)))

df_phenos = pd.concat(dfs_list_phenos)
df_phenos = df_phenos.query("category in @pheno_category_lst")

# check that there are no duplicated phenotypes
assert len(df_phenos) == len(df_phenos.sample_id.unique())

# check that there is resistance data for all samples
assert sum(pd.isnull(df_phenos.phenotype)) == 0
    
df_phenos["phenotype"] = df_phenos["phenotype"].map({'S': 0, 'R': 1})

print(f"{len(df_phenos)} samples with phenotypes for {drug}")


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
    print(fName)

    # read in the dataframe
    df = pd.read_csv(fName)

    # get only genotypes for samples that have a phenotype
    df_avail_isolates = df.loc[df.sample_id.isin(df_phenos.sample_id)]
    
    # keep all variants
    if include_synonymous:
        dfs_lst.append(df_avail_isolates)
    else:
        # keep only 1) noncoding variants and 2) non-synonymous variants in coding regions. 
        # P = coding variants, C = synonymous or upstream variants (get only upstream variants by getting only negative positions), and N = non-coding variants on rrs/rrl
        dfs_lst.append(df_avail_isolates.loc[((df_avail_isolates.category.astype(str).str[0] == 'c') & (df_avail_isolates.category.str.contains('-'))) | 
                                             (df_avail_isolates.category.astype(str).str[0].isin(['n', 'p']))])
        
df_model = pd.concat(dfs_lst)


############# STEP 3: PROCESS HETEROZYGOUS ALLELES -- I.E. THOSE WITH 0.25 <= AF <= 0.75 #############


# set variants with AF <= the threshold as wild-type and AF > the threshold as alternative
if het_mode == "BINARY":
    print(f"Binarizing heterozygous variants with AF threshold of {af_thresh}")
    df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] <= af_thresh), "variant_binary_status"] = 0
    df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] > af_thresh), "variant_binary_status"] = 1

# use heterozygous AF as the matrix value
elif het_mode == "AF":
    print("Encoding heterozygous variants with AF")
    df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] <= af_thresh), "variant_binary_status"] = df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] <= af_thresh)]["variant_allele_frequency"].values

# drop all isolates with heterozygous variants with ANY AF below the threshold. DON'T DROP FEATURES BECAUSE MIGHT DROP SOMETHING RELEVANT
elif het_mode == "DROP":
    drop_isolates = df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (df_model["variant_allele_frequency"] <= af_thresh)].sample_id.unique()
    print(f"Dropped {len(drop_isolates)} isolates with any variant with an AF >= 0.25 and <= {af_thresh}")
    df_model = df_model.query("sample_id not in @drop_isolates")
else:
    raise ValueError(f"{het_mode} is not a valid mode for handling heterozygous alleles")
    
# check that the only NaNs in the variant binary status column are also NaN in the variant_allele_frequency column (truly missing data) 
#assert len(df_model.loc[(pd.isnull(df_model["variant_binary_status"])) & (~pd.isnull(df_model["variant_allele_frequency"]))]) == 0 


############# STEP 4: PIVOT TO MATRIX AND DROP BAD ISOLATES / FEATURES #############


# 1 = alternative allele, 0 = reference allele, NaN = missing
df_model["mutation"] = df_model["gene_symbol"] + "_" + df_model["category"]
matrix = df_model.pivot(index="sample_id", columns="mutation", values="variant_binary_status")

# in this case, only 3 possible values -- 0 (ref), 1 (alt), NaN (missing)
if het_mode.upper() in ["BINARY", "DROP"]:
    assert len(np.unique(matrix.values)) <= 3

# drop isolates (rows) with too much missingness. Not going to impute (if needed) the bad isolates
filtered_matrix = matrix.dropna(axis=0, thresh=(1-missing_thresh)*matrix.shape[1])


############# STEP 5: IMPUTE OR DROP REMAINING NANS IN THE GENOTYPES #############


if impute:
    
    # use only samples that are in the filtered matrix for imputation
    df_phenos = df_phenos.query("sample_id in @filtered_matrix.index.values")
    
    impute_cols = filtered_matrix.columns[filtered_matrix.isna().any()]
    print(f"There are NaNs in {len(impute_cols)}/{filtered_matrix.shape[1]} genetic features")

    if len(impute_cols) > 0:
        print("Imputing missing data...")
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
       
# there should not be any more NaNs
assert sum(pd.isnull(np.unique(filtered_matrix.values))) == 0

# this comparison is only for the missing isolates. Treating heterozygous alleles is done above, and that number is not considered here. 
print(f"Dropped {matrix.shape[0] - filtered_matrix.shape[0]} isolates with a lot of missingness")
filtered_matrix.to_pickle(os.path.join(out_dir, drug, model_prefix, "filt_matrix.pkl"))

# keep only samples with genotypes available because everything should be represented, including samples without variants
df_phenos = df_phenos.query("sample_id in @filtered_matrix.index.values")
df_phenos.to_csv(os.path.join(out_dir, drug, model_prefix, "phenos.csv"), index=False)