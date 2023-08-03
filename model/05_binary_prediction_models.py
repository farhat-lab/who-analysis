import numpy as np
import pandas as pd
import glob, os, yaml, sparse, sys
import scipy.stats as st
import sklearn.metrics
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
import tracemalloc, pickle, warnings
warnings.filterwarnings("ignore")

# utils files are in a separate folder
sys.path.append("utils")
from data_utils import *
from stats_utils import *


# starting the memory monitoring
tracemalloc.start()

_, config_file, drug = sys.argv

kwargs = yaml.safe_load(open(config_file))

analysis_dir = kwargs["output_dir"]
tiers_lst = kwargs["tiers_lst"]
pheno_category_lst = kwargs["pheno_category_lst"]
num_bootstrap = kwargs["num_bootstrap"]
amb_mode = kwargs["amb_mode"]

# make sure that both phenotypes are included
if "ALL" in pheno_category_lst:
    phenos_name = "ALL"
    pheno_category_lst = ["ALL", "WHO"]
else:
    phenos_name = "WHO"
    
out_dir = os.path.join(analysis_dir, drug, "BINARY", f"tiers={'+'.join(tiers_lst)}", f"phenos={phenos_name}")
print(f"Saving results to {out_dir}")
assert os.path.isdir(out_dir)
    
if drug == "Pretomanid":
    if phenos_name == "WHO":
        print("There are no WHO phenotypes for Pretomanid. Quitting this model...\n")
        exit()
    elif len(tiers_lst) == 2:
        print("There are no Tier 2 genes for Pretomanid. Quitting this model...\n")
        exit()
        
if amb_mode == "BINARY":
    model_prefix = "_binarized"
elif amb_mode == "AF":
    model_prefix = "_HET"
elif amb_mode == "DROP":
    model_prefix = ""
        
# if os.path.isfile(os.path.join(out_dir, f"model_stats_CV{model_prefix}.csv")):
#     print("Binary model was already fit. Quitting...\n")
#     exit()

results_excel = pd.read_excel(f"results/NEW/{drug}.xlsx", sheet_name=None)

if drug == "Pretomanid":
    # pick any other drug to get the keys for
    model_abbrev = pd.read_excel(f"results/NEW/Bedaquiline.xlsx", sheet_name=None).keys()
else:
    model_abbrev = results_excel.keys()

binary_analyses_lst = [
                        ########### Tier 1, WHO phenos ###########
                        "tiers=1/phenos=WHO/dropAF_noSyn_unpooled",
                        "tiers=1/phenos=WHO/dropAF_noSyn_poolSeparate",
                        "tiers=1/phenos=WHO/dropAF_withSyn_unpooled",
                        ########### Tiers 1 + 2, WHO phenos ###########
                        "tiers=1+2/phenos=WHO/dropAF_noSyn_unpooled",
                        "tiers=1+2/phenos=WHO/dropAF_noSyn_poolSeparate",
                        "tiers=1+2/phenos=WHO/dropAF_withSyn_unpooled",
                        ########### Tier 1, ALL phenos ###########
                        "tiers=1/phenos=ALL/dropAF_noSyn_unpooled",
                        "tiers=1/phenos=ALL/dropAF_noSyn_poolSeparate",
                        "tiers=1/phenos=ALL/dropAF_withSyn_unpooled",
                        ########### Tiers 1 + 2, ALL phenos ###########
                        "tiers=1+2/phenos=ALL/dropAF_noSyn_unpooled",
                        "tiers=1+2/phenos=ALL/dropAF_noSyn_poolSeparate",
                        "tiers=1+2/phenos=ALL/dropAF_withSyn_unpooled",
                      ]

# need to make this dictionary, then can remove the pooled models from the above list
analyses_keys_dict = dict(zip(binary_analyses_lst, model_abbrev))

# not going to include pooled variants in these models
binary_analyses_lst = [path for path in binary_analyses_lst if "_unpooled" in path]

# also keep only variants found to be relevant in the particular phenotypes group -- only for WHO phenotypes. For ALL phenotypes, include everything
binary_analyses_lst = [path for path in binary_analyses_lst if phenos_name in path]

if drug == "Pretomanid":
    binary_analyses_lst = [path for path in binary_analyses_lst if "WHO" not in path]
    
analyses_lst = []

for path in binary_analyses_lst:
    if len(tiers_lst) == 2:
        analyses_lst.append(path)
    else:
        if "1+2" not in path:
            analyses_lst.append(path)
    
# get the names of the Excel file sheets
keys_lst = [analyses_keys_dict[key] for key in analyses_lst]
print(keys_lst)

mutations_lst = set()

if drug in ["Levofloxacin", "Moxifloxacin"]:
    freq_thresh = 5
else:
    freq_thresh = 1
print(f"Including significant mutations present in at least {freq_thresh} isolates")

for key in keys_lst:
    
    # keep only significant mutations that are NOT neutral
    # mutations_lst = mutations_lst.union(results_excel[key].query("regression_confidence not in ['Uncertain', 'Neutral']")["mutation"].values)
    
    if "1+2" in key:
        thresh = 0.01
    else:
        thresh = 0.05
        
    # # add uncertain mutations that were significant in regression and frequent enough. Basically they didn't pass the LRT or MIC coef criteria
    # mutations_lst = mutations_lst.union(results_excel[key].query("regression_confidence=='Uncertain' & BH_pval < @thresh & Num_Isolates >= 5 & (PPV_LB >= 0.25 | NPV_LB >= 0.25)")["mutation"].values)
    
    mutations_lst = mutations_lst.union(results_excel[key].query("BH_pval < @thresh & Num_Isolates >= @freq_thresh")["mutation"].values)

mutations_lst = list(mutations_lst)
print(f"{len(mutations_lst)} mutations in the model")

if len(mutations_lst) == 0:
    print("There are no significant non-neutral, unpooled mutations. Quitting this model...\n")
    exit()
    
if len(tiers_lst) == 2:
    tier1_mutations = pd.read_csv(os.path.join(out_dir.replace("tiers=1+2", "tiers=1"), "mutations_lst.txt"), sep="\t", header=None)[0].values
    if len(set(tier1_mutations).symmetric_difference(mutations_lst)) == 0:
        print("There are no significant Tier 2 mutations. Quitting this model...\n")
        exit()
        
pd.Series(mutations_lst).to_csv(os.path.join(out_dir, "mutations_lst.txt"), sep="\t", index=False, header=None)
    
    
############# STEP 1: READ IN THE PREVIOUSLY GENERATED MATRICES #############
    
    
# read in only the genotypes files for the tiers for this model
df_phenos = pd.read_csv(os.path.join(analysis_dir, drug, "phenos_binary.csv")).query("phenotypic_category in @pheno_category_lst")

df_genos = pd.concat([pd.read_csv(os.path.join(analysis_dir, drug, f"genos_{num}.csv.gz"), compression="gzip") for num in tiers_lst], axis=0)
df_genos["mutation"] = df_genos["resolved_symbol"] + "_" + df_genos["variant_category"]
df_genos = df_genos.query("mutation in @mutations_lst & sample_id in @df_phenos.sample_id.values").drop_duplicates()
del df_genos["resolved_symbol"]
del df_genos["variant_category"]


# set variants with AF <= the threshold as wild-type and AF > the threshold as alternative
if amb_mode == "BINARY":
    print(f"Binarizing ambiguous variants with AF threshold of {AF_thresh}")
    df_genos.loc[(df_genos["variant_allele_frequency"] <= AF_thresh), "variant_binary_status"] = 0
    df_genos.loc[(df_genos["variant_allele_frequency"] > AF_thresh), "variant_binary_status"] = 1

# use ambiguous AF as the matrix value for variants with AF > 0.25. Below 0.25, the AF measurements aren't reliable
elif amb_mode == "AF":
    print("Encoding all variants with AF >= 0.25 with their AF")
    # encode all variants with AF > 0.25 with their AF
    df_genos.loc[df_genos["variant_allele_frequency"] >= 0.25, "variant_binary_status"] = df_genos.loc[df_genos["variant_allele_frequency"] >= 0.25, "variant_allele_frequency"].values

# drop all isolates with ambiguous variants with ANY AF below the threshold. DON'T DROP FEATURES BECAUSE MIGHT DROP SOMETHING RELEVANT
elif amb_mode == "DROP":    
    drop_isolates = df_genos.query("variant_allele_frequency > 0.25 & variant_allele_frequency < 0.75").sample_id.unique()
    print(f"    Dropped {len(drop_isolates)} isolates with any intermediate AFs. Remainder are binary")
    df_genos = df_genos.query("sample_id not in @drop_isolates")    
                
# check after this step that the only NaNs left are truly missing data --> NaN in variant_binary_status must also be NaN in variant_allele_frequency
assert len(df_genos.loc[(~pd.isnull(df_genos["variant_allele_frequency"])) & (pd.isnull(df_genos["variant_binary_status"]))]) == 0

# pivot and drop any rows with NaNs
model_matrix = df_genos.pivot(index="sample_id", columns="mutation", values="variant_binary_status").dropna(how="any", axis=0)
    
# there should not be any more NaNs
assert sum(pd.isnull(np.unique(model_matrix.values))) == 0

# remove any features that have no signal (should be nothing though because we kept only mutations that are significant)
model_matrix = model_matrix.loc[:, model_matrix.nunique() > 1]

# in this case, only 2 possible values -- 0 (ref), 1 (alt) because we already dropped NaNs
if amb_mode.upper() in ["BINARY", "DROP"]:
    assert len(np.unique(model_matrix.values)) <= 2
# the smallest value will be 0. Check that the second smallest value is greater than 0.25 (below this, AFs are not really reliable)
else:
    assert np.sort(np.unique(model_matrix.values))[1] >= 0.25
    print(np.sort(np.unique(model_matrix.values))[1], np.sort(np.unique(model_matrix.values))[-1])
    
# keep only samples (rows) that are in matrix and use loc with indices to ensure they are in the same order
df_phenos = df_phenos.set_index("sample_id")
df_phenos = df_phenos.loc[model_matrix.index]

# check that the sample ordering is the same in the genotype and phenotype matrices
assert sum(model_matrix.index != df_phenos.index) == 0

X = model_matrix.values
print(f"{X.shape[0]} isolates and {X.shape[1]} features in the model")
y = df_phenos["phenotype"].values


########################## STEP 2: FIT MODEL ##########################


def bootstrap_binary_metrics(X, y, num_bootstrap=None):
    
    model = LogisticRegressionCV(Cs=np.logspace(-6, 6, 13), 
                             cv=5,
                             penalty='l2',
                             max_iter=10000, 
                             multi_class='ovr',
                             scoring='neg_log_loss',
                             class_weight='balanced'
                            )

    model.fit(X, y)
    reg_param = model.C_[0]
    print(f"Regularization parameter: {reg_param}") 

    if drug in ["Clofazimine", "Delamanid", "Pretomanid"]:
        print("Setting specificity minimum at 0.9")
        spec_thresh = 0.9
    else:
        spec_thresh = None
        
    all_model_results = [pd.DataFrame(get_binary_metrics_from_model(model, X, y, maximize="sens_spec", spec_thresh=spec_thresh), index=[0])]
    print(all_model_results[0])
    
    if num_bootstrap is not None:
        
        print(f"Performing bootstrapping with {num_bootstrap} replicates")
        for i in range(num_bootstrap):

            bs_idx = np.random.choice(np.arange(len(y)), size=len(y), replace=True)

            X_bs = X[bs_idx, :]
            y_bs = y[bs_idx]

            model = LogisticRegression(C=reg_param, 
                                       penalty='l2',
                                       max_iter=10000, 
                                       multi_class='ovr',
                                       class_weight='balanced'
                                      )

            model.fit(X_bs, y_bs)        
            all_model_results.append(pd.DataFrame(get_binary_metrics_from_model(model, X_bs, y_bs, maximize="sens_spec", spec_thresh=spec_thresh), index=[0]))

            if i % int(num_bootstrap / 10) == 0:
                print(i)

    df_combined = pd.concat(all_model_results, axis=0).reset_index(drop=True)
    df_combined["CV"] = df_combined.index.values
    return df_combined


results_df = bootstrap_binary_metrics(X, y, num_bootstrap)
results_df.to_csv(os.path.join(out_dir, f"model_stats_CV{model_prefix}_bootstrap.csv"), index=False)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")