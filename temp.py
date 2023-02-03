import pandas as pd
import os
analysis_dir = '/n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue'


drug = "Rifampicin"

df_genos = pd.concat([pd.read_csv(os.path.join(analysis_dir, drug, "genos_1.csv.gz"), compression="gzip", usecols=["sample_id", "resolved_symbol", "variant_category", "variant_allele_frequency", "variant_binary_status"]), 
                      pd.read_csv(os.path.join(analysis_dir, drug, "genos_2.csv.gz"), compression="gzip", usecols=["sample_id", "resolved_symbol", "variant_category", "variant_allele_frequency", "variant_binary_status"])
                     ], axis=0)

df_genos.to_csv(os.path.join(analysis_dir, drug, "all_genos.csv.gz"), compression="gzip", index=False)