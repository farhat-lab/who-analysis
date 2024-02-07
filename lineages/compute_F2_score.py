import vcf
import os
import pandas as pd
import numpy as np
import sys
import pickle
import subprocess

# INPUTS: gzipped VCF file, directory for a single sample where to store F2 scores, lineage-defining reference positions
_, vcf_fName = sys.argv

# suffix
sample_ID = os.path.basename(vcf_fName).replace("_lineage_positions.vcf", "")

if not os.path.isfile(vcf_fName):
    print(f"No VCF file named {vcf_fName}")
    exit()


##########################################################################################################################################################


# Lineage Defining SNPs - Import full set of Lineage Defining SNP sets from Coll et. al. 2014
lineage_defining_SNPs = pd.read_csv("~/who-analysis/lineages/Coll2014_SNPs_full.csv").rename(columns={"#lineage": "lineage"})
lineage_defining_SNPs['lineage'] = [val.replace('lineage', '') for val in lineage_defining_SNPs['lineage'].values]

# According to Wyllie et. al. 2018, drop SNP sets corresponding to branches with fewer than 20 SNPs
# excluded_branches = ['1.2' , '3.1' , '3.1.2' , '4.1.2' , '4.3.4.2.1' , '4.6' , '4.7']
exclude_branch_SNP_thresh = 20

# use lineage as the counts column too, so the index and the column have the same name
excluded_branches = pd.DataFrame(lineage_defining_SNPs.groupby('lineage')['lineage'].count()).query("lineage < @exclude_branch_SNP_thresh").index.values
print(f"Excluding lineages {excluded_branches} with fewer than {exclude_branch_SNP_thresh} SNPs defining them")
full_length = len(lineage_defining_SNPs)

lineage_defining_SNPs = lineage_defining_SNPs.query("lineage not in @excluded_branches").reset_index(drop=True)
print(f"Keeping {len(lineage_defining_SNPs)}/{full_length} lineage-defining SNPs from the Coll 2014 scheme")

# separate REF and ALT alleles
lineage_defining_SNPs['REF'] = [val.split('/')[0] for val in lineage_defining_SNPs['allele_change'].values]
lineage_defining_SNPs['ALT'] = [val.split('/')[1] for val in lineage_defining_SNPs['allele_change'].values]


##########################################################################################################################################################


def check_call_is_SNP(ref_allele , alt_alleles):
	
	'''This function checks to see if Call is a SNP and not an InDel or Structural Variant'''
	
	#check that Reference Allele is 1 base
	if len(ref_allele) == 1:
		
		#check to see if there is no alternate allele
		if alt_alleles == [None]:
			
			good_SNP = True
			
		#if there is an alternate allele(s) check to see that they are all just 1 base
		elif ( sum( [(len(alt_allele) == 1) for alt_allele in alt_alleles] ) == len(alt_alleles) ):
			
			good_SNP = True
			
		#at least 1 alternate allele was longer than 1 base
		else:
			
			good_SNP = False
	
	#reference allele was longer than 1 base        
	else:
		good_SNP = False
			
	return good_SNP


##########################################################################################################################################################


def get_lineage_defining_SNP_depths(vcf_fName):
    '''
    This function gets the minor alleles and depth information for all SNPs in the full Coll 2014 scheme. 
    '''
    
    lineage_df = pd.DataFrame(columns=['position', 'Depth', 'REF', 'ALT', 'Minor_Depth'])

    i = 0

    vcf_reader = vcf.Reader(filename=vcf_fName)

    # iterate through each record from VCF file
    for record in vcf_reader:
    
        # check to see if call is at a lineage defining site AND that call is a SNP
        if ( record.POS in list( lineage_defining_SNPs.position ) ) and check_call_is_SNP(record.REF, record.ALT):
    
            # check that the SNP identities match too
            #if record.REF == lineage_defining_SNPs.query("position == @record.POS")['REF'].values[0] and record.ALT[0] == lineage_defining_SNPs.query("position == @record.POS")['ALT'].values[0]:
                
            ref_depth = record.INFO['RO']
            alt_depth = record.INFO['AO'][0]
            total_depth = record.INFO['DP']
    
            # check that the sum of the number of reads supporting ref and the number of reads supporting alt is ≤ total depth
            assert ref_depth + alt_depth <= total_depth
    
            # only need to consider a single alternative allele because the check_call_is_SNP already checks that there is a single alternative allele
            # find the allele with the lower depth (i.e., support)
            # minor_depth = np.min([ref_depth, alt_depth])
            # min_idx = [ref_depth, alt_depth].index(minor_depth)
            # minor_allele = list([record.REF, record.ALT[0]])[min_idx]
    
            # the minor depth should be less than or equal to 1/2 the full depth
            # assert minor_depth <= total_depth / 2
            # lineage_df.loc[i, :] = [record.POS, total_depth, record.REF, str(record.ALT[0]), minor_allele, minor_depth]
            lineage_df.loc[i, :] = [record.POS, total_depth, record.REF, str(record.ALT[0]), alt_depth]
            i += 1

    # keeps only variants that also match the REF and ALT identities in the Coll 2014 scheme
    lineage_df = lineage_df.merge(lineage_defining_SNPs[['position', 'lineage', 'REF', 'ALT']], how='inner', on=['position', 'REF', 'ALT'])
    
    # check that the minor allele is one of REF and ALT
    # assert len(lineage_df.query("Minor_Allele != REF and Minor_Allele != ALT")) == 0
    
    return lineage_df


lineage_df = get_lineage_defining_SNP_depths(vcf_fName)


##########################################################################################################################################################


# sum over m (minor depth), sum over d (total depth) for all SNPs in a lineage set
lineage_df_sets = lineage_df.groupby('lineage')[['Depth', 'Minor_Depth']].sum()

# minor allele fraction, p = M / D
lineage_df_sets['Minor_AF'] = lineage_df_sets['Minor_Depth'] / lineage_df_sets['Depth']

# get the two lineages with the largest minor allele frequencies
top_two_lineages = []

# iterate, and remove lineages that are already encompassed. i.e. if 2.2.1 is in the list already, then don't add 2 or 2.2
for query_lineage in lineage_df_sets.sort_values('Minor_AF', ascending=False).index.values:

    encompassed = False
    
    if len(top_two_lineages) == 2:
        break

    else:
        # if there are already lineages inside
        if len(top_two_lineages) > 0:
            for kept_lineage in top_two_lineages:

                # if the query lineage is a substring of a lineage that is already present, don't add it
                if query_lineage in kept_lineage:
                    encompassed = True

            if not encompassed:
                top_two_lineages.append(query_lineage)
        else:
            top_two_lineages.append(query_lineage)

F2_sets = lineage_df.query("lineage in @top_two_lineages")
F2_metric = F2_sets['Minor_Depth'].sum() / F2_sets['Depth'].sum()
print(f"Top two lineage sets for {sample_ID}: {np.sort(top_two_lineages)}. F2 = {F2_metric}")

with open(os.path.join(os.path.dirname(vcf_fName), f'{sample_ID}_F2.txt') , 'w+') as file:
    file.write(str(F2_metric))