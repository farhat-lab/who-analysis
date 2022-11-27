# WHO <i>M. tuberculosis</i> Resistance Mutation Catalog, 2022

## Create Environment

Add required channels, then build the environment from the `environment_reqs.txt` file

<code>conda config --add channels bioconda</code>

<code>conda config --add channels conda-forge</code>

<code>conda create --name env_name --file environment_reqs.txt</code>
    
If <code>fast-lineage-caller</code> does not install properly from the environment requirements file, install it with <code>pip install fast-lineage-caller</code>. I have had issues in the past installing it with conda. 
    
<!-- ## Genotype Annotations

<ul>
    <li>resolved_symbol: gene name</li>
    <li>variant_category</li>
    <ul>
        <li>p: coding variants</li>
        <li>c: synonymous or upstream variants</li>
        <li>n: non-coding variants in <i>rrs/rrl</i></li>
        <li>deletion: large-scale deletion of a gene</li>
    </ul>
    <li>Effect</li>
    <ul>
        <li>upstream_gene_variant</li>
        <li>missense_variant, synonymous_variant, inframe_deletion, inframe_insertion, stop_lost: self-explanatory</li>
        <li>lof: any frameshift, large-scale deletion, nonsense, or loss of start mutation</li>
        <li>initiator_codon_variant: Valine start codon</li>
        <li>stop_retained_variant: variant in the stop codon that did not change it</li>
    </ul>
</ul>

## Phenotype Annotations

<ul>
    <li>Variant binary status</li>
        <ul>
            <li>1 if the variant meets QC and AF > 0.75</li>
            <li>0 if the variant meets QC, and AF < 0.25</li>
            <li>NA if the variant does not meet QC or 0.25 $\leq$ AF $\leq$ 0.75</li>
        </ul>
    <li>Variant allele frequency</li>
        <ul>
            <li>AF if the variant meets QC and AF $\geq$ 0.25</li>
            <li>0 if the variant meets QC and AF < 0.25</li>
            <li>NA if the variant does not meet QC</li>
        </ul>
</ul> -->

## Files in the Data Directory

1. <code>drug_CC.csv</code>: Critical concentrations for each drug used for binarization of MIC data.
2. <code>drug_gene_mapping.csv</code>: Names of genes and tiers used to build models for each drug.
3. <code>drugs_loci.csv</code>: Dataframe of resistance loci and their coordinates in H37Rv. The Start column is 0-indexed, and the End column is 1-indexed.
4. <code>minor_allele_counts.npz</code>: Minor allele counts from the SNP matrices: 1 for a minor allele, 0 for a major allele. The genetic relatedness matrix is computed from this.
5. <code>overview_MIC_data.xlsx</code>: Overviews of MIC data, counts, sources, number resistant vs. susceptible, etc.
6. <code>samples_summary.csv</code>: Dataframe of the number of samples across drugs. Includes columns for the numbers of samples with genotypes, phenotypes, SNP counts, lineages, and the number of (sample, gene) pairs with multiple frameshift mutations.
7. <code>v1_to_v2_variants_mapping.csv</code>: File mapping the variant names between the 2021 and 2022 iterations of the catalog.

## Running the Analysis
        
### Core Model Features:
    
<ul>
    <li>Tier 1 genes</li>
    <li>Isolates with WHO-approved phenotypes</li>
    <li>No synonymous mutations</li>
    <li>Pool loss-of-function (LOF) mutations</li>
    <li>Pool inframe (insertions and deletions) mutations</li>
    <li>Drop isolates with ambiguous allele frequencies (i.e. "HETs")</li>
</ul>
    
### Model Scripts

Run the numbered scripts in order, with the `config.yaml` file, the full drug name, and the 3-letter abbreviation used in the 2021 WHO catalog. For example, for isoniazid, the arguments after the script name would be `config.yaml Isoniazid INH`. 
   
1. <code>01_make_model_inputs.py</code>: create input matrices to fit a regression model.
2. <code>02_regression_with_bootstrap.py</code> performs a Ridge (L2-penalized) regression. 
3. <code>03_model_analysis.py</code> gets p-values (including false discovery rate-corrected p-values) and confidence intervals for the coefficients/odds ratios. It creates a summary file called `model_analysis.csv` in every output directory, which contains all variants with non-zero coefficients and nominally significant p-values (p-value before FDR is less than 0.05).
    
Parameters in the yaml file are as follows:
    
<ul>
    <li><code>binary</code>: boolean for whether to fit a binary or quantitative (MIC) model</li>
    <li><code>tiers_lst</code>: list of tiers to include in the model</li>
    <li><code>pheno_category_lst</code>: list of phenotype categories to include. The list can include the strings WHO and ALL.</li>
    <li><code>synonymous</code>: boolean for whether synonymous variants should be included</li>
    <li><code>pool_lof</code>: boolean for whether or not LOF variants should be pooled</li>
    <li><code>amb_mode</code>: how to handle mutations with intermediate AF. Options are DROP, AF, and BINARY. </li>
    <li><code>missing_isolate_thresh</code>: threshold for missing isolates (0-1). i.e. if an isolate has more than N% of variants missing, drop it.</li>
    <li><code>missing_feature_thresh</code>: threshold for missing variants (0-1), i.e. if a variant has more than N% of isolates missing, drop it.</li>
    <li><code>AF_thresh</code>: Only used if <code>amb_mode</code> = BINARY. Variants with AF > the threshold will be assigned to 1, the others to 0.</li>
    <li><code>drop_isolates_before_variants</code>: boolean to drop isolates with lot of missingness before variants. If this is set to False, the one with more missingness will be dropped first.</li>
    <li><code>impute</code>: boolean for whether missing values should be imputed (if False, then they will be dropped)</li>
    <li><code>num_PCs</code>: number of principal components (>= 0)</li>
    <li><code>num_bootstrap</code>: number of bootstrap samples</li>
    <li><code>alpha</code>: significance level</li>
</ul>

### Order of Models:

1. Tier 1, WHO, no synonymous, DROP Hets
<!-- 2. Tier 1, WHO, no synonymous, DROP Hets, <span style="background-color: #BFF8D4">unpool LOFs and inframes</span> -->
2.  Tier 1, WHO, no synonymous, DROP Hets, <b>unpool LOFs and inframes</b>
3. 	Tier 1, WHO, with synonymous, DROP Hets
4. 	Tier 1+2, WHO, no synonymous, DROP Hets
5. 	Tier 1+2, WHO, no synonymous, DROP Hets, unpool LOFs and inframes
6. 	Tier 1+2, WHO, with synonymous, DROP Hets
7. 	Tier 1, ALL, no synonymous, DROP Hets
8. 	Tier 1, ALL, no synonymous, DROP Hets, unpool LOFs and inframes
9. 	Tier 1, ALL, with synonymous, DROP Hets
10. Tier 1+2, ALL, no synonymous, DROP Hets
11. Tier 1+2, ALL, no synonymous, DROP Hets, unpool LOFs and inframes
12. Tier 1+2, ALL, with synonymous, DROP Hets
13. Tier 1, WHO, no synonymous, Hets as AF
14. Tier 1+2, WHO, no synonymous, Hets as AF
15. Tier 1, ALL, no synonymous, Hets as AF
16. Tier 1+2, ALL, no synonymous, Hets as AF

### Analysis Scripts:

AFTER running the 3 scripts in the home directory for a given drug, run the 3 numbered scripts in the `analysis` directory in order. Each script requires only the drug name as an argument. The analysis scripts must be run AFTER running all models with all parameter combinations. The analysis scripts combine the results from multiple models and output a single final dataframe. 
  
1. <code>analysis/01_combine_model_analyses.py</code> combines the `model_analysis.csv` files from different models for the same drug and generates a summary dataframe of variants and in which type of model they were detected. The core model has <b> tier 1 genes only, WHO phenotypes only, no synonymous mutations, and pooling LOF mutations</b>. This dataframe will contain additional boolean columns indicating in which model a variant was detected. 
2. <code>analysis/02_compute_univariate_stats.py</code> computes univariate statistics, such as <b>sensitivity, specificity, likelihood ratios</b>, and <b>positive predictive value</b>. These columns are appended to the `model_analysis.csv` file created by script #3. 
3. <code>analysis/03_model_metrics.py</code> fits a new regression model using only the significant features (coefficient confidence interval does not include 0) and performs nonparametric bootstrapping to get confidence intervals for model statistics. 
 
### Pooling LOF Mutations
    
If the argument `pool_lof` is set to True, then LOF mutations are pooled for each (sample, gene) pair. A custom function is used for this so that genes containing multiple frameshift mutations are not considered LOF. If this is decided against, then pooling can be done on the `Effect` column in the genotypes dataframes. 
    
When an LOF variant (i.e. loss of start or stop codons, early stop, large deletion) and multiple frameshift mutations co-occur, LOF will be generated as a new feature, and the frameshift mutations will remain as additional features. If an LOF variant co-occurs with a single frameshift mutation, then they are pooled into a single LOF feature. 

### Intermediate Allele Frequencies

Intermediate = allele fractions in the range [0.25, 0.75]. Below this range, alleles are reference (0), and above it, they are alternative (1). Selection of how to encode intermediate allele frequencies is made using the `config.yaml` file. The options are:

<ul>
    <li>Drop all isolates containing any ambiguous mutations.</li>
    <li>Encode all variants with AF > 0.25 (including those with AF > 0.75) with their AF, not binary. </li>
    <li>Binarize them using an AF threshold</li>
</ul>

Currently, all isolates with any intermediate allele frequencies are <b>dropped</b> from a given model before running.

### Missing Data

Isolates with a lot of missingness are far more common than features with a lot of missingness because most of the sequenced regions have high mappability, except the ribosomal regions. 
    
A threshold of <b>1%</b> is used to drop isolates, i.e. if more than 1% of an isolate's features for a given analysis are NA, then drop the isolate. 
Similarly, if more than <b>25%</b> of the isolates in a given analysis are missing that feature, then drop the feature. 

Then, all remaining features with anything missing are dropped. Imputation can be done instead by setting the argument `impute` to True. This will impute every element in the matrix that is missing by averaging the feature, stratified by resistance phenotype. 
