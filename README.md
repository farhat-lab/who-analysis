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
6. <code>samples_summary.csv</code>: Dataframe of the number of samples across drugs. Includes columns for the numbers of samples with genotypes, binary phenotypes, SNP counts, MICs, lineages, and the numbers of (sample, gene) pairs with LOF and inframe mutations (to see how many get pooled).
7. <code>v1_to_v2_variants_mapping.csv</code>: File mapping the variant names between the 2021 and 2022 iterations of the catalog.

## Running the Analysis
        
### Primary Model Features:
    
<ul>
    <li>Tier 1 genes</li>
    <li>Isolates with WHO-approved phenotypes</li>
    <li>No synonymous mutations</li>
    <li>Pool loss-of-function (LOF) mutations</li>
    <li>Pool inframe (insertions and deletions) mutations</li>
    <li>Drop isolates with ambiguous allele frequencies (i.e. "HETs")</li>
</ul>
    
### Model Scripts

Before building any models, run <code>00_samples_summary_minor_allele_counts.py</code> and <code>make_gene_drug_mapping.py</code>.

The first script generates the dataframe of minor allele counts (<code>data/minor_allele_counts.npz</code>) from the SNP data directory. We compute the genetic relatedness matrix for population structure correction from this matrix. The script also creates the <code>data/samples_summary.csv</code> file to see how many samples there are for each drug (this was primarily used for debugging).

The second script generates a dataframe mapping every gene to the drug it is hypothesized or known to be relevant for and the tier. This dataframe is used to select variants to pass into the model from the master <code>genos.csv.gz</code> file for each drug. 

For every drug, run the following numbered scripts in order, with `config.yaml`, the full drug name, and the 3-letter abbreviation used in the 2021 WHO catalog. For example, for isoniazid, the arguments after the script name would be `config.yaml Isoniazid INH`. 
  
1. <code>01_make_model_inputs.py</code>: create input matrices to fit a regression model.
2. <code>02_regression_with_bootstrap.py</code> performs a Ridge (L2-penalized) regression. 
3. <code>03_model_analysis.py</code> computes p-values (including false discovery rate-corrected p-values) and confidence intervals for the coefficients/odds ratios for each mutation in the model. It creates a summary file called `model_analysis.csv` in every model output directory.
    
Parameters in the yaml file:
    
<ul>
    <li><code>input_dir</code>: Directory where all input directories are stored. Contains subfolders "grm", "phenotypes", and "full_genotypes".</li>
    <li><code>output_dir</code>: Directory where model results will be written.</li>
    <li><code>binary</code>: boolean for whether to fit a binary or quantitative (MIC) model</li>
    <li><code>atu_analysis</code>: boolean for whether this is the normal binary analysis or a CC vs. CC-ATU analysis</li>
    <li><code>atu_analysis_type</code>: string "CC" or "CC-ATU" denoting which model to run in this analysis. Only used if <code>atu_analysis = True</code></li>
    <li><code>tiers_lst</code>: list of tiers to include in the model</li>
    <li><code>pheno_category_lst</code>: list of phenotype categories to include. The list can include the strings WHO and ALL. Only used if <code>binary = True</code> and <code>atu_analysis = False</code></li>
    <li><code>synonymous</code>: boolean for whether synonymous variants should be included</li>
    <li><code>pool_type</code>: one of 3 strings (<code>poolSeparate</code>, <code>poolALL</code>, or <code>unpooled</code>). The first pools features into 2 aggregate features: "LOF" and "inframe". The second pools both into a combined feature "LOF_all," and the third disaggregates all features.</li>
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

Some of the parameters above were kept constant throughout all analyses, but they remained as parameters if they need to be toggled in the future. 

### Order of Binary Models:

1. Tier 1, WHO, no synonymous, DROP Hets, pool LOF and inframe mutations SEPARATELY
2. Tier 1, WHO, no synonymous, DROP Hets, <b>pool LOF and inframe mutations TOGETHER</b>
3. Tier 1, WHO, no synonymous, DROP Hets, <b>unpool LOFs and inframes</b>
4. Tier 1, WHO, <b>with synonymous</b>, DROP Hets, pool LOF and inframe mutations SEPARATELY
5. <b>Tier 1+2</b>, WHO, no synonymous, DROP Hets, pool LOF and inframe mutations SEPARATELY
6. <b>Tier 1+2</b>, WHO, no synonymous, DROP Hets, <b>pool LOF and inframe mutations TOGETHER</b>
7. <b>Tier 1+2</b>, WHO, no synonymous, DROP Hets, <b>unpool LOFs and inframes</b>
8. <b>Tier 1+2</b>, WHO, <b>with synonymous</b>, DROP Hets, pool LOF and inframe mutations SEPARATELY
9. Tier 1, <b>ALL</b>, no synonymous, DROP Hets, pool LOF and inframe mutations SEPARATELY
10. Tier 1, <b>ALL</b>, no synonymous, DROP Hets, <b>pool LOF and inframe mutations TOGETHER</b>
11. Tier 1, ALL, no synonymous, DROP Hets, <b>unpool LOFs and inframes</b>
12. Tier 1, <b>ALL, with synonymous</b>, DROP Hets, pool LOF and inframe mutations SEPARATELY
13. <b>Tier 1+2, ALL</b>, no synonymous, DROP Hets, pool LOF and inframe mutations SEPARATELY
14. <b>Tier 1+2, ALL</b>, no synonymous, DROP Hets, <b>pool LOF and inframe mutations TOGETHER</b>
15. <b>Tier 1+2, ALL</b>, no synonymous, DROP Hets, <b>unpool LOFs and inframes</b>
16. <b>Tier 1+2, ALL</b>, <b>with synonymous</b>, DROP Hets, pool LOF and inframe mutations SEPARATELY
17. Tier 1, WHO, no synonymous, <b>encode AF</b>, pool LOF and inframe mutations SEPARATELY
18. <b>Tier 1+2</b>, WHO, no synonymous, <b>encode AF</b>, pool LOF and inframe mutations SEPARATELY
19. Tier 1, <b>ALL</b>, no synonymous, <b>encode AF</b>, pool LOF and inframe mutations SEPARATELY
20. <b>Tier 1+2, ALL</b>, no synonymous, <b>encode AF</b>, pool LOF and inframe mutations SEPARATELY

### Final Analysis

After all 20 models above have been run, run <code>04_compute_univariate_stats.py</code> to compute univariate statistics, 95% binomial confidence intervals, and add some other annotations for the mutations in all models. This script requires the following arguments:

<ul>
    <li><code>drug</code> (same as the previous scripts)</li>
    <li><code>folder</code> (should be one of "BINARY", "ATU", or "MIC", depending on the analysis</li>
    <li><code>input_dir</code> (same as the previous scripts)</li>
</ul>
    
Finally, run the top cells in the Jupyter notebook `analysis/analysis.ipynb` to write all model analyses to individual Excel files (one for each drug) and generate a summary Excel file for all drugs.

### Intermediate Allele Frequencies

Intermediate = allele fractions in the range [0.25, 0.75]. Below this range, alleles are reference (0), and above it, they are alternative (1). Selection of how to encode intermediate allele frequencies is made using the `config.yaml` file. The options are:

<ul>
    <li>Drop all isolates containing any ambiguous mutations.</li>
    <li>Encode all variants with AF > 0.25 (including those with AF > 0.75) with their AF, not binary. </li>
    <li>Binarize them using an AF threshold</li>
</ul>

Currently, we are using the top 2 modes. In the code, they are referred to as <b>dropAF</b> and <b>encodeAF</b>, respectively. 

### Missing Data

Isolates with a lot of missingness are far more common than features with a lot of missingness because most of the sequenced regions have high mappability, except the ribosomal regions. 
    
A threshold of <b>1%</b> is used to drop isolates, i.e. if more than 1% of an isolate's features for a given analysis are NA, then drop the isolate. 
Similarly, if more than <b>25%</b> of the isolates in a given analysis are missing that feature, then drop the feature. 

Then, all remaining features with anything missing are dropped. Imputation can be done instead by setting the argument `impute` to True. This will impute every element in the matrix that is missing by averaging the feature, stratified by resistance phenotype. 
