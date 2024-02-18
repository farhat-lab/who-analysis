# WHO <i>M. tuberculosis</i> Resistance Mutation Catalog, 2022

## Create Environment

This project uses `conda` to manage packages (install Anaconda <a href="https://www.anaconda.com/" target="_blank">here</a>). All packages are found in the `environment.yaml` file. Change the last line of this file to reflect the path to your Anaconda distribution and the environment name you want to use. Then run

```conda env create -f environment.yaml```

to create the environment. Run `conda activate <env_name>` to activate it and `conda deactivate` to deactivate once you are in it.
    
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
            <li>0 if the variant meets QC, and AF ≤ 0.25</li>
            <li>NA if the variant does not meet QC or 0.25 $\leq$ AF $\leq$ 0.75</li>
        </ul>
    <li>Variant allele frequency</li>
        <ul>
            <li>AF if the variant meets QC and AF $\geq$ 0.25</li>
            <li>0 if the variant meets QC and AF ≤ 0.25</li>
            <li>NA if the variant does not meet QC</li>
        </ul>
</ul> -->

## <code>data/</code>

1. <code>drug_CC.csv</code>: Critical concentrations for each drug used for binarization of MIC data.
2. <code>drug_gene_mapping.csv</code>: Names of genes and tiers used to build models for each drug.
3. <code>drugs_loci.csv</code>: Dataframe of resistance loci and their coordinates in H37Rv. The Start column is 0-indexed, and the End column is 1-indexed.
4. <code>NotAwR by literature.xlsx</code>: List of variants by drug that are not associated with resistance based on literature evidence.
5. <code>coll2014_SNP_scheme.tsv</code>: Orthogonal lineage-defining SNPs for 62 lineages based on the <a href="https://www.nature.com/articles/ncomms5812" target="_blank">Coll 2014 scheme</a>.
6. <code>overview_MIC_data.xlsx</code>: Overviews of MIC data, counts, sources, number resistant vs. susceptible, etc.
7. <code>samples_summary.csv</code>: Dataframe of the number of samples across drugs. Includes columns for the numbers of samples with genotypes, binary phenotypes, SNP counts, MICs, lineages, and the numbers of (sample, gene) pairs with LOF and inframe mutations (to see how many get pooled).
8. <code>v1_to_v2_variants_mapping.csv</code>: File mapping the variant names between the <a href="https://www.who.int/publications/i/item/9789240028173" target="_blank">first</a> <a href="https://www.who.int/publications/i/item/9789240082410" target="_blank">second</a> iterations of the catalog.

## <code>PCA/</code>

* Note: The minor allele counts file (<code>minor_allele_counts.pkl</code>) eigenvectors (<code>eigenvec_100PC.csv</code>) were not committed to the repository because these file are too large.
   
1. <code>pca_explained_var.npy</code>: Array of explained variances of the first 100 principal components.
2. <code>pca_explained_var_ratio.npy</code>: Array of explained variance ratios (array sums to 1) of the first 100 principal components.
3. <code>Vargas_PNAS_2023_homoplasy.xlsx</code>: List of 1,525 homoplasic sites in MTBC. Dataset S1 from <a href="https://www.pnas.org/doi/10.1073/pnas.2301394120" target="_blank">Vargas <i>et al., PNAS</i>, 2023</a>.
4. <code>mixed_site_counts.xlsx</code>: SNVs for PCA with the proportion of isolates containing an unfixed variant (25% < AF ≤ 75%). Used for filtering out sites at which more than 1% of isolates have an unfixed variant. 

## Running the Analysis
        
### Primary Model Features:
    
    
### Model Scripts

Before building any models, run <code>00_samples_summary_andPCA.py</code>.

The first script generates the eigenvectors for population structure correction and stores the coordinates of every sample in <code>data/eigenvec_10PC.csv</code>. 10 principal coordinates were saved, but only 5 were used in the models. Intermediate files that this script creates are a dataframe of minor allele counts (<code>data/minor_allele_counts.npz</code>) and an array of the explained variance ratios of each of the 10 principal components (<code>data/pca_explained_var.npy</code>). The script also creates the <code>data/samples_summary.csv</code> file to see how many samples there are for each drug (this was primarily used for debugging).


For every drug, run the bash script <code>run_regression.sh</code> with the following command line arguments: full drug name, the 3-letter abbreviation used in the 2021 WHO catalog, and the directory to which output files should be written. For example, for isoniazid, the arguments after the script name would be `run_regression.sh Isoniazid INH /n/data1/hms/dbmi/farhat/Sanjana/who-mutation-catalogue`. This bash script runs the following scripts:
  
1. <code>01_make_model_inputs.py</code>: create input matrices to fit a regression model.
2. <code>02_run_regression.py</code> performs a Ridge (L2-penalized) regression, bootstrapping to get coefficient confidence intervals, and a permutation test to assess coefficient significance. 
3. <code>03_compute_univariate_statistics.py</code> 

After the above scripts are finished, run <code>run_unpooled_tests.sh</code> with only the full drug name and the 3-letter abbreviation used in the 2021 WHO catalog as additional command line arguments. These scripts run the LRT and AUC test. 

4. <code>04_likelihood_ratio_test.py</code>: Performs LRT for all mutations in the unpooled models.
5. <code>05_AUC_permutation_test.py</code>: Performs a test of how much a mutation contributes to AUC using a permutation test. 
    
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

### All Tier 1-Only Models (9 per drug):

For all models, we dropped isolates containing unfixed variants.

1. WHO, - silent variants, all variants unpooled.
2. WHO, - silent variants, <b>pool</b> LoF and inframe mutations SEPARATELY
3. WHO, <b>+ silent</b> variants, all variants unpooled
4. <b>ALL</b>, - silent variants, all variants unpooled
5. ALL, - silent variants, <b>pool</b> LoF and inframe mutations SEPARATELY
6. ALL, <b>+ silent</b> variants, all variants unpooled
7. MIC, - silent variants, all variants unpooled
8. MIC, - silent variants, <b>pool</b> LoF and inframe mutations SEPARATELY
9. MIC, <b>+ silent</b> variants, all variants unpooled

### Final Analysis

TODO

### Missing Data

All samples with any missing data are excluded from models. This is done at the model-level, so a sample can be exluded from one model but not another for the same drug. <code>scikit-learn</code> can not fit models with NaNs, and imputation can introduce biases, so we decided to drop all samples with missingness.

<!-- Isolates with a lot of missingness are far more common than features with a lot of missingness because most of the sequenced regions have high mappability, except the ribosomal regions. 
    
A threshold of <b>1%</b> is used to drop isolates, i.e. if more than 1% of an isolate's features for a given analysis are NA, then drop the isolate. 
Similarly, if more than <b>25%</b> of the isolates in a given analysis are missing that feature, then drop the feature. 

Then, all remaining features with anything missing are dropped. Imputation can be done instead by setting the argument `impute` to True. This will impute every element in the matrix that is missing by averaging the feature, stratified by resistance phenotype.  -->


Matrix from Sacha after removing sites close to PE/PPE genes and applying the 1% variant frequency threshold: 6,938 sites.


6614/6938 sites do not have low AF in more than 1.0% of isolates 


6491/6614 sites are not in drug-resistance regions 


6190/6491 sites are not homoplastic 


Final size: (52567, 6190) 
