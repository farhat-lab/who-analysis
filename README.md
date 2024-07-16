# WHO <i>M. tuberculosis</i> Resistance Mutation Catalog, 2022

## Create Environment

This project uses `conda` to manage packages (install Anaconda <a href="https://www.anaconda.com/" target="_blank">here</a>). All packages are found in the `environment.yaml` file. Change the last line of this file to reflect the path to your Anaconda distribution and the environment name you want to use. Then run

```conda env create -f environment.yaml```

to create the environment. Run `conda activate <env_name>` to activate it and `conda deactivate` to deactivate once you are in it.

## <code>data/</code>

1. <code>drug_CC.csv</code>: Critical concentrations for each drug used for binarization of MIC data.
2. <code>drug_gene_mapping.csv</code>: Names of genes and tiers used to build models for each drug.
3. <code>drugs_loci.csv</code>: Dataframe of resistance loci and their coordinates in H37Rv. The Start column is 0-indexed, and the End column is 1-indexed.
4. <code>NotAwR by literature.xlsx</code>: List of variants by drug that are not associated with resistance based on literature evidence.
5. <code>coll2014_SNP_scheme.tsv</code>: Orthogonal lineage-defining SNPs for 62 lineages based on the <a href="https://www.nature.com/articles/ncomms5812" target="_blank">Coll 2014 scheme</a>.
6. <code>overview_MIC_data.xlsx</code>: Overviews of MIC data, counts, sources, number resistant vs. susceptible, etc.
7. <code>samples_summary.csv</code>: Dataframe of the number of samples across drugs. Includes columns for the numbers of samples with genotypes, binary phenotypes, SNP counts, MICs, lineages, and the numbers of (sample, gene) pairs with LOF and inframe mutations (to see how many get pooled).
8. <code>v1_to_v2_variants_mapping.csv</code>: File mapping the variant names between the <a href="https://www.who.int/publications/i/item/9789240028173" target="_blank">first</a> and <a href="https://www.who.int/publications/i/item/9789240082410" target="_blank">second</a> versions of the catalog.

## <code>PCA/</code>

1. <code>pca_explained_var.npy</code>: Array of explained variances of the first 50 principal components.
2. <code>pca_explained_var_ratio.npy</code>: Array of explained variance ratios (array sums to 1) of the first 50 principal components (PCs).
3.<code>eigenvec_50PC.csv</code>: First 50 eigenvectors of the PCA run on 6,190 single nucleotide variant sites.
4. <code>minor_allele_counts.pkl.gz</code>: 52,567 (isolates) × 6,938 (positions) matrix of minor allele counts (binary encoding) across the full genome. Note that uncompressed, this file is almost 3 GB, so ensure that you have enough RAM to read it in.
5. <code>mixed_site_counts.xlsx</code>: SNVs for PCA with the proportion of isolates containing an unfixed variant (25% < AF ≤ 75%). Used for filtering out sites at which more than 1% of isolates have an unfixed variant.
6. <code>Vargas_PNAS_2023_homoplasy.xlsx</code>: List of 1,525 homoplasic sites in MTBC. Dataset S1 from <a href="https://www.pnas.org/doi/10.1073/pnas.2301394120" target="_blank">Vargas <i>et al., PNAS</i>, 2023</a>.
7. Figures for the manuscript showing different isolates in principal coordinate space, colored by lineage.

## Running the Analysis
        
### 0. Preprocessing (<code>preprocessing/</code>):
    
Before building any models, run the two scripts in the <code>preprocessing</code> directory in numerical order.

1. <code>preprocessing/01_make_gene_drug_mapping.py</code> creates the file <code>data/drug_gene_mapping.csv</code>, which maps the input gene names to each drug, which facilitates constructing the input model matrices
2. <code>preprocessing/02_samples_summary_andPCA.py</code> generates 50 eigenvectors for population structure correction and saves them to <code>PCA/eigenvec_50PC.csv</code>. Intermediate files that this script creates are a dataframe of minor allele counts (<code>data/minor_allele_counts.pkl;</code>) and an array of the explained variance ratios of each of the 50 principal components (<code>data/pca_explained_var.npy</code>). These files were too large to commit to this repository.

### 1. Model Scripts (<code>model/</code>)

#### Missing Data

All samples with any missing data are excluded from models. This is done at the model-level, so a sample can be exluded from one model but not another for the same drug. <code>scikit-learn</code> can not fit models with NaNs, and imputation can introduce biases, so we decided to drop all samples with missingness.

The following model scripts require the config file (`config.yaml`) and the full drug name to run the analysis on as arguments.
  
1. <code>01_make_model_inputs.py</code>: create input matrices to fit a regression model.
2. <code>02_run_regression.py</code> performs a Ridge (L2-penalized) regression and a permutation test to assess coefficient significance.
3. <code>03_likelihood_ratio_test.py</code>: performs the likelihood ratio test for every mutation in the 
4. <code>04_compute_univariate_statistics.py</code>: computes statistics like sensitivity, specificity, and positive predictive value for each mutation, with 95% exact binomial confidence intervals.

Parameters in the config file:
    
<ul>
    <li><code>input_dir</code>: Directory where all input directories are stored. Contains subfolders "grm", "phenotypes", and "full_genotypes".</li>
    <li><code>output_dir</code>: Directory where model results will be written.</li>
    <li><code>binary</code>: boolean for whether to fit a binary or quantitative (MIC) model</li>
    <li><code>atu_analysis</code>: boolean for whether this is the normal binary analysis or a CC vs. CC-ATU analysis</li>
    <li><code>atu_analysis_type</code>: string "CC" or "CC-ATU" denoting which model to run in this analysis. Only used if <code>atu_analysis = True</code>. We did not run ATU analyses for this work. </li>
    <li><code>tiers_lst</code>: list of integers tiers to include in the model. We only run tier 1 models in this work. </li>
    <li><code>pheno_category_lst</code>: list of phenotype categories to include. The list can include the strings "WHO" and "ALL." Only used if <code>binary = True</code> and <code>atu_analysis = False</code></li>
    <li><code>silent</code>: boolean for whether silent variants (synonymous, initiator codon, and stop retained variants) should be included</li>
    <li><code>pool_type</code>: one of 2 strings (<code>poolSeparate</code> or <code>unpooled</code>). The first pools features into 2 aggregate features: "LoF" and "inframe". The second leaes all variants disaggregated</li>
    <li><code>amb_mode</code>: how to handle mutations with intermediate AF. Options are DROP, AF, and BINARY.</li>
    <li><code>AF_thresh</code>: Only used if <code>amb_mode</code> = BINARY. Variants with AF > the threshold will be assigned to 1, the others to 0.</li>
    <li><code>num_PCs</code>: number of principal components (>= 0)</li>
    <li><code>num_bootstrap</code>: number of bootstrap samples</li>
    <li><code>alpha</code>: significance level</li>
</ul>

Some of the parameters above were kept constant throughout all analyses, but they remained as parameters if they need to be toggled in the future. 

### All Tier 1-Only Models (9 per drug):

For all models, we dropped isolates containing unfixed variants.

1. WHO, - silent variants, all variants unpooled.
2. WHO, - silent variants, <b>pool</b> LoF mutations
3. WHO, <b>+ silent</b> variants, all variants unpooled
4. <b>ALL</b>, - silent variants, all variants unpooled
5. ALL, - silent variants, <b>pool</b> LoF mutations
6. ALL, <b>+ silent</b> variants, all variants unpooled
7. MIC, - silent variants, all variants unpooled
8. MIC, - silent variants, <b>pool</b> LoF mutations
9. MIC, <b>+ silent</b> variants, all variants unpooled

For a single drug, the scripts can be run as follows:

```
drug='Delamanid'

python3 -u model/01_make_model_inputs.py -c config.yaml -d $drug
python3 -u model/02_run_regression.py -c config.yaml -d $drug
python3 -u model/03_likelihood_ratio_test.py -c config.yaml -d $drug
python3 -u model/04_compute_univariate_stats.py -c config.yaml
```

The drug argument is not required for the last script because it computes univariate statistics for all the models it finds in the specified folders.

### 2. Grading Algorithm

After running the scripts in `/model`, run the two numbered scripts in `/grading` to run the grading algorithm. The algorithm will be run on all the models found in the specified folders.

1. <code>01_get_single_model_results.py</code>: Combines results from all the permutation test, LRT, and univarite statistics into a single table for each (model, drug) pair. Reuslts of all logistic regression models for a single drug are written to a single Excel file in a new `/results` directory in the home directory of the repository. The only required argument is `config.yaml` to get the directory in which the model results are stored.
2. <code>02_combine_WHO_ALL_results.py<code>: Integrates results from different models and gets a consensus grading for each (drug, variant) pair. Writes it to an output file. Required arguments: `config.yaml` and an output file to store the regression-based catalog at.

### 3. Resistance Predictions

1. <code>catalog_model.py</code>: Uses the final regression catalog to get resistance predictions. Any isolate that contains a Group 1 or 2 mutations is predicted resistance.
2. <code>catalog_model_SOLO.py<code>: Does the same as the above script for the "SOLO INITIAL" results. The "SOLO FINAL" results were taken from Table 3 of the WHO report.