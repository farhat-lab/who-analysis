# Create Environment

This project uses `conda` to manage packages (install Anaconda <a href="https://www.anaconda.com/" target="_blank">here</a>). All packages are found in the `environment.yaml` file. <b>Change the first and last line of this file to reflect the environment name you wish to use and the path to your Anaconda directory.</b> Then run

```conda env create -f environment.yaml```

to create the environment, which should take no more than 30 munutes. Run `conda activate <env_name>` to activate it and `conda deactivate` to deactivate once you are in it.

# Required Computing Resources

<ul>
        <li>The genotypes matrices in this work are very large, and depending on the drug, can require up to 100 GB of RAM to run.</li>
        <li>All models were run on 1 CPU core. A single model can take anywhere from a few minutes to a few hours to run.</li>
        <li>Drugs with smaller datasets, such as Delamanid, will run much faster than drugs with many variants and samples, such as Ethambutol.</li>
</ul>

# <code>data/</code>

1. <code>drug_CC.csv</code>: Critical concentrations for each drug used for binarization of MIC data.
2. <code>drug_gene_mapping.csv</code>: Names of genes and tiers used to build models for each drug.
3. <code>drugs_loci.csv</code>: Dataframe of resistance loci and their coordinates in H37Rv. The Start column is 0-indexed, and the End column is 1-indexed.
4. <code>NotAwR by literature.xlsx</code>: List of variants by drug that are not associated with resistance based on literature evidence.
5. <code>coll2014_SNP_scheme.tsv</code>: Orthogonal lineage-defining SNPs for 62 lineages based on the <a href="https://www.nature.com/articles/ncomms5812" target="_blank">Coll 2014 scheme</a>.
6. <code>overview_MIC_data.xlsx</code>: Overviews of MIC data, counts, sources, number resistant vs. susceptible, etc.
7. <code>samples_summary.csv</code>: Dataframe of the number of samples across drugs. Includes columns for the numbers of samples with genotypes, binary phenotypes, SNP counts, MICs, lineages, and the numbers of (sample, gene) pairs with LOF and inframe mutations (to see how many get pooled).
8. <code>v1_to_v2_variants_mapping.csv</code>: File mapping the variant names between the <a href="https://www.who.int/publications/i/item/9789240028173" target="_blank">first</a> and <a href="https://www.who.int/publications/i/item/9789240082410" target="_blank">second</a> versions of the catalog.

# <code>PCA/</code>

1. <code>pca_explained_var.npy</code>: Array of explained variances of the first 50 principal components.
2. <code>pca_explained_var_ratio.npy</code>: Array of explained variance ratios (array sums to 1) of the first 50 principal components (PCs).
3. <code>eigenvec_50PC.csv</code>: First 50 eigenvectors of the PCA run on 6,190 single nucleotide variant sites.
4. <code>minor_allele_counts.pkl.gz</code>: 52,567 (isolates) × 6,938 (positions) matrix of minor allele counts (binary encoding) across the full genome. Note that uncompressed, this file is almost 3 GB, so ensure that you have enough RAM to read it in.
5. <code>mixed_site_counts.xlsx</code>: SNVs for PCA with the proportion of isolates containing an unfixed variant (25% < AF ≤ 75%). Used for filtering out sites at which more than 1% of isolates have an unfixed variant.
6. <code>Vargas_PNAS_2023_homoplasy.xlsx</code>: List of 1,525 homoplasic sites in MTBC. Dataset S1 from <a href="https://www.pnas.org/doi/10.1073/pnas.2301394120" target="_blank">Vargas <i>et al., PNAS</i>, 2023</a>.

# Running the Grading Algorithm

## Raw Data

Due to their large size, the raw genotypes, phenotypes, and MICs are available to download from the releases page of this repository. Each drug has a separate folder, which contains `genos_1.csv.gz`, `phenos_binary.csv`, and `phenos_mic.csv`. These were created by concatenating individual CSV files from this <a href="https://github.com/GTB-tbsequencing/mutation-catalogue-2023/tree/main/Input%20data%20files%20for%20Solo%20algorithms/2023-04-25T06_00_10.443990_jr_b741dc136e079fa8583604a4915c0dc751724ae9880f06e7c2eacc939e086536" taget="_blank">repository</a>.

To use this data, create a new directory and update the <code>output_dir</code> parameter in the `config.yaml` file (more about this later) to this same directory name. <b>Place each drug folder into this output directory. Keep the 3 files for each drug in separate drug-specific subfolders as in the release.</b>

When you run the scripts, they will write the results to the same `output_dir` where the raw data are. The scripts will skip analyses that have already been done, so if a step is interrupted, it will pick up at the next one by detecting which steps have already completed based on the files that have been created.
        
## 0. Preprocessing (<code>preprocessing/</code>):
    
Before running any models, run the two scripts in the <code>preprocessing</code> directory in numerical order.

1. <code>preprocessing/01_make_gene_drug_mapping.py</code> creates the file <code>data/drug_gene_mapping.csv</code>, which maps the input gene names to each drug, which facilitates constructing the input model matrices
2. <code>preprocessing/02_samples_summary_andPCA.py</code> generates 50 eigenvectors for population structure correction and saves them to <code>PCA/eigenvec_50PC.csv</code>. Intermediate files that this script creates are a dataframe of minor allele counts (<code>data/minor_allele_counts.pkl;</code>) and an array of the explained variance ratios of each of the 50 principal components (<code>data/pca_explained_var.npy</code>). These files were too large to commit to this repository.

## 1. Model Scripts (<code>model/</code>)

All samples with any missing variants or unfixed variants were excluded from models. This is done at the model-level, so a sample can be exluded from one model but not another for the same drug. <code>scikit-learn</code> can not fit models with NaNs, and imputation can introduce biases, so we decided to drop all samples with missingness.

The following model scripts require the config file (`config.yaml`) and the full drug name (first letter capitalized) to run the analysis on as arguments.
  
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
</ul>

Some of the parameters above were kept constant throughout all analyses, but they remained as parameters if they need to be toggled in the future. We fit 9 models per drug: 

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

## 2. Grading Algorithm (`/grading`)

After running the scripts in `/model`, run the two numbered scripts in `/grading` to run the grading algorithm. The algorithm will be run on all the models found in the specified folders.

1. <code>01_get_single_model_results.py</code>: Combines results from all the permutation test, LRT, and univarite statistics into a single table for each (model, drug) pair. Reuslts of all logistic regression models for a single drug are written to a single Excel file in a new `/results` directory in the home directory of the repository. The only required argument is `config.yaml` to get the directory in which the model results are stored.
2. <code>02_combine_WHO_ALL_results.py</code>: Integrates results from different models and gets a consensus grading for each (drug, variant) pair. Writes it to an output file. Required arguments: `config.yaml` and an output file to store the regression-based catalog at.

<b>The regression-based catalog results from this work are in the file `/results/Regression_Final_June2024_Tier1.csv`</b>

## 3. Resistance Predictions (`/prediction`)

1. <code>catalog_model.py</code>: Uses the final regression catalog (created from <code>02_combine_WHO_ALL_results.py</code>) to get resistance predictions. Any isolate that contains a Group 1 or 2 mutations is predicted resistance.
2. <code>catalog_model_SOLO.py</code>: Does the same as the above script for the "SOLO INITIAL" results. The "SOLO FINAL" results were taken from Table 3 of the WHO report.
