# World Health Organization TB Resistance Mutation Catalog, 2022

## Genotype Annotations

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
</ul>

## Files in the Data Directory

1. <code>drugs_loci.csv</code>: Dataframe of resistance loci and their coordinates in H37Rv. The Start column is 0-indexed, and the End column is 1-indexed.
2. <code>minor_allele_counts.npz</code>: Minor allele counts from the SNP matrices: 1 for a minor allele, 0 for a major allele. The genetic relatedness matrix is computed from this.
3. <code>samples_summary.csv</code>: Dataframe of the number of samples across drugs. Includes columns for the numbers of samples with genotypes, phenotypes, SNP counts, lineages, and the number of (sample, gene) pairs with multiple frameshift mutations.

## Running the Analysis

Run the numbered scripts in order, with the `config.yaml` file, the full drug name, and the 3-letter abbreviation used in the 2021 WHO catalog. For example, for isoniazid, the arguments after the script name would be `config.yaml Isoniazid INH`. Parameters in the yaml file are as follows:
    
<ul>
    <li><code>tiers_lst</code>: list of tiers to include in the model</li>
    <li><code>pheno_category_lst</code>: list of phenotype categories to include. The list can include the strings WHO and ALL.</li>
    <li><code>model_prefix</code>: directory name for the current analysis</li>
    <li><code>missing_isolate_thresh</code>: threshold for missing isolates (0-1). i.e. if an isolate has more than N% of variants missing, drop it.</li>
    <li><code>missing_feature_thresh</code>: threshold for missing variants (0-1), i.e. if a variant has more than N% of isolates missing, drop it.</li>
    <li><code>het_mode</code>: how to handle heterozygous alleles. Options are DROP, AF, and BINARY. </li>
    <li><code>AF_thresh</code>: Only used if <code>het_mode</code> = BINARY. Heterozygous alleles with AF > the threshold will be assigned to 1, the others to 0.</li>
    <li><code>drop_isolates_before_variants</code>: boolean to drop isolates with lot of missingness before variants. If this is set to False, the one with more missingness will be dropped first.</li>
    <li><code>impute</code>: boolean for whether missing values should be imputed (if False, then they will be dropped)</li>
    <li><code>synonymous</code>: boolean for whether synonymous variants should be included</li>
    <li><code>pool_lof</code>: boolean for whether or not LOF variants should be pooled</li>
    <li><code>num_PCs</code>: number of principal components (>= 0)</li>
    <li><code>num_bootstrap</code>: number of bootstrap samples</li>
    <li><code>alpha</code>: significance level</li>
</ul>
    
### Pooling LOF Mutations
    
If the argument `pool_lof` is set to True, then LOF mutations are pooled for each (sample, gene) pair. A custom function is used for this so that genes containing multiple frameshift mutations are not considered LOF. If this is decided against, then pooling can be done on the `Effect` column in the genotypes dataframes. 
    
When an LOF variant (i.e. loss of start or stop codons, early stop, large deletion) and multiple frameshift mutations co-occur, LOF will be generated as a new feature, and the frameshift mutations will remain as additional features. If an LOF variant co-occurs with a single frameshift mutation, then they are pooled into a single LOF feature. 

### Heterozygous Alleles

Heterozygous variants have allele fractions in the range [0.25, 0.75]. Below this range, alleles are reference (0), and above it, they are alternative (1). Selection of how to encode heterozygous alleles is made using the `config.yaml` file. The options are:

<ul>
    <li>Drop</li>
    <li>Encode all variants with AF > 0.25 (including those with AF > 0.75) with their AF, not binary. </li>
    <li>Binarize them using an AF threshold</li>
</ul>

### Missing Data

Isolates with a lot of missingness are far more common than features with a lot of missingness because most of the sequenced regions have high mappability, except the ribosomal regions. 
    
A threshold of <b>1%</b> is used to drop isolates, i.e. if more than 1% of an isolate's features for a given analysis are NA, then drop the isolate. 
Similarly, if more than <b>25%</b> of the isolates in a given analysis are missing that feature, then drop the feature. 

Then, all remaining features with anything missing are dropped. Imputation can be done instead by setting the argument `impute` to True. This will impute every element in the matrix that is missing by averaging the feature, stratified by resistance phenotype. 
    
---
**NOTE**

Insert table of analyses.

---

## TO-DO:
1. Run models with LOF pooling last. Add step that checks if the model matrix is the same shape as the corresponding model without LOF pooling (i.e. same synonymous, HET treatment, tiers, and phenotypes), and stops the model if it's the same to save time.
2. Some BDQ and LZD sample IDs have not been found in the sample IDs:VCF names mapping file. 
