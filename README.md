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


## Workflow
    
### Pooling LOF Mutations
    
If the argument `pool_lof` is set to True, then LOF mutations are pooled for each (sample, gene) pair. A custom function is used for this so that genes containing multiple frameshift mutations are not considered LOF. If this is decided against, then pooling can be done on the `Effect` column in the genotypes dataframes. 
    
When an LOF variant (i.e. loss of start or stop codons, early stop, large deletion) and multiple frameshift mutations co-occur, LOF will be generated as a new feature, and the frameshift mutations will remain as additional features. If an LOF variant co-occurs with a single frameshift mutation, then they are pooled into a single LOF feature. 

### Heterozygous Alleles

Heterozygous variants have allele fractions in the range [0.25, 0.75]. Below this range, alleles are reference (0), and above it, they are alternative (1). 

The options for encoding heterozygous variants are 

<ul>
    <li>Drop</li>
    <li>Encode as the float AF value, not a binary</li>
    <li>Select a threshold to binarize them</li>
</ul>

Selection is made using a parameter in the `config.yaml` file. 

### Missing Data

Isolates with a lot of missingness are far more common than features with a lot of missingness because most of the sequenced regions have high mappability. 

A threshold of <b>1%</b> is being used to drop isolates, i.e. if more than 1% of an isolate's features for a given analysis are NA, then drop the isolate. 

Then, all remaining features with anything missing are dropped. Imputation can be done instead by setting the argument `impute` to True. This will impute every element in the matrix that is missing by averaging the feature, stratified by resistance phenotype. 

## Running the Analysis

Run the numbered scripts in order, with the `config.yaml` file. Arguments in the yaml file are as follows:
    
---
**NOTE**
    
drug: full drug name
drug_WHO_abbr: 3-letter drug name abbreviation in the WHO catalog
out_dir: output directory (the same for all analyses)
model_prefix: directory name for the current analysis
tiers_lst: list of tiers to include in the model
pheno_category_lst: list of phenotype categories to include
missing_thresh: threshold for missing rows/columns (0-1)
het_mode: how to handle heterozygous alleles
AF_thresh: 0.75
impute: boolean for whether missing values should be imputed (if False, then they will be dropped)
synonymous: boolean for whether synonymous variants should be included
pool_lof: boolean for whether or not LOF variants should be pooled
MAF: minor allele frequency for computing the genetic relatedness matrix
num_PCs: number of principal components (>= 0)
num_bootstrap: number of bootstrap samples
alpha: significance level
    
---
    
    
---
**NOTE**

Insert table of analyses.

---

## TO-DO:

1. Exclude positions in resistance-determining regions for genetic relatedness matrix calculation.
2. Visualize numbers of resistant vs. susceptible isolates with each variant.
3. Test for whether population structure correction significantly affects results.