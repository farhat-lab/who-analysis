# World Health Organization TB Resistance Mutation Catalog, 2022

## Genotype Annotations

<ul>
    <li>resolved_symbol: gene name</li>
    <li>variant_category</li>
    <ul>
        <li>p: coding variants</li>
        <li>c: synonymous or upstream variants</li>
        <li>n: non-coding variants in <i>rrs/rrl</i></li>
        <li>deletion: large-scale deletion of a gene<li>
    </ul>
    <li>Effect</li>
    <ul>
        <li>upstream_gene_variant</li>
        <li>missense_variant, synonymous_variant, inframe_deletion, inframe_insertion, stop_lost: self-explanatory</li>
        <li>lof: any frameshift, large-scale deletion, nonsense, or loss of start mutation
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

FIRST, such isolates are dropped. NEXT, all features with anything missing are dropped. <i>Can change this to allow for imputation</i>. 


### Make model inputs

Run the script `01_make_model_inputs.py`

---
**NOTE**

Insert table of analyses that we will run. Maha said she would update it

---



## TO-DO:

Visualize numbers of resistant vs. susceptible isolates with each variant. This was suggested in a meeting, still determining how best to do this.