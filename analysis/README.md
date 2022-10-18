# Post-Model Fitting Analysis Steps

## 1. Lineage Analysis for Significant Principal Components

Principal components may have significant coefficients / odds ratios in the logistic regression models. Run the script <code>lineage_PC_analysis.py</code> to plot up to 2 significant eigenvectors, with isolates colored by their primary lineage (1-6 + <i>M. bovis</i>). 

The resulting two-dimensional plot shows the eigenvectors with the largest magnitudes of odds ratios. The eigenvectors represent the directions of variance and separate the data points.

## 2. Compare Logistic Regression to Associative Studies

The script in this 