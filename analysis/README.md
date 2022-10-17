# Compare Logistic Regression to Associative Studies

Associative: also known as catalog-based methods. 

See if logistic regression outperforms a univariate catalog-based method.

i.e. for a given variant, if it's OR > 1, assign all isolates without it to susceptible and all isolates with it to resistant. Can use the TP, FP, TN, and FN values that have already been computed for each variant. 