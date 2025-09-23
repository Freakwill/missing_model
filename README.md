# missing_model
model of missing data; data imputation

## Use

```python
from missing_model import MissingPCA

mpca = MissingPCA()
mpca.fit(X, R) # R is the missing matrix
X_imputed = mpca.impute(X)
```

## Cautions

In general case, missing model based on PCA is better then based on NMF, since the former considers the joint distribution of the sample.
If X (design matrix) is non-negative, then it is recommended to use missing model based on NMF, or use logit/expit to transform the data.
