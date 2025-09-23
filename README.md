# missing_model
model of missing data; data imputation

## Use

```python
from missing_model import MissingPCA

mpca = MissingPCA()
mpca.fit(X, R) # R is the missing matrix
X_imputed = mpca.impute(X)
```
