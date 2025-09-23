#!/usr/bin/env python3

"""
X = [X1, X2] ~ N(0, V Sigma V' + sigma^2), Sigma ~ IW (inverse Wishart)
or Z = X V ~ N(0, Sigma)
or X = Z V^T + sigma^2
Sigma ~ IW(nv0, Psi0)

Gibbs sampling:
    0. initalize Sigma V by all samples (set hidden values as 0s)
    1. calculate mu', Sigma' from X1 and Cov matrix of X
      X2 ~ N(mu', Sigma')
    2. do PCA for X imputed in 1-step, and get Z
    3. calculate nv, Psi from Z and priori parameters
      Sigma ~ IW(nv, Psi)
"""

import numpy as np
from utils import logit, expit
from scipy.stats import multivariate_normal, rv_continuous, gamma

from missing_model import MissingPCA, MissingMixin, GaussianMissingMixin, cond_gauss
from nlpca import NLPCA


class MissingNLPCA(GaussianMissingMixin, NLPCA):
    max_impute_iter = 20

    def fit(self, X_missing, R=None, sample_weight=None):

        X = self.init_impute(X_missing, R)
        N, p = X.shape

        # Gibbs sampling
        ## params. for priori distr.
        nu0 = 0.5
        Psi0 = 1

        sigma = 0.001
        nu = nu0
        for _ in range(self.max_impute_iter):

            self.full_fit(X)
            Y = self.full_transform(X)

            Psi = Psi0 + np.var(Y, axis=0)/2
            Sigma = gamma(nu, Psi).rvs()
            V = self.components_
            Sigma_x = (V.T * (Sigma+sigma)) @ V
            
            X = self.cond_gauss(X, Sigma_x, R)

        self.X_imputed_ = X
        self.Sigma_Psi_ = Psi
        self.Sigma_nu_ = nu

        return self

    def full_fit(self, X):
        return NLPCA.fit(self, X)


if __name__ == "__main__":

    from datasets import fashion_missing
    import joblib

    X, mask, test_index, size = fashion_missing(channel=0)
    N, p = X.shape

    def _pca(X):
        filename = 'mising-nlpca-model.joblib'
        try:
            raise
            pca = joblib.load(filename)
        except:
            pca = MissingNLPCA(n_components=12)
            logX = logit(X)
            pca.fit(logX, mask)
            joblib.dump(pca, filename)

        X_ = pca.X_imputed_
        X_ = expit(X_)
        return X_


    X_ = _pca(X)


    X_mask = X * mask
    Xs = [[X[i].reshape(size),
        X_mask[i].reshape(size),
        X_[i].reshape(size)] for i in test_index]

    from utils import make_grid_image

    im = make_grid_image(Xs, 255)
    # im.save(f'../src/missing-nlpca.png')
    im.show()
