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
import numpy.linalg as LA
from scipy.stats import multivariate_normal, rv_continuous, gamma
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


class MaskValueError(ValueError):

    def __str__(self):
        return self.message


class NoMissingError(MaskValueError):
    message = 'The weight matrix is too good! It should have a zero element.'


class ZeroColumnError(MaskValueError):

    def __init__(self, n):
        self.message = f'The weight matrix has {n} zero colomns!'


class ZeroRowError(MaskValueError):

    def __init__(self, n):
        self.message = f'The weight matrix has {n} zero rows!'


def _check(M, raise_error=False):
    try:
        if np.any(np.all(M ==0, axis=0)):
            N = np.nonzero(np.all(M ==0, axis=0))[0].size
            raise ZeroColumnError(N)
        elif np.any(np.all(M ==0, axis=1)):
            N = np.nonzero(np.all(M ==0, axis=1))[0].size
            raise ZeroRowError(N)
        elif np.all(M==1):
            raise NoMissingError()
    except ValueError as e:
        if isinstance(e, NoMissingError):
            raise e
        elif isinstance(e, ZeroColumnError):
            raise e
        elif raise_error:
            raise e
        else:
            print(e)


class MissingMixin:

    """Mixin class for missing value model

    It is Created with reference to the `TransformerMixin` class of scikit-learn.
    
    Attributes:
        init_impute_strategy (string|number): the strategy of initial imputing
        max_impute_iter (int): max iteration for imputing
        X_imputed_ (array): imputed data
    """

    max_impute_iter = 20
    init_impute_strategy = None

    def init_impute(self, X, missing_matrix=None, strategy='mean', check=False):
        """initalize to impute
        
        Args:
            X (2D array): the data
            missing_matrix (2D array with the same size of X): A missing matrix composed of 0s and 1s
                1 means available, 0 means missing.
            strategy (str|callable, optional): the method of initial imputing
        
        Returns:
            array: imputed data
        """

        X = X.copy()  # copy X, unless X may change
        if hasattr(self, 'X_imputed_'):
            return self.X_imputed_
        if missing_matrix is None:
            return X

        if check:
            _check(missing_matrix)

        mask = ~missing_matrix  # mask matrix
        if strategy == 'mean':
            for x, m in zip(X.T, mask.T):
                if np.any(m):
                    # at least one element is not missing
                    x[m] = np.mean(x[~m])
                else:
                    x[m] = X.mean()
        elif strategy == 'constant':
            X[mask] = X.mean()
        elif strategy == 'linear':
            X = np.array([linear_impute(x, m) for x, m in zip(X.T, missing_matrix.T)]).T
        elif isinstance(strategy, (int, float)):
            X[mask] = strategy
        elif isinstance(strategy, np.ndarray):
            assert X.shape == strategy.shape
            X[mask] = strategy[mask]
        elif callable(stragegy):
            X = stragegy(X, mask)
        elif strategy == 'none':
            pass
        else:
            raise 'No such initial imputing strategy!'
        self.X_imputed_ = X
        return X

    def transform(self, X_missing, missing_matrix=None):
        X = self.impute(X_missing, missing_matrix=missing_matrix)
        return self.full_transform(X)

    def fit_full_transform(self, X_missing, missing_matrix=None):
        X = self.fit_imptue(X_missing, missing_matrix=None)
        return self.full_transform(X)

    def full_transform(self, X):
        return super().transform(X)

    def fit(self, X_missing, missing_matrix):
        X_ = self.init_impute(X_missing, missing_matrix)
        self.__fit__(X_, missing_matrix)
        return self

    def __fit__(self, X_, missing_matrix):
        for k in range(self.max_impute_iter):
            self.full_fit(X_)
            X_r = self.full_reconstruct(X_)
            X_[~missing_matrix] = X_r[~missing_matrix]
        self.X_imputed_ = X_
        return self

    def impute(self, X_missing, missing_matrix=None, max_iter=1):
        X_ = self.init_impute(X_missing, missing_matrix)
        for _ in range(max_iter):
            X_ = self.full_reconstruct(X_)
            X_[missing_matrix] = X_missing[missing_matrix]
        self.X_imputed_ = X_
        return X_

    def full_fit(self, X):
        super().fit(X)
        return self

    def fit_imptue(self, X_missing, missing_matrix):
        self.fit(X_missing, missing_matrix)
        return self.impute(X_missing, missing_matrix)

    def reconstruct(self, X):
        if not hasattr(self, 'inverse_transform'):
            return super().transform(X)
        return super().inverse_transform(super().transform(X))

    def full_reconstruct(self, X=None):
        if X is None:
            X = self.X_imputed_
        if not hasattr(self, 'inverse_transform'):
            return self.full_transform(X)
        return super().inverse_transform(self.full_transform(X))


def shur(Sigma, r):
    return Sigma[~r][:,~r] - Sigma[~r][:,r] @ np.linalg.lstsq(Sigma[r][:,r], Sigma[r][:,~r], rcond=None)[0]


def cond_gauss(x, Sigma, r):
    mu = np.dot(Sigma[~r][:,r], np.linalg.lstsq(Sigma[r][:,r], x[r], rcond=None)[0])
    x[~r] = mu
    # S = shur(Sigma, r)
    # x[~r] = multivariate_normal(mu, S).rvs()
    return x


class GaussianMissingMixin(MissingMixin):

    max_iter = 10

    def fit(self, X_missing, missing_matrix=None, sample_weight=None):

        X = self.init_impute(X_missing, missing_matrix)

        N, p = X.shape

        # Gibbs sampling
        ## params. for priori distr.
        nu0 = 0.5
        Psi0 = 1

        sigma = 0.001
        nu = nu0
        for _ in range(self.max_iter):

            self.full_fit(X)
            Y = self.full_transform(X)

            Psi = Psi0 + self.explained_variance_/2
            Sigma = gamma(nu, Psi).rvs()
            V = self.components_
            Sigma_x = (V.T * (Sigma+sigma)) @ V

            X = self.cond_gauss(X, Sigma_x, missing_matrix)

        self.X_imputed_ = X
        self.Sigma_Psi_ = Psi
        self.Sigma_nu_ = nu

        return self

    def impute(self, X_missing, missing_matrix):
        X = self.init_impute(X_missing, missing_matrix)
        V = self.components_
        Sigma_x = (V.T * (Sigma+sigma)) @ V
        return self.cond_gauss(X, Sigma_x, missing_matrix)

    def cond_gauss(self, X, Sigma, missing_matrix):
        X -= self.mean_
        try:
            for x, r in zip(X, missing_matrix):
                x = cond_gauss(x, Sigma, r)
        except Exception as e:
            print(f"Catch the exception: {e}. Just ignore it.")
        return X + self.mean_

    def full_fit(self, X):
        return super().fit(X)


class MissingPCA(GaussianMissingMixin, PCA):
    pass


def linear_impute(x, missing_matrix=None, boundary=None):
    """Easy completion

    Arguments:
        x: array
        missing_matrix: None | array (with the same shape to x)

    Example
    >>> linear_impute([0,1, 0,8,0,0,2,0], boundary=True)
    >>> [-2.5  1.   4.5  8.   6.   4.   2.   0. ]

    Using scipy:
        import scipy.interpolate as si
        ind = missing_matrix.nonzero()[0]
        f=si.interp1d(ind, x[ind], fill_value='extrapolate')
        x = f(np.arange(len(x))
    """

    x = np.asarray(x, dtype='float64')
    if missing_matrix is None: missing_matrix = (x!=0)
    assert np.any(missing_matrix), f'linear impute is not suitable for x={x}!'

    ind = missing_matrix.nonzero()[0]
    L = len(x)
    if len(ind)==1:
        x[:] = x[ind[0]]
        return x
    if ind[0] > 0:
        i0, i1 = ind[0], ind[1]
        if boundary:
            dx = (x[i1] - x[i0]) / (i1 - i0)
            x[:i0] = x[i0] - np.arange(1, i0+1)*dx
        else:
            x[:i0] = x[i0]
    if ind[-1] < L-1:
        i1, i0 = ind[-1], ind[-2]
        if boundary:
            dx = (x[i1] - x[i0]) / (i1 - i0)
            x[i1+1:] = x[i1] + np.arange(1, L-i1)*dx
        else:
            x[i1+1:] = x[i1]
    for i, j in zip(ind[:-1],ind[1:]):
        d = j - i
        if d > 1:
            dx = (x[j] - x[i]) / (j - i)
            x[i+1:j] = x[i] + np.arange(1, d) *dx
    return x
