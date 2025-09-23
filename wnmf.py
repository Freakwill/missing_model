#!/usr/bin/env python

"""
Weighted NMF
"""

import numpy as np
from scipy.stats import poisson
from scipy.special import kl_div
from sklearn.metrics import r2_score
from sklearn.decomposition import NMF
from sklearn.utils.validation import check_is_fitted

from missing_model import MissingMixin


def divergence(x, mu, *args, **kwargs):
    # generalized KL divergence
    return np.sum(kl_div(x, mu), *args, **kwargs)


def re_divergence(x, mu, *args, **kwargs):
    # relative KL divergence
    return divergence(x, mu) / np.sum(x, *args, **kwargs)


class MFMixin:
    # mixin class for matrix factorization

    def __init__(self, mu_alpha=0.1, alpha_H=0.25, alpha_W=0.25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu_alpha = mu_alpha
        self.alpha_H = alpha_H
        self.alpha_W = alpha_W

    def errors(self, X, rec=None, weight=None, p=1, *args, **kwargs):
        # assume X >=0
        if rec is None:
            check_is_fitted(self, ('coefs_', 'components_'))
            rec = self.reconstruct(*args, **kwargs)
        return np.mean(np.abs(X - rec)**p*weight, axis=1) / np.mean(X**p*weight, axis=1)

    def error(self, *args, **kwargs):
        return np.mean(self.errors(*args, **kwargs))

    def sort(self, total):
        # sort the components according to their significances
        s = np.sum(self.coefs_, axis=0)
        t = np.sum(self.components_, axis=1)
        self.significance_ = s * t / total

        index = np.argsort(-self.significance_)
        self.significance_ = self.significance_[index]

        self.normal_components_ = (self.components_ / t[:,None])[index]
        self.components_ = self.components_[index]
        self.normal_coefs_ = (self.coefs_ / s)[:, index]
        self.coefs_ = self.coefs_[:, index]
        return self

    def reconstruct(self, X=None, *args, **kwargs):
        if X is not None:
            Z = self.transform(X)
        else:
            Z = self.coefs_
        return self.inverse_transform(Z, *args, **kwargs)

    def inverse_transform(self, Z, to_int=None, n_components=None, significance=None):
        """Reconstruct the whole data
        
        Parameters
        ----------
        to_int : None, optional
            post-processing of the reconstructing result
            `None`: no post-processing
            `True|m`: mode of Poisson distribution
            `p`: Sample as a Poisson vr
            callable: apply in the result
        n_components : None, optional
            the number of components used for reconstruction
        
        Returns
        -------
        array
            the reconstruted data
        """
        
        check_is_fitted(self, ('coefs_', 'components_'))
        if n_components is None:
            if significance is None:
                rec = np.dot(Z, self.components_)
            else:
                s = 0
                for n, s_ in enumerate(self.significance_, 1):
                    s += s_
                    if s >= significance: break
                n_components = n
                rec = np.dot(Z[:, :n_components], self.components_[:n_components])
        else:
            assert isinstance(n_components, int) and 0 < n_components <= self.n_components_, \
             f'n_components should be an integer from 0 to {self.n_components_}!'
            rec = np.dot(Z[:, :n_components], self.components_[:n_components])
        if to_int is None:
            return rec
        elif to_int is True or to_int=='m':
            return np.floor(rec).astype(dtype=np.int_)
        elif to_int == 'p':
            return poisson(rec).rvs()
        elif callable(to_int):
            return to_int(rec)
        else:
            return np.floor(rec)

    def generate(self, rec=None):
        if rec is None:
            check_is_fitted(self, ('coefs_', 'components_'))
            rec = self.reconstruct()
        return poisson(rec).rvs()

    def rll(self, X, rec=None, *args, **kwargs):
        # relative likelihood

        if rec is None:
            check_is_fitted(self, ('coefs_', 'components_'))
            rec = self.reconstruct(*args, **kwargs)

        def _rll(x, mean):
            # relative likelihood: P(x|mean) / Pmax(mean)
            mode = np.floor(mean)
            return  np.mean(poisson.logpmf(x, mean) - poisson.logpmf(mode, mean))
        return _rll(X, rec)


class MyNMF(MFMixin, NMF):

    def fit_transform(self, X, *args, **kwargs):
        self.coefs_ = W = super().fit_transform(X, **kwargs)
        self.sort(total=np.sum(X))
        return W


class WeightedNMF(MFMixin, NMF):

    def __init__(self, mu_beta=0, weight=None, **kwargs):
        super().__init__(**kwargs)
        self.mu_beta = mu_beta
        self.weight = weight

    def _fit_transform(self, X, weight=None, W=None, H=None, update_H=True):
        # initalize W, H

        self._check_params(X)

        if weight is None:
            weight = self.weight or np.ones_like(X)

        W, H = self._check_w_h(X, W, H, update_H)

        # weighted MU
        W += 0.01
        H += 0.01

        if weight is None:
            super().fit(X)
        else:
            for _ in range(self.max_iter - 1):
                H, W = _update(W, H, X, weight, self.mu_alpha, self.mu_beta, self.alpha_H, self.alpha_W)

        self.components_ = H
        self.coefs_ = W

        self.sort(total=np.sum(X))
        return W, H

    def transform(self, X, weight=None, W=None, H=None):
        W, H = self._fit_transform(X, weight=weight, W=W, H=H)

        return W

    def fit(self, *args, **kwargs):

        W, H = self._fit_transform(*args, **kwargs)

        # self.reconstruction_err_ = _beta_divergence(
        #     X, W, H, self._beta_loss, square_root=True
        # )

        self.n_components_ = H.shape[0]
        self.components_ = H
        return self


class ImputingNMF(WeightedNMF, MissingMixin):

    """Missing data imputing by NMF
    
    Attributes
    ----------
    init_impute_strategy : str
        initial imputing strategy
    strategy : str
        imputing strategy
    """
    
    def __init__(self, mu_beta=0.1, init_impute_strategy='mean', **kwargs):
        super().__init__(**kwargs)
        self.mu_beta = mu_beta
        self.init_impute_strategy = init_impute_strategy
        self.strategy = 'nmf'

    @property
    def missing_matrix(self):
        return self.weight

    def _fit_transform(self, X, missing_matrix=None, *args, **kwargs):
        """to fit the NMF model for incompleted data
        
        Parameters
        ----------
        X : array
            the training data
        missing_matrix : None, optional
            The missing matrix
        
        Returns
        -------
        array
            The transforming result
        """

        if missing_matrix is None:
            missing_matrix = self.missing_matrix or X!=0

        # initial completion
        if hasattr(self, 'init_completion_') and self.init_completion_ is not None:
            M = self.init_completion_
        else:
            M = self.init_impute(X=X, missing_matrix=missing_matrix, strategy=self.init_impute_strategy)
        if self.strategy == 'nmf':
            return super()._fit_transform(M, weight=missing_matrix, *args, **kwargs)
        elif self.strategy == 'ae':
            X = M
            for _ in range(10):
                self.fit(X)
                X[~missing_matrix] = self.reconstruct()[~missing_matrix]
            return X

    def impute(self, X, missing_matrix=None, *args, **kwargs):
        check_is_fitted(self, ('coefs_', 'components_'))

        if missing_matrix is None: missing_matrix = (X != 0)

        X_rec = self.reconstruct(*args, **kwargs)
        X_rec[missing_matrix] = X[missing_matrix]
        return X_rec

    # def error(self, X, X_rec=None, missing_matrix=None, p=1, *args, **kwargs):
    #     # X >=0
    #     if X_rec is None:
    #         check_is_fitted(self, ('coefs_', 'components_'))
    #         X_rec = self.reconstruct(*args, **kwargs)
    #     return np.mean(missing_matrix_filter(np.abs(X*missing_matrix - X_rec*missing_matrix)**p / (X*missing_matrix)**p, missing_matrix))

    def errors(self, X, X_rec=None, missing_matrix=None, p=1, *args, **kwargs):
        # assume X >=0
        if X_rec is None:
            check_is_fitted(self, ('coefs_', 'components_'))
            X_rec = self.reconstruct(*args, **kwargs)
        return np.mean(np.abs(X *missing_matrix- X_rec*missing_matrix)**p, axis=1) / np.mean((X*missing_matrix)**p, axis=1)

    def error(self, *args, **kwargs):
        return np.mean(self.errors(*args, **kwargs))

    # def _fit_transform(self, X, missing_matrix=None, W=None, H=None, update_H=True, log_config=None):
    #     # initalize W, H

    #     W, H = self._check_w_h(X, W, H, update_H)

    #     # weighted MU
    #     W += 0.01
    #     H += 0.01
        
    #     if log_config:
    #         test_errors = []
    #         train_errors = []
    #         if 'real' in log_config:
    #             X_real = log_config['real']
    #         else:
    #             raise Exception('Have not provide argument `real` in `log_config`')
    #     if missing_matrix is None:
    #         missing_matrix = self.missing_matrix or X!=0
  
    #     for _ in range(self.max_iter - 1):
    #         H, W = update(W, H, X, missing_matrix, self.mu_alpha, self.mu_beta, self.alpha_H, self.alpha_W)

    #         if log_config:
    #             if _ % log_config.get('period', 2)==0:
    #                 _error = self.error(X=X_real, X_rec=W @ H, missing_matrix=log_config['train_missing_matrix'])
    #                 train_errors.append(_error)
    #                 _error = self.error(X=X_real, X_rec=W @ H, missing_matrix=log_config['test_missing_matrix'])
    #                 test_errors.append(_error)

    #     self.components_ = H
    #     self.coefs_ = W

    #     self.sort(total=np.sum(X))
    #     if log_config:
    #         return W, train_errors, test_errors
    #     return W, H


# def update(W, H, X, weight, alpha=0, beta=0):
#     # Weighted MU rule based on Frobenius norm
#     WH = W @ H
#     H *= (W.T @ (X * weight) + alpha) / (W.T @ (WH * weight) + alpha)
#     WH = W @ H
#     W *= ((X * weight) @ H.T + alpha) / ((WH * weight) @ H.T + alpha)
#     H[np.isnan(H)] = 0; W[np.isnan(W)] = 0
#     return H, W


# def update(W, H, X, weight, alpha=0, beta=0):
#     # Weighted SGD rule based on divergence

#     weight_X = weight * X
#     WH = W @ H; WH =np.maximum(WH, 0.0001)
#     H += 0.1*((W.T @ (weight_X / WH))- (W.T @ weight))
#     WH = W @ H; WH =np.maximum(WH, 0.0001)
#     W += 0.1*(((weight_X / WH) @ H.T)) - (weight  @ H.T)
#     H[np.isnan(H)] = 0; W[np.isnan(W)] = 0
#     W = np.maximum(W, 0)
#     H = np.maximum(H, 0)
#     return H, W


def _update(W, H, X, weight, alpha=0, beta=0, alpha_H=0.01, alpha_W=0.01):
    # Weighted MU rule based on divergence

    weight_X = weight * X
    WH = W @ H; WH =np.maximum(WH, 0.0001)
    Hr = H
    H *= ((W.T @ (weight_X / WH)) + alpha) / (W.T @ (weight + (1-weight) * beta / WH) + alpha + alpha_H * Hr)
    WH = W @ H; WH =np.maximum(WH, 0.0001)
    Wr = W
    W *= (((weight_X / WH) @ H.T) + alpha) / ((weight + (1-weight) * beta / WH) @ H.T + alpha + alpha_W * np.mean(W,axis=1)[:, None])
    H[np.isnan(H)] = 0; W[np.isnan(W)] = 0
    return H, W


# def update(W, H, X, weight, alpha=0.01, beta=1):
#     # Weighted MU rule based on divergence

#     weight_X = weight * X
#     WH = W @ H; WH =np.maximum(WH, 0.0001)
#     H *= ((W.T @ (weight_X / WH)) + alpha) / (W.T @ (weight * (X +beta) / (WH + beta)) + alpha)
#     WH = W @ H; WH =np.maximum(WH, 0.0001)
#     W *= ((weight_X / WH) @ H.T + alpha) / ((weight * (X +beta) / (WH + beta)) @ H.T + alpha)
#     H[np.isnan(H)] = 0; W[np.isnan(W)] = 0
#     return H, W


# class ZIPNMF(MFMixin, NMF):

#     def fit_transform(self, X, *args, **kwargs):
        
#         alpha = 0.01

#         W = super().fit_transform(X * weight, *args, **kwargs)
#         H = self.components_
#         W += (W + 1) * np.random.rand(*W.shape) * 0.01
#         H += (H + 1) * np.random.rand(*H.shape) * 0.01
#         for _ in range(self.max_iter):
#             H1, W1, mu1 = update(W, H, mu, X, alpha)
#             H, W, mu = H1, W1, mu1
#         self.components_ = H
#         self.coefs_ = W

#         s = np.sum(self.coefs_, axis=0)
#         t = np.sum(self.components_, axis=1)
#         total = np.sum(X)
#         self.significance_ = s * t / total

#         # self.sort()

#         return self.coefs_


class ImputingPF(ImputingNMF):
    # imputing by PF

    def fit_transform(self, X, missing_matrix=None, *args, **kwargs):
        """to fit the NMF model for incompleted data
        
        Parameters
        ----------
        X : array
            the training data
        missing_matrix : None, optional
            if it is None, then it is set to be X!=0, namely,
            0-values are treated as missing data by default
        *args, **kwargs
            the parameters of parent method
        
        Returns
        -------
        array
            The transforming result
        """

        if missing_matrix is None:
            missing_matrix = self.missing_matrix or X!= 0
      
        # initial completion
        if hasattr(self, 'init_completion_') and self.init_completion_ is not None:
            M = self.init_completion_
        else:
            M = self.init_impute(X=X, missing_matrix=missing_matrix, strategy=self.init_impute_strategy)

        X = M
        for _ in range(15):
            super().fit(X)
            X[~missing_matrix] = self.reconstruct(to_int='p')[~missing_matrix]
        return X


# def linear_impute(x, missing_matrix=None, boundary=None):
#     """Easy completion

#     Arguments:
#         x: array
#         missing_matrix: None | array (with the same shape to x)

#     Example
#     >>> linear_impute([0,1, 0,8,0,0,2,0], boundary=True)
#     >>> [-2.5  1.   4.5  8.   6.   4.   2.   0. ]

#     Using scipy:
#         import scipy.interpolate as si
#         ind = missing_matrix.nonzero()[0]
#         f=si.interp1d(ind, x[ind], fill_value='extrapolate')
#         x = f(np.arange(len(x))
#     """

#     x = np.asarray(x, dtype='float64')
#     if missing_matrix is None: missing_matrix = (x!=0)
#     assert np.any(missing_matrix), 'x is bad!'

#     ind = missing_matrix.nonzero()[0]
#     L = len(x)
#     if len(ind)==1:
#         x[:] = x[ind[0]]
#         return x
#     if ind[0] > 0:
#         i0, i1 = ind[0], ind[1]
#         if boundary:
#             dx = (x[i1] - x[i0]) / (i1 - i0)
#             x[:i0] = x[i0] - np.arange(1, i0+1)*dx
#         else:
#             x[:i0] = x[i0]
#     if ind[-1] < L-1:
#         i1, i0 = ind[-1], ind[-2]
#         if boundary:
#             dx = (x[i1] - x[i0]) / (i1 - i0)
#             x[i1+1:] = x[i1] + np.arange(1, L-i1)*dx
#         else:
#             x[i1+1:] = x[i1]
#     for i, j in zip(ind[:-1],ind[1:]):
#         d = j - i
#         if d > 1:
#             dx = (x[j] - x[i]) / (j - i)
#             x[i+1:j] = x[i] + np.arange(1, d) *dx
#     return x
