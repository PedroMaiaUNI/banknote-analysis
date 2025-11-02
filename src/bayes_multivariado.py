# Implementação do classificador bayesiano multivariado

import numpy as np

class BayesMultivariado:
    def fit(self, X, y):
        self._classes = np.unique(y)
        self._mean = {}
        self._cov = {}
        self._priors = {}
        for c in self._classes:
            X_c = X[y == c]
            self._mean[c] = np.mean(X_c, axis=0)
            cov = np.cov(X_c.T)
            cov += np.eye(cov.shape[0]) * 1e-6
            self._cov[c] = cov
            self._priors[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        posteriors = []
        for c in self._classes:
            prior = np.log(self._priors[c])
            ll = self._log_likelihood(x, c)
            posteriors.append(prior + ll)
        return self._classes[np.argmax(posteriors)]

    def _log_likelihood(self, x, c):
        mean = self._mean[c]
        cov = self._cov[c]
        d = mean.shape[0]
        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)
        diff = (x - mean).reshape(-1, 1)
        exponent = -0.5 * float(diff.T @ cov_inv @ diff)
        log_likelihood = exponent - 0.5 * (d * np.log(2 * np.pi) + np.log(cov_det))
        return log_likelihood