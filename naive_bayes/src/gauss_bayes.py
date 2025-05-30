"""
This class assumes that all underlying data is
gaussian distributed
"""

import numpy as np
from collections import defaultdict


class GaussianNaiveBayes():
    def __init__(self):
        self.class_priors = None
        self.class_means = None
        self.class_vars = None
        self.classes = None
        self.epsilon = 1e-9

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # various outcomes
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.class_priors = np.zeros(n_classes, dtype=np.float64)
        self.class_means = np.zeros((n_classes, n_features), dtype=np.float64)
        self.class_vars = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]

            self.class_priors[idx] = X_c.shape[0] / float(n_samples)

            self.class_means[idx, :] = X_c.mean(axis=0) # mean for each feature
            self.class_vars[idx, :] = X_c.var(axis=0) + self.epsilon # variance for each feature

    def _gaussian_pdf(self, class_idx, x):
        # return the log likelihood for the gaussian of the class
        print(x)
        mean = self.class_means[class_idx]
        var = self.class_vars[class_idx]
        ll = -0.5 * np.sum(np.log(2 * np.pi * var) + (x - mean)**2/var)
        return ll
    
    def _predict_log_proba(self, x):
        # calculate all of the posteriors
        log_posteriors = []
        for idx, c in enumerate(self.classes):
            log_prior = np.log(self.class_priors[idx])
            ll = self._gaussian_pdf(idx, x)
            log_post = log_prior + ll
            log_posteriors.append(log_post)

        return np.array(log_posteriors)
    
    def predict_proba(self, X):
        """ Predict probabilities for each class for each sample in X """
        log_probas = np.array([self._predict_log_proba(x_sample) for x_sample in X])

        max_log_probas = np.max(log_probas, axis=1, keepdims=True)
        probas = np.exp(log_probas - max_log_probas)

        probas_sum = np.sum(probas, axis=1, keepdims=True)
        return probas / probas_sum

    def predict(self, X):
        log_probas = np.array([self._predict_log_proba(x) for x in X])

        print(log_probas)

        return self.classes[np.argmax(log_probas, axis=1)]