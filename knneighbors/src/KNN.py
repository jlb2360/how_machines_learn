import numpy as np
import pandas as pd
from collections import Counter

class KNN():

    def fit(self, X, y):
        self.classes = np.unique(y)

        self.neighbors = X
        self.class_names = y

    def predict(self, x, k):
        """we will assume a euclidian distance"""
        predictions = []
        for x_i in x:
            diff = self.neighbors - x_i[np.newaxis]
            dist = np.sqrt(np.diagonal(diff @ diff.T))

            sorted_idx = np.argsort(dist)

            class_sorted = self.class_names[sorted_idx]

            class_count = dict(Counter(class_sorted[:k]))

            predictions.append(max(class_count, key=class_count.get))


        return predictions





