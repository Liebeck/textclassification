from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging
import numpy as np


def build():
    pipeline = Pipeline([('transformer',
                          SentiWSAveragePolarity()),
                         ])
    return ('polarity_sentiws_average', pipeline)


def extract_average_polarity(document):
    polarity_scores = []
    for token in document.tokens:
        if token.polarity_sentiws is not None:
            polarity_scores.append(token.polarity_sentiws)
    if not polarity_scores:
        return [0.0]
    else:
        return [np.mean(polarity_scores)]


class SentiWSAveragePolarity(BaseEstimator):
    def __init__(self):
        self.logger = logging.getLogger()

    def fit(self, X, y):
        return self

    def transform(self, X):
        transformed = list(map(lambda x: extract_average_polarity(x), X))
        return transformed
