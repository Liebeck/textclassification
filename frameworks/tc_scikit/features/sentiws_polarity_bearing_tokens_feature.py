from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging


def build():
    pipeline = Pipeline([('transformer',
                          SentiWSPolarityBearingTokens()),
                         ])
    return ('polarity_sentiws_polarity_bearing_tokens', pipeline)


def count_polarity_bearing_tokens(document):
    token_count = 0
    for token in document.tokens:
        if token.polarity_sentiws is not None:
            token_count += 1
    return [token_count]


class SentiWSPolarityBearingTokens(BaseEstimator):
    def __init__(self):
        self.logger = logging.getLogger()

    def fit(self, X, y):
        return self

    def transform(self, X):
        transformed = list(map(lambda x: count_polarity_bearing_tokens(x), X))
        return transformed
