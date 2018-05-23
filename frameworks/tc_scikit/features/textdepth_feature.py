from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import logging


def build():
    pipeline = Pipeline([('transformer', TextDepthFeature()),
                         ])
    return ('textdepth_feature', pipeline)


class TextDepthFeature(BaseEstimator):
    """A THF corpus specific feature that integrates the depth of text content in a tree hierarchy"""
    def __init__(self):
        self.logger = logging.getLogger()

    def fit(self, X, y):
        return self

    def transform(self, X):
        transformed = list(map(lambda x: [x.textdepth], X))
        return transformed
