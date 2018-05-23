import logging

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from spacy.orth import word_shape

from frameworks.tc_scikit.transformers.normalizer_toggle import NormalizerToggle


def build(feature_name='bag_of_words', ngram_range=(1, 1), token_form='text', lowercase=False, normalize=False,
          min_df=1, max_features=None, stopwords=None):
    pipeline = Pipeline(
        [('transformer',
          BagOfWords(ngram_range=ngram_range, token_form=token_form, lowercase=lowercase,
                     min_df=min_df,
                     max_features=max_features, stopwords=stopwords)),
         ('normalizer', NormalizerToggle(use_normalize=normalize))
         ])
    return (feature_name, pipeline)


def tokenizer_words(document):
    return list(map(lambda token: token.text, document.tokens))


def tokenizer_words_lowercase(document):
    return list(map(lambda token: token.text.lower(), document.tokens))


def tokenizer_shape(document):
    return list(map(lambda token: token.spacy_shape, document.tokens))


def tokenizer_shape_lemma(document):
    words = []
    for token in document.tokens:
        if token.iwnlp_lemma is not None and len(token.iwnlp_lemma) == 1:
            words.append(word_shape(token.iwnlp_lemma[0]))
        else:
            words.append(token.spacy_shape)
    return words


def tokenizer_lemma(sentence):
    words = []
    for token in sentence.tokens:
        if token.iwnlp_lemma is not None and len(token.iwnlp_lemma) == 1:
            words.append(token.iwnlp_lemma[0])
        else:
            words.append(token.text)
    return words


def tokenizer_lemma_lowercase(sentence):
    words = []
    for token in sentence.tokens:
        if token.iwnlp_lemma is not None and len(token.iwnlp_lemma) == 1:
            words.append(token.iwnlp_lemma[0].lower())
        else:
            words.append(token.text.lower())
    return words


class BagOfWords(BaseEstimator):
    def __init__(self, ngram_range=(1, 1), token_form='text', lowercase=False, min_df=1,
                 max_features=None, stopwords=None):
        self.ngram_range = ngram_range
        self.token_form = token_form
        self.logger = logging.getLogger()
        self.lowercase = lowercase
        self.min_df = min_df
        self.max_features = max_features
        self.stopwords = stopwords

    def get_tokenizer(self):
        if self.token_form == 'text' and not self.lowercase:
            return tokenizer_words
        elif self.token_form == 'text' and self.lowercase:
            return tokenizer_words_lowercase
        elif self.token_form == 'IWNLP_lemma' and not self.lowercase:
            return tokenizer_lemma
        elif self.token_form == 'IWNLP_lemma' and self.lowercase:
            return tokenizer_lemma_lowercase
        elif self.token_form == 'shape':
            return tokenizer_shape
        elif self.token_form == 'shape_lemma':
            return tokenizer_shape_lemma

    def fit(self, X, y):
        tokenizer = self.get_tokenizer()
        self.vectorizer = CountVectorizer(tokenizer=tokenizer,
                                          ngram_range=self.ngram_range,
                                          min_df=self.min_df,
                                          max_features=self.max_features,
                                          stop_words=self.stopwords,
                                          lowercase=False)  # the lowercase is workaround for passing a custom class
        self.vectorizer.fit(X)
        self.logger.info("Created a vocabulary with length {}".format(len(self.vectorizer.get_feature_names())))
        return self

    def transform(self, X):
        transformed = list(map(lambda x: self.transform_sentence(x), X))
        transformed = np.concatenate(transformed, axis=0)
        return transformed

    def transform_sentence(self, thf_sentence):
        vectorized = self.vectorizer.transform([thf_sentence]).toarray()
        vectorized = vectorized.astype(np.float64)
        return vectorized
