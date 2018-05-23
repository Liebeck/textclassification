import frameworks.tc_scikit.features.bag_of_words as bag_of_words
import frameworks.tc_scikit.features.character_embeddings as character_embeddings
import frameworks.tc_scikit.features.character_ngrams as character_ngrams
import frameworks.tc_scikit.features.dependency_distribution_spacy as dependency_distribution_spacy
import frameworks.tc_scikit.features.embedding_centroid as embedding_centroid
import frameworks.tc_scikit.features.long_word_count as long_word_count
import frameworks.tc_scikit.features.pos_distribution_spacy as pos_distribution_spacy
import frameworks.tc_scikit.features.sentiws_average_polarity_feature as sentiws_average_polarity_feature
import frameworks.tc_scikit.features.sentiws_polarity_bearing_tokens_feature as sentiws_polarity_bearing_tokens_feature
import frameworks.tc_scikit.features.sentiws_polarity_distribution as sentiws_polarity_distribution
import frameworks.tc_scikit.features.structural_features_spacy as structural_features_spacy
import frameworks.tc_scikit.features.textdepth_feature as textdepth_feature
import frameworks.tc_scikit.representations.stopwords as stopwords
from frameworks.tc_scikit.features.tfidf import build_tfidf

import frameworks.tc_scikit.features.lda_distribution as lda_distribution

STRATEGIES = {'unigram': [bag_of_words.build(ngram_range=(1, 1))],
              'unigram_stopwords': [bag_of_words.build(ngram_range=(1, 1), stopwords=stopwords.german_stopwords_nltk())],
              'unigram_lowercase': [bag_of_words.build(ngram_range=(1, 1), lowercase=True)],
              'unigram_iwnlp': [bag_of_words.build(ngram_range=(1, 1), token_form='IWNLP_lemma')],
              'unigram_iwnlp_lowercase': [bag_of_words.build(ngram_range=(1, 1), token_form='IWNLP_lemma', lowercase=True)],
              'unigram_frequency_test': [bag_of_words.build(ngram_range=1, min_df=3, max_features=None,
                                                            stopwords=stopwords.german_stopwords_nltk())],
              'character_ngrams': [character_ngrams.build(ngram_range=(3, 5), min_df=20)],
              'n_unigram': [bag_of_words.build(ngram_range=(1, 1), normalize=True)],
              'n_unigram_lowercase': [bag_of_words.build(ngram_range=(1, 1), lowercase=True, normalize=True)],
              'n_unigram_iwnlp': [bag_of_words.build(ngram_range=(1, 1), token_form='IWNLP_lemma', normalize=True)],
              'n_unigram_iwnlp_lowercase': [
                  bag_of_words.build(ngram_range=(1, 1), token_form='IWNLP_lemma', lowercase=True, normalize=True)],
              'unigram_lowercase_tfidf': [
                  build_tfidf(ngram_range=(1, 1))],
              'bigram': [bag_of_words.build(ngram_range=(2, 2))],
              'bigram_lowercase': [bag_of_words.build(ngram_range=(2, 2), lowercase=True)],
              'bigram_iwnlp': [bag_of_words.build(ngram_range=(2, 2), token_form='IWNLP_lemma')],
              'bigram_iwnlp_lowercase': [bag_of_words.build(ngram_range=(2, 2), token_form='IWNLP_lemma', lowercase=True)],
              'n_bigram': [bag_of_words.build(ngram_range=(2, 2), normalize=True)],
              'n_bigram_lowercase': [bag_of_words.build(ngram_range=(2, 2), lowercase=True, normalize=True)],
              'n_bigram_iwnlp': [bag_of_words.build(ngram_range=(2, 2), token_form='IWNLP_lemma', normalize=True)],
              'n_bigram_iwnlp_lowercase': [
                  bag_of_words.build(ngram_range=(2, 2), token_form='IWNLP_lemma', lowercase=True, normalize=True)],
              'unigram_bigram': [bag_of_words.build(ngram_range=(1, 1), feature_name='unigram'),
                                 bag_of_words.build(ngram_range=(2, 2), feature_name='bigram')],
              'dependency_distribution_spacy': [dependency_distribution_spacy.build()],
              'structural_spacy': [structural_features_spacy.build()],
              'structural_spacy_without_token_length': [structural_features_spacy.build(use_sentence_length=False)],
              'sentiws_polarity': [sentiws_average_polarity_feature.build(),
                                   sentiws_polarity_bearing_tokens_feature.build()],
              'sentiws_distribution': [sentiws_polarity_distribution.build(bins='auto')],
              'character_embeddings_centroid_100': [character_embeddings.build(embedding_length=100)],
              'embedding_centroid_100': [embedding_centroid.build(embedding_length=100)],
              'embedding_centroid_stopwords_100': [
                  embedding_centroid.build(embedding_length=100, stopwords=stopwords.german_stopwords_nltk())],
              'n_unigram+pos_distribution+embedding_centroid': [
                  bag_of_words.build(ngram_range=(1, 1), normalize=True, stopwords=stopwords.german_stopwords_nltk()),
                  embedding_centroid.build(embedding_length=100, stopwords=stopwords.german_stopwords_nltk())],
              'pos_distribution_spacy': [pos_distribution_spacy.build()],
              'pos_distribution_spacy_universal': [pos_distribution_spacy.build(coarse_grained=False)],
              'textdepth_feature': [textdepth_feature.build()],
              'lda_distribution': [lda_distribution.build()],
              'n_unigram+lda_distribution': [
                  bag_of_words.build(ngram_range=(1, 1), normalize=True, stopwords=stopwords.german_stopwords_nltk()),
                  lda_distribution.build()],
              'n_unigram_shape': [bag_of_words.build(ngram_range=(1, 1), token_form='shape', normalize=True)],
              'n_unigram_shape_lemma': [bag_of_words.build(ngram_range=(1, 1), token_form='shape_lemma', normalize=True)],
              'long_word_count': [long_word_count.build(length=3)],
              }
