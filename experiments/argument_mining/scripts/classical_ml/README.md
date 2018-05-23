
## Sentence-level scripts
### cross_validation.py
Performs a cross-validation for a strategy with fixed feature parameters on the training set
``` bash
  python3 experiments/argument_mining/scripts/classical_ml/cross_validation.py -subtask A -strategy unigram -c svm
  python3 experiments/argument_mining/scripts/classical_ml/cross_validation.py -subtask A -strategy character_ngrams -c svm
  python3 experiments/argument_mining/scripts/classical_ml/cross_validation.py -subtask A -strategy pos_distribution_spacy -c svm
  python3 experiments/argument_mining/scripts/classical_ml/cross_validation.py -subtask A -strategy sentiws_polarity -c svm
  python3 experiments/argument_mining/scripts/classical_ml/cross_validation.py -subtask A -embeddings_path data/word_embeddings/word2vec_wiki-de_20161120_100_binary -strategy embedding_centroid_100 -c svm
  python3 experiments/argument_mining/scripts/classical_ml/cross_validation.py -subtask A -embeddings_path data/word_embeddings/word2vec_wiki-de_20161120_100_binary -strategy embedding_centroid_stopwords_100 -c svm
  python3 experiments/argument_mining/scripts/classical_ml/cross_validation.py -subtask A -embeddings_path data/word_embeddings/word2vec_wiki-de_20161120_100_binary -strategy n_unigram+pos_distribution+embedding_centroid -c svm
  python3 experiments/argument_mining/scripts/classical_ml/cross_validation.py -subtask A -strategy unigram_lowercase_tfidf -c svm
  python3 experiments/argument_mining/scripts/classical_ml/cross_validation.py -subtask A -strategy n_unigram_shape -c svm
  python3 experiments/argument_mining/scripts/classical_ml/cross_validation.py -subtask A -strategy n_unigram_shape_lemma -c svm


```

### gridsearch.py
Performs a cross-validation on the training set with a feature combination and experiments with different parameters for the features. The result of the gridsearch is saved in the file system.
``` bash
    python3 experiments/argument_mining/scripts/classical_ml/gridsearch.py -subtask A -gridsearchstrategy unigram -c svm
    python3 experiments/argument_mining/scripts/classical_ml/gridsearch.py -subtask A -gridsearchstrategy bigram -c svm
    python3 experiments/argument_mining/scripts/classical_ml/gridsearch.py -subtask A -gridsearchstrategy pos_distribution_spacy -c svm
    python3 experiments/argument_mining/scripts/classical_ml/gridsearch.py -subtask A -gridsearchstrategy unigram+grammatical_spacy -c svm
    python3 experiments/argument_mining/scripts/classical_ml/gridsearch.py -subtask A -gridsearchstrategy character_ngrams -c svm
    python3 experiments/argument_mining/scripts/classical_ml/gridsearch.py -subtask B -gridsearchstrategy unigram+grammatical_spacy -c svm
    python3 experiments/argument_mining/scripts/classical_ml/gridsearch.py -subtask B -gridsearchstrategy n_unigram+shape -c svm
    python3 experiments/argument_mining/scripts/classical_ml/gridsearch.py -subtask A -embeddings_path data/word_embeddings/word2vec_wiki-de_20161120_100_binary -gridsearchstrategy embedding_centroid_100 -c svm
    python3 experiments/argument_mining/scripts/classical_ml/gridsearch.py -subtask A -embeddings_path data/word_embeddings/word2vec_wiki-de_20161120_100_binary -nfold 10 -gridsearchstrategy embedding_centroid_100 -c svm
    python3 experiments/argument_mining/scripts/classical_ml/gridsearch.py -subtask A -embeddings_path data/word_embeddings/word2vec_wiki-de_20161120_100_binary -nfold 10 -gridsearchstrategy unigram+embedding_centroid_100 -c svm
    
    python3 experiments/argument_mining/scripts/classical_ml/gridsearch.py -subtask C -gridsearchstrategy pos_distribution_spacy -c svm

```

### evaluate.py
Given a path to a settings file, the evaluate script trains the specified classiers on the training set and predicts on the test set.
``` bash
    python3 experiments/argument_mining/scripts/classical_ml/evaluate.py -configfile experiments/argument_mining/results/sentence/temp/XXX
```

