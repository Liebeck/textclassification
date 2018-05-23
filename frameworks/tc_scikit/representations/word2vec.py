import gensim
import logging


class Word2Vec:
    def __init__(self, model_path='data/word_embeddings/word2vec_wiki-de_20161120_100', text_type='lowercase'):
        self.model_path = model_path
        self.text_type = text_type
        self.logger = logging.getLogger()
        self.coverage = 0
        self.total_tokens = 0

    def load(self):
        self.logger.info("Loading model: {}".format(self.model_path))
        self.logger.info("text_type: {}".format(self.text_type))
        self.model = gensim.models.KeyedVectors.load(self.model_path)
        self.logger.info("Model loaded")

    def annotate_document(self, document):
        for token in document.tokens:
            self.total_tokens = self.total_tokens + 1
            key = token.get_key(self.text_type)
            if key in self.model.wv.vocab:
                self.coverage = self.coverage + 1
                token.embedding = self.model[key]

    def annotate_documents(self, documents):
        list(map(lambda sentence: self.annotate_document(sentence), documents))
        self.logger.info("Annotated Tokens {}/{}".format(self.coverage, self.total_tokens))
