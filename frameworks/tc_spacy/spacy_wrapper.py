import logging
import spacy
from spacy_iwnlp import spaCyIWNLP
from spacy_sentiws import spaCySentiWS
from frameworks.tc_scikit.models.token import Token
from frameworks.tc_scikit.models.dependency import Dependency


class SpacyWrapper(object):
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug('Loading Spacy model')
        self.nlp = spacy.load('de')
        self.nlp.add_pipe(spaCyIWNLP(lemmatizer_path='data/IWNLP/IWNLP.Lemmatizer_20170501.json'))
        self.nlp.add_pipe(spaCySentiWS(sentiws_path='data/sentiws/'))
        self.logger.debug('Spacy loaded')

    def process_document(self, text):
        result = self.nlp(text)
        tokens = []
        dependencies = []
        for index, token in enumerate(result):
            token_model = Token(index + 1,
                                text=token.text,
                                spacy_pos_stts=token.tag_,
                                spacy_pos_universal_google=token.pos_,
                                iwnlp_lemma=token._.iwnlp_lemmas,
                                spacy_ner_type=token.ent_type_,
                                spacy_ner_iob=token.ent_iob_,
                                spacy_is_punct=token.is_punct,
                                spacy_is_space=token.is_space,
                                spacy_like_num=token.like_num,
                                spacy_like_url=token.like_url,
                                spacy_shape=token.shape_,
                                polarity_sentiws=token._.sentiws)
            tokens.append(token_model)
            dependency_model = Dependency(token.i + 1, token.dep_, token.head.i + 1)
            dependencies.append(dependency_model)
        return {
            'tokens': tokens,
            'dependencies': dependencies
        }
